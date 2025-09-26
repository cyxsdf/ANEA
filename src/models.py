import torch
from torch import nn
import torch.nn.functional as F

from modules.unimf import MultimodalTransformerEncoder
from modules.transformer import TransformerEncoder
from transformers import BertTokenizer, BertModel


class DenoisingAE(nn.Module):
    """去噪自编码器模块，用于提取模态不变特征"""

    def __init__(self, input_dim=128, hidden_dim=64):
        super(DenoisingAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)  # 去噪dropout，核心组件
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )

    def forward(self, feat_m):
        # 添加高斯噪声（模拟数据扰动）
        noise = torch.randn_like(feat_m) * 0.1
        noisy_feat = feat_m + noise
        # 编码→解码过程
        latent = self.encoder(noisy_feat)
        recon_feat = self.decoder(latent)
        # 计算重建损失
        recon_loss = F.mse_loss(recon_feat, feat_m)
        return latent, recon_loss  # 返回不变特征和重建损失


class Discriminator(nn.Module):
    """模态判别器，使用CNN+Transformer结构区分真实模态和生成模态"""

    def __init__(self, input_dim, seq_len, hidden_dim=256, num_layers=2, num_heads=4, kernel_sizes=[3, 5, 7]):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_kernels = len(kernel_sizes)
        self.num_heads = num_heads

        self.out_channels_per_kernel = hidden_dim // self.num_kernels
        remaining_channels = hidden_dim % self.num_kernels

        self.conv_layers = nn.ModuleList()
        for i, kernel_size in enumerate(kernel_sizes):
            out_channels = self.out_channels_per_kernel + (remaining_channels if i == self.num_kernels - 1 else 0)
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=input_dim,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.LeakyReLU(0.2),
                    nn.MaxPool1d(kernel_size=2, stride=2)
                )
            )

        self.cnn_output_len = seq_len // 2
        self.cnn_output_dim = hidden_dim

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.cnn_output_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.pos_encoder = nn.Embedding(self.cnn_output_len, self.cnn_output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        cnn_outputs = []
        for conv in self.conv_layers:
            cnn_out = conv(x)
            cnn_outputs.append(cnn_out)

        cnn_combined = torch.cat(cnn_outputs, dim=1)
        if cnn_combined.size(1) != self.cnn_output_dim:
            projection = nn.Linear(cnn_combined.size(1), self.cnn_output_dim, device=cnn_combined.device)
            cnn_combined = projection(cnn_combined.transpose(1, 2)).transpose(1, 2)

        transformer_input = cnn_combined.transpose(1, 2)
        batch_size, seq_len, _ = transformer_input.shape
        positions = torch.arange(seq_len, device=transformer_input.device).unsqueeze(0).expand(batch_size, -1)
        transformer_input += self.pos_encoder(positions)

        transformer_out = self.transformer_encoder(transformer_input)
        last_out = transformer_out[:, -1, :]
        pred = self.classifier(last_out)
        return pred


class TRANSLATEModel(nn.Module):
    def __init__(self, hyp_params, missing=None):
        super(TRANSLATEModel, self).__init__()
        if hyp_params.dataset == 'meld_senti' or hyp_params.dataset == 'meld_emo':
            self.l_len, self.a_len = hyp_params.l_len, hyp_params.a_len
            self.orig_d_l, self.orig_d_a = hyp_params.orig_d_l, hyp_params.orig_d_a
            self.v_len, self.orig_d_v = 0, 0
        else:
            self.l_len, self.a_len, self.v_len = hyp_params.l_len, hyp_params.a_len, hyp_params.v_len
            self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
        self.embed_dim = hyp_params.embed_dim
        self.num_heads = hyp_params.num_heads
        self.trans_layers = hyp_params.trans_layers
        self.attn_dropout = hyp_params.attn_dropout
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.trans_dropout = hyp_params.trans_dropout
        self.modalities = hyp_params.modalities
        self.missing = missing

        # 新增DnAE模块（根据模态设置输入维度）
        self.dnae_l = DenoisingAE(input_dim=self.orig_d_l,
                                  hidden_dim=self.embed_dim // 2) if 'L' in hyp_params.modalities else None
        self.dnae_a = DenoisingAE(input_dim=self.orig_d_a,
                                  hidden_dim=self.embed_dim // 2) if 'A' in hyp_params.modalities else None
        self.dnae_v = DenoisingAE(input_dim=self.orig_d_v, hidden_dim=self.embed_dim // 2) if (
                    'V' in hyp_params.modalities and hyp_params.dataset != 'meld_senti' and hyp_params.dataset != 'meld_emo') else None

        self.position_embeddings = nn.Embedding(max(self.l_len, self.a_len, self.v_len), self.embed_dim)
        self.modal_type_embeddings = nn.Embedding(4, self.embed_dim)

        self.multi = nn.Parameter(torch.Tensor(1, self.embed_dim))
        nn.init.xavier_uniform_(self.multi)

        self.translator = TransformerEncoder(embed_dim=self.embed_dim,
                                             num_heads=self.num_heads,
                                             lens=(self.l_len, self.a_len, self.v_len),
                                             layers=self.trans_layers,
                                             modalities=self.modalities,
                                             missing=self.missing,
                                             attn_dropout=self.attn_dropout,
                                             relu_dropout=self.relu_dropout,
                                             res_dropout=self.res_dropout)

        # 调整投影层维度以融合原始特征和DnAE提取的不变特征
        if 'L' in self.modalities or self.missing == 'L':
            l_input_dim = self.orig_d_l + (self.embed_dim // 2 if self.dnae_l else 0)
            self.proj_l = nn.Linear(l_input_dim, self.embed_dim)
        if 'A' in self.modalities or self.missing == 'A':
            a_input_dim = self.orig_d_a + (self.embed_dim // 2 if self.dnae_a else 0)
            self.proj_a = nn.Linear(a_input_dim, self.embed_dim)
        if 'V' in self.modalities or self.missing == 'V' and hyp_params.dataset != 'meld_senti' and hyp_params.dataset != 'meld_emo':
            v_input_dim = self.orig_d_v + (self.embed_dim // 2 if self.dnae_v else 0)
            self.proj_v = nn.Linear(v_input_dim, self.embed_dim)

        if self.missing == 'L':
            self.out = nn.Linear(self.embed_dim, self.orig_d_l)
        elif self.missing == 'A':
            self.out = nn.Linear(self.embed_dim, self.orig_d_a)
        elif self.missing == 'V':
            self.out = nn.Linear(self.embed_dim, self.orig_d_v)
        else:
            raise ValueError('Unknown missing modality type')

    def forward(self, src, tgt, phase='train', eval_start=False):
        dnae_loss = 0.0  # 初始化DnAE损失

        if self.modalities == 'L':
            if self.missing == 'A':
                x_l, x_a = src, tgt
                # 文本模态不变特征提取
                l_latent = None
                if self.dnae_l:
                    l_latent, l_loss = self.dnae_l(x_l)
                    dnae_loss += l_loss
                # 音频模态不变特征提取
                a_latent = None
                if self.dnae_a:
                    a_latent, a_loss = self.dnae_a(x_a)
                    dnae_loss += a_loss

                # 融合原始特征与不变特征
                x_l_combined = torch.cat([x_l, l_latent], dim=-1) if l_latent is not None else x_l
                x_a_combined = torch.cat([x_a, a_latent], dim=-1) if a_latent is not None else x_a

                # 投影到嵌入维度
                x_l = F.dropout(F.relu(self.proj_l(x_l_combined)), p=self.trans_dropout, training=self.training)
                x_a = F.dropout(F.relu(self.proj_a(x_a_combined)), p=self.trans_dropout, training=self.training)
                x_l = x_l.transpose(0, 1)
                x_a = x_a.transpose(0, 1)

            elif self.missing == 'V':
                x_l, x_v = src, tgt
                # 文本模态不变特征提取
                l_latent = None
                if self.dnae_l:
                    l_latent, l_loss = self.dnae_l(x_l)
                    dnae_loss += l_loss
                # 视频模态不变特征提取
                v_latent = None
                if self.dnae_v:
                    v_latent, v_loss = self.dnae_v(x_v)
                    dnae_loss += v_loss

                # 融合原始特征与不变特征
                x_l_combined = torch.cat([x_l, l_latent], dim=-1) if l_latent is not None else x_l
                x_v_combined = torch.cat([x_v, v_latent], dim=-1) if v_latent is not None else x_v

                # 投影到嵌入维度
                x_l = F.dropout(F.relu(self.proj_l(x_l_combined)), p=self.trans_dropout, training=self.training)
                x_v = F.dropout(F.relu(self.proj_v(x_v_combined)), p=self.trans_dropout, training=self.training)
                x_l = x_l.transpose(0, 1)
                x_v = x_v.transpose(0, 1)

            else:
                raise ValueError('Unknown missing modality type')

        elif self.modalities == 'A':
            if self.missing == 'L':
                x_a, x_l = src, tgt
                # 音频模态不变特征提取
                a_latent = None
                if self.dnae_a:
                    a_latent, a_loss = self.dnae_a(x_a)
                    dnae_loss += a_loss
                # 文本模态不变特征提取
                l_latent = None
                if self.dnae_l:
                    l_latent, l_loss = self.dnae_l(x_l)
                    dnae_loss += l_loss

                # 融合原始特征与不变特征
                x_a_combined = torch.cat([x_a, a_latent], dim=-1) if a_latent is not None else x_a
                x_l_combined = torch.cat([x_l, l_latent], dim=-1) if l_latent is not None else x_l

                # 投影到嵌入维度
                x_a = F.dropout(F.relu(self.proj_a(x_a_combined)), p=self.trans_dropout, training=self.training)
                x_l = F.dropout(F.relu(self.proj_l(x_l_combined)), p=self.trans_dropout, training=self.training)
                x_a = x_a.transpose(0, 1)
                x_l = x_l.transpose(0, 1)

            elif self.missing == 'V':
                x_a, x_v = src, tgt
                # 音频模态不变特征提取
                a_latent = None
                if self.dnae_a:
                    a_latent, a_loss = self.dnae_a(x_a)
                    dnae_loss += a_loss
                # 视频模态不变特征提取
                v_latent = None
                if self.dnae_v:
                    v_latent, v_loss = self.dnae_v(x_v)
                    dnae_loss += v_loss

                # 融合原始特征与不变特征
                x_a_combined = torch.cat([x_a, a_latent], dim=-1) if a_latent is not None else x_a
                x_v_combined = torch.cat([x_v, v_latent], dim=-1) if v_latent is not None else x_v

                # 投影到嵌入维度
                x_a = F.dropout(F.relu(self.proj_a(x_a_combined)), p=self.trans_dropout, training=self.training)
                x_v = F.dropout(F.relu(self.proj_v(x_v_combined)), p=self.trans_dropout, training=self.training)
                x_a = x_a.transpose(0, 1)
                x_v = x_v.transpose(0, 1)

            else:
                raise ValueError('Unknown missing modality type')

        elif self.modalities == 'V':
            if self.missing == 'L':
                x_v, x_l = src, tgt
                # 视频模态不变特征提取
                v_latent = None
                if self.dnae_v:
                    v_latent, v_loss = self.dnae_v(x_v)
                    dnae_loss += v_loss
                # 文本模态不变特征提取
                l_latent = None
                if self.dnae_l:
                    l_latent, l_loss = self.dnae_l(x_l)
                    dnae_loss += l_loss

                # 融合原始特征与不变特征
                x_v_combined = torch.cat([x_v, v_latent], dim=-1) if v_latent is not None else x_v
                x_l_combined = torch.cat([x_l, l_latent], dim=-1) if l_latent is not None else x_l

                # 投影到嵌入维度
                x_v = F.dropout(F.relu(self.proj_v(x_v_combined)), p=self.trans_dropout, training=self.training)
                x_l = F.dropout(F.relu(self.proj_l(x_l_combined)), p=self.trans_dropout, training=self.training)
                x_v = x_v.transpose(0, 1)
                x_l = x_l.transpose(0, 1)

            elif self.missing == 'A':
                x_v, x_a = src, tgt
                # 视频模态不变特征提取
                v_latent = None
                if self.dnae_v:
                    v_latent, v_loss = self.dnae_v(x_v)
                    dnae_loss += v_loss
                # 音频模态不变特征提取
                a_latent = None
                if self.dnae_a:
                    a_latent, a_loss = self.dnae_a(x_a)
                    dnae_loss += a_loss

                # 融合原始特征与不变特征
                x_v_combined = torch.cat([x_v, v_latent], dim=-1) if v_latent is not None else x_v
                x_a_combined = torch.cat([x_a, a_latent], dim=-1) if a_latent is not None else x_a

                # 投影到嵌入维度
                x_v = F.dropout(F.relu(self.proj_v(x_v_combined)), p=self.trans_dropout, training=self.training)
                x_a = F.dropout(F.relu(self.proj_a(x_a_combined)), p=self.trans_dropout, training=self.training)
                x_v = x_v.transpose(0, 1)
                x_a = x_a.transpose(0, 1)

            else:
                raise ValueError('Unknown missing modality type')

        elif self.modalities == 'LA':
            (x_l, x_a), x_v = src, tgt
            # 文本模态不变特征提取
            l_latent = None
            if self.dnae_l:
                l_latent, l_loss = self.dnae_l(x_l)
                dnae_loss += l_loss
            # 音频模态不变特征提取
            a_latent = None
            if self.dnae_a:
                a_latent, a_loss = self.dnae_a(x_a)
                dnae_loss += a_loss
            # 视频模态不变特征提取
            v_latent = None
            if self.dnae_v:
                v_latent, v_loss = self.dnae_v(x_v)
                dnae_loss += v_loss

            # 融合原始特征与不变特征
            x_l_combined = torch.cat([x_l, l_latent], dim=-1) if l_latent is not None else x_l
            x_a_combined = torch.cat([x_a, a_latent], dim=-1) if a_latent is not None else x_a
            x_v_combined = torch.cat([x_v, v_latent], dim=-1) if v_latent is not None else x_v

            # 投影到嵌入维度
            x_l = F.dropout(F.relu(self.proj_l(x_l_combined)), p=self.trans_dropout, training=self.training)
            x_a = F.dropout(F.relu(self.proj_a(x_a_combined)), p=self.trans_dropout, training=self.training)
            x_v = F.dropout(F.relu(self.proj_v(x_v_combined)), p=self.trans_dropout, training=self.training)
            x_l = x_l.transpose(0, 1)
            x_a = x_a.transpose(0, 1)
            x_v = x_v.transpose(0, 1)

        elif self.modalities == 'LV':
            (x_l, x_v), x_a = src, tgt
            # 文本模态不变特征提取
            l_latent = None
            if self.dnae_l:
                l_latent, l_loss = self.dnae_l(x_l)
                dnae_loss += l_loss
            # 视频模态不变特征提取
            v_latent = None
            if self.dnae_v:
                v_latent, v_loss = self.dnae_v(x_v)
                dnae_loss += v_loss
            # 音频模态不变特征提取
            a_latent = None
            if self.dnae_a:
                a_latent, a_loss = self.dnae_a(x_a)
                dnae_loss += a_loss

            # 融合原始特征与不变特征
            x_l_combined = torch.cat([x_l, l_latent], dim=-1) if l_latent is not None else x_l
            x_v_combined = torch.cat([x_v, v_latent], dim=-1) if v_latent is not None else x_v
            x_a_combined = torch.cat([x_a, a_latent], dim=-1) if a_latent is not None else x_a

            # 投影到嵌入维度
            x_l = F.dropout(F.relu(self.proj_l(x_l_combined)), p=self.trans_dropout, training=self.training)
            x_v = F.dropout(F.relu(self.proj_v(x_v_combined)), p=self.trans_dropout, training=self.training)
            x_a = F.dropout(F.relu(self.proj_a(x_a_combined)), p=self.trans_dropout, training=self.training)
            x_l = x_l.transpose(0, 1)
            x_v = x_v.transpose(0, 1)
            x_a = x_a.transpose(0, 1)

        elif self.modalities == 'AV':
            (x_a, x_v), x_l = src, tgt
            # 音频模态不变特征提取
            a_latent = None
            if self.dnae_a:
                a_latent, a_loss = self.dnae_a(x_a)
                dnae_loss += a_loss
            # 视频模态不变特征提取
            v_latent = None
            if self.dnae_v:
                v_latent, v_loss = self.dnae_v(x_v)
                dnae_loss += v_loss
            # 文本模态不变特征提取
            l_latent = None
            if self.dnae_l:
                l_latent, l_loss = self.dnae_l(x_l)
                dnae_loss += l_loss

            # 融合原始特征与不变特征
            x_a_combined = torch.cat([x_a, a_latent], dim=-1) if a_latent is not None else x_a
            x_v_combined = torch.cat([x_v, v_latent], dim=-1) if v_latent is not None else x_v
            x_l_combined = torch.cat([x_l, l_latent], dim=-1) if l_latent is not None else x_l

            # 投影到嵌入维度
            x_a = F.dropout(F.relu(self.proj_a(x_a_combined)), p=self.trans_dropout, training=self.training)
            x_v = F.dropout(F.relu(self.proj_v(x_v_combined)), p=self.trans_dropout, training=self.training)
            x_l = F.dropout(F.relu(self.proj_l(x_l_combined)), p=self.trans_dropout, training=self.training)
            x_a = x_a.transpose(0, 1)
            x_v = x_v.transpose(0, 1)
            x_l = x_l.transpose(0, 1)

        else:
            raise ValueError('Unknown modalities type')

        L_MODAL_TYPE_IDX = 0
        A_MODAL_TYPE_IDX = 1
        V_MODAL_TYPE_IDX = 2

        batch_size = tgt.shape[0]
        multi = self.multi.unsqueeze(1).repeat(1, batch_size, 1)

        if phase != 'test':
            if self.missing == 'L':
                x_l = torch.cat((multi, x_l[:-1]), dim=0)
            elif self.missing == 'A':
                x_a = torch.cat((multi, x_a[:-1]), dim=0)
            elif self.missing == 'V':
                x_v = torch.cat((multi, x_v[:-1]), dim=0)
            else:
                raise ValueError('Unknown missing modality type')
        else:
            if eval_start:
                if self.missing == 'L':
                    x_l = multi
                elif self.missing == 'A':
                    x_a = multi
                elif self.missing == 'V':
                    x_v = multi
                else:
                    raise ValueError('Unknown missing modality type')
            else:
                if self.missing == 'L':
                    x_l = torch.cat((multi, x_l), dim=0)
                elif self.missing == 'A':
                    x_a = torch.cat((multi, x_a), dim=0)
                elif self.missing == 'V':
                    x_v = torch.cat((multi, x_v), dim=0)
                else:
                    raise ValueError('Unknown missing modality type')

        if 'L' in self.modalities or self.missing == 'L':
            x_l_pos_ids = torch.arange(x_l.shape[0], device=tgt.device).unsqueeze(1).expand(-1, batch_size)
            l_pos_embeds = self.position_embeddings(x_l_pos_ids)
            l_modal_type_embeds = self.modal_type_embeddings(torch.full_like(x_l_pos_ids, L_MODAL_TYPE_IDX))
            l_embeds = l_pos_embeds + l_modal_type_embeds
            x_l = x_l + l_embeds
            x_l = F.dropout(x_l, p=self.embed_dropout, training=self.training)
        if 'A' in self.modalities or self.missing == 'A':
            x_a_pos_ids = torch.arange(x_a.shape[0], device=tgt.device).unsqueeze(1).expand(-1, batch_size)
            a_pos_embeds = self.position_embeddings(x_a_pos_ids)
            a_modal_type_embeds = self.modal_type_embeddings(torch.full_like(x_a_pos_ids, A_MODAL_TYPE_IDX))
            a_embeds = a_pos_embeds + a_modal_type_embeds
            x_a = x_a + a_embeds
            x_a = F.dropout(x_a, p=self.embed_dropout, training=self.training)
        if 'V' in self.modalities or self.missing == 'V':
            x_v_pos_ids = torch.arange(x_v.shape[0], device=tgt.device).unsqueeze(1).expand(-1, batch_size)
            v_pos_embeds = self.position_embeddings(x_v_pos_ids)
            v_modal_type_embeds = self.modal_type_embeddings(torch.full_like(x_v_pos_ids, V_MODAL_TYPE_IDX))
            v_embeds = v_pos_embeds + v_modal_type_embeds
            x_v = x_v + v_embeds
            x_v = F.dropout(x_v, p=self.embed_dropout, training=self.training)

        if self.modalities == 'L':
            if self.missing == 'A':
                x = torch.cat((x_l, x_a), dim=0)
            elif self.missing == 'V':
                x = torch.cat((x_l, x_v), dim=0)
            else:
                raise ValueError('Unknown missing modality type')
        elif self.modalities == 'A':
            if self.missing == 'L':
                x = torch.cat((x_a, x_l), dim=0)
            elif self.missing == 'V':
                x = torch.cat((x_a, x_v), dim=0)
            else:
                raise ValueError('Unknown missing modality type')
        elif self.modalities == 'V':
            if self.missing == 'L':
                x = torch.cat((x_v, x_l), dim=0)
            elif self.missing == 'A':
                x = torch.cat((x_v, x_a), dim=0)
            else:
                raise ValueError('Unknown missing modality type')
        elif self.modalities == 'LA':
            x = torch.cat((x_l, x_a, x_v), dim=0)
        elif self.modalities == 'LV':
            x = torch.cat((x_l, x_v, x_a), dim=0)
        elif self.modalities == 'AV':
            x = torch.cat((x_a, x_v, x_l), dim=0)
        else:
            raise ValueError('Unknown modalities type')

        output = self.translator(x)

        if self.modalities == 'L':
            output = output[self.l_len:].transpose(0, 1)
        elif self.modalities == 'A':
            output = output[self.a_len:].transpose(0, 1)
        elif self.modalities == 'V':
            output = output[self.v_len:].transpose(0, 1)
        elif self.modalities == 'LA':
            output = output[self.l_len + self.a_len:].transpose(0, 1)
        elif self.modalities == 'LV':
            output = output[self.l_len + self.v_len:].transpose(0, 1)
        elif self.modalities == 'AV':
            output = output[self.a_len + self.v_len:].transpose(0, 1)
        else:
            raise ValueError('Unknown modalities type')

        output = self.out(output)
        return output, dnae_loss  # 返回生成结果和DnAE损失


class UNIMFModel(nn.Module):
    def __init__(self, hyp_params):
        super(UNIMFModel, self).__init__()
        if hyp_params.dataset == 'meld_senti' or hyp_params.dataset == 'meld_emo':
            self.orig_l_len, self.orig_a_len = hyp_params.l_len, hyp_params.a_len
            self.orig_d_l, self.orig_d_a = hyp_params.orig_d_l, hyp_params.orig_d_a
        else:
            self.orig_l_len, self.orig_a_len, self.orig_v_len = hyp_params.l_len, hyp_params.a_len, hyp_params.v_len
            self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
        self.l_kernel_size = hyp_params.l_kernel_size
        self.a_kernel_size = hyp_params.a_kernel_size
        if hyp_params.dataset != 'meld_senti' and hyp_params.dataset != 'meld_emo':
            self.v_kernel_size = hyp_params.v_kernel_size
        self.embed_dim = hyp_params.embed_dim
        self.num_heads = hyp_params.num_heads
        self.multimodal_layers = hyp_params.multimodal_layers
        self.attn_dropout = hyp_params.attn_dropout
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.modalities = hyp_params.modalities
        self.dataset = hyp_params.dataset
        self.language = hyp_params.language
        self.use_bert = hyp_params.use_bert

        self.distribute = hyp_params.distribute

        if self.dataset == 'meld_senti' or self.dataset == 'meld_emo':
            self.cls_len = 33
        else:
            self.cls_len = 1
        self.cls = nn.Parameter(torch.Tensor(self.cls_len, self.embed_dim))
        nn.init.xavier_uniform_(self.cls)

        self.l_len = self.orig_l_len - self.l_kernel_size + 1
        self.a_len = self.orig_a_len - self.a_kernel_size + 1
        if self.dataset != 'meld_senti' and self.dataset != 'meld_emo':
            self.v_len = self.orig_v_len - self.v_kernel_size + 1

        output_dim = hyp_params.output_dim

        if self.use_bert:
            self.text_model = BertTextEncoder(language=hyp_params.language, use_finetune=True)

        self.proj_l = nn.Conv1d(self.orig_d_l, self.embed_dim, kernel_size=self.l_kernel_size)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.embed_dim, kernel_size=self.a_kernel_size)
        if self.dataset != 'meld_senti' and self.dataset != 'meld_emo':
            self.proj_v = nn.Conv1d(self.orig_d_v, self.embed_dim, kernel_size=self.v_kernel_size)
        if 'meld' in self.dataset:
            self.proj_cls = nn.Conv1d(self.orig_d_l + self.orig_d_a, self.embed_dim, kernel_size=1)

        self.t = nn.GRU(input_size=self.embed_dim, hidden_size=self.embed_dim)
        self.a = nn.GRU(input_size=self.embed_dim, hidden_size=self.embed_dim)
        if self.dataset != 'meld_senti' and self.dataset != 'meld_emo':
            self.v = nn.GRU(input_size=self.embed_dim, hidden_size=self.embed_dim)

        if self.dataset == 'meld_senti' or self.dataset == 'meld_emo':
            self.position_embeddings = nn.Embedding(max(self.cls_len, self.l_len, self.a_len), self.embed_dim)
        else:
            self.position_embeddings = nn.Embedding(max(self.l_len, self.a_len, self.v_len), self.embed_dim)
        self.modal_type_embeddings = nn.Embedding(4, self.embed_dim)

        self.unimf = MultimodalTransformerEncoder(embed_dim=self.embed_dim,
                                                  num_heads=self.num_heads,
                                                  layers=self.multimodal_layers,
                                                  lens=(self.cls_len, self.l_len, self.a_len),
                                                  modalities=self.modalities,
                                                  attn_dropout=self.attn_dropout,
                                                  relu_dropout=self.relu_dropout,
                                                  res_dropout=self.res_dropout)

        combined_dim = self.embed_dim
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

        # 优化的门控机制
        self.glu_gate = nn.Linear(combined_dim, combined_dim)
        # 模态感知门控参数
        self.modal_gate = nn.Linear(combined_dim, combined_dim)
        # 动态缩放因子
        self.scale_factor = nn.Parameter(torch.tensor(1.0))
        # 模态缺失适应偏置
        self.missing_bias = nn.Parameter(torch.zeros(combined_dim))

    def forward(self, x_l, x_a, x_v=None):
        if self.distribute:
            self.t.flatten_parameters()
            self.a.flatten_parameters()
            if x_v is not None:
                self.v.flatten_parameters()

        L_MODAL_TYPE_IDX = 0
        A_MODAL_TYPE_IDX = 1
        V_MODAL_TYPE_IDX = 2
        MULTI_MODAL_TYPE_IDX = 3

        batch_size = x_l.shape[0]
        if self.dataset != 'meld_senti' and self.dataset != 'meld_emo':
            cls = self.cls.unsqueeze(1).repeat(1, batch_size, 1)
        else:
            cls = self.proj_cls(torch.cat((x_l, x_a), dim=-1).transpose(1, 2)).permute(2, 0, 1)

        cls_pos_ids = torch.arange(self.cls_len, device=x_l.device).unsqueeze(1).expand(-1, batch_size)
        h_l_pos_ids = torch.arange(self.l_len, device=x_l.device).unsqueeze(1).expand(-1, batch_size)
        h_a_pos_ids = torch.arange(self.a_len, device=x_a.device).unsqueeze(1).expand(-1, batch_size)
        if x_v is not None:
            h_v_pos_ids = torch.arange(self.v_len, device=x_v.device).unsqueeze(1).expand(-1, batch_size)

        cls_pos_embeds = self.position_embeddings(cls_pos_ids)
        h_l_pos_embeds = self.position_embeddings(h_l_pos_ids)
        h_a_pos_embeds = self.position_embeddings(h_a_pos_ids)
        if x_v is not None:
            h_v_pos_embeds = self.position_embeddings(h_v_pos_ids)

        cls_modal_type_embeds = self.modal_type_embeddings(torch.full_like(cls_pos_ids, MULTI_MODAL_TYPE_IDX))
        l_modal_type_embeds = self.modal_type_embeddings(torch.full_like(h_l_pos_ids, L_MODAL_TYPE_IDX))
        a_modal_type_embeds = self.modal_type_embeddings(torch.full_like(h_a_pos_ids, A_MODAL_TYPE_IDX))
        if x_v is not None:
            v_modal_type_embeds = self.modal_type_embeddings(torch.full_like(h_v_pos_ids, V_MODAL_TYPE_IDX))

        if self.use_bert:
            x_l = self.text_model(x_l)

        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a = x_a.transpose(1, 2)
        if x_v is not None:
            x_v = x_v.transpose(1, 2)

        proj_x_l = self.proj_l(x_l)
        proj_x_a = self.proj_a(x_a)
        if x_v is not None:
            proj_x_v = self.proj_v(x_v)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        if x_v is not None:
            proj_x_v = proj_x_v.permute(2, 0, 1)

        h_l, _ = self.t(proj_x_l)
        h_a, _ = self.a(proj_x_a)
        if x_v is not None:
            h_v, _ = self.v(proj_x_v)

        cls_embeds = cls_pos_embeds + cls_modal_type_embeds
        l_embeds = h_l_pos_embeds + l_modal_type_embeds
        a_embeds = h_a_pos_embeds + a_modal_type_embeds
        if x_v is not None:
            v_embeds = h_v_pos_embeds + v_modal_type_embeds
        cls = cls + cls_embeds
        h_l = h_l + l_embeds
        h_a = h_a + a_embeds
        if x_v is not None:
            h_v = h_v + v_embeds
        h_l = F.dropout(h_l, p=self.embed_dropout, training=self.training)
        h_a = F.dropout(h_a, p=self.embed_dropout, training=self.training)
        if x_v is not None:
            h_v = F.dropout(h_v, p=self.embed_dropout, training=self.training)

        if x_v is not None:
            x = torch.cat((cls, h_l, h_a, h_v), dim=0)
        else:
            x = torch.cat((cls, h_l, h_a), dim=0)
        x = self.unimf(x)

        if x_v is not None:
            last_hs = x[0]
        else:
            last_hs = x[:self.cls_len]

        # 残差块
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs

        # 优化的门控机制
        # 1. 基础门控
        base_gate = torch.sigmoid(self.glu_gate(last_hs_proj))

        # 2. 模态感知门控 - 基于输入模态动态调整
        modal_gate = torch.sigmoid(self.modal_gate(last_hs_proj))

        # 3. 融合门控
        combined_gate = base_gate * modal_gate

        # 4. 模态缺失适应 - 对缺失模态施加偏置
        if x_v is None and self.dataset != 'meld_senti' and self.dataset != 'meld_emo':
            gated_output = (last_hs_proj + self.missing_bias) * combined_gate
        else:
            gated_output = last_hs_proj * combined_gate

        # 5. 动态缩放
        gated_output = self.scale_factor * gated_output

        # 最终投影
        output = self.out_layer(gated_output)
        if x_v is None:
            output = output.transpose(0, 1)
        return output, last_hs


class BertTextEncoder(nn.Module):
    def __init__(self, language='en', use_finetune=False):
        super(BertTextEncoder, self).__init__()

        assert language in ['en', 'cn']

        tokenizer_class = BertTokenizer
        model_class = BertModel
        if language == 'en':
            self.tokenizer = tokenizer_class.from_pretrained('pretrained_bert/bert_en', do_lower_case=True)
            self.model = model_class.from_pretrained('pretrained_bert/bert_en')
        elif language == 'cn':
            self.tokenizer = tokenizer_class.from_pretrained('pretrained_bert/bert_cn')
            self.model = model_class.from_pretrained('pretrained_bert/bert_cn')

        self.use_finetune = use_finetune

    def get_tokenizer(self):
        return self.tokenizer

    def from_text(self, text):
        input_ids = self.get_id(text)
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]
        return last_hidden_states.squeeze()

    def forward(self, text):
        input_ids, input_mask, segment_ids = text[:, 0, :].long(), text[:, 1, :].float(), text[:, 2, :].long()
        if self.use_finetune:
            last_hidden_states = self.model(input_ids=input_ids,
                                            attention_mask=input_mask,
                                            token_type_ids=segment_ids)[0]
        else:
            with torch.no_grad():
                last_hidden_states = self.model(input_ids=input_ids,
                                                attention_mask=input_mask,
                                                token_type_ids=segment_ids)[0]
        return last_hidden_states