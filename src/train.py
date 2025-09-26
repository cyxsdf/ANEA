import torch
import torch.nn.functional as F
from torch import nn
import sys
import csv
from src import models
from src import ctc
from src.utils import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pickle
from tqdm import tqdm

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from src.eval_metrics import *


####################################################################
#
# Construct the model
#
####################################################################

def initiate(hyp_params, train_loader, valid_loader, test_loader):
    # 初始化判别器和生成器
    discriminators = None
    disc_optimizers = None

    if hyp_params.modalities != 'LAV':
        # 初始化生成器(译者模型)
        if hyp_params.modalities == 'L':
            translator1 = getattr(models, 'TRANSLATEModel')(hyp_params, 'A')
            translator2 = getattr(models, 'TRANSLATEModel')(hyp_params, 'V')
            # 初始化判别器
            disc_a = models.Discriminator(hyp_params.orig_d_a,
                                          hyp_params.a_len).cuda() if hyp_params.use_cuda else models.Discriminator(
                hyp_params.orig_d_a, hyp_params.a_len)
            disc_v = models.Discriminator(hyp_params.orig_d_v,
                                          hyp_params.v_len).cuda() if hyp_params.use_cuda else models.Discriminator(
                hyp_params.orig_d_v, hyp_params.v_len)
            disc_a_optim = optim.Adam(disc_a.parameters(), lr=hyp_params.disc_lr, betas=(0.5, 0.999))
            disc_v_optim = optim.Adam(disc_v.parameters(), lr=hyp_params.disc_lr, betas=(0.5, 0.999))
            discriminators = (disc_a, disc_v)
            disc_optimizers = (disc_a_optim, disc_v_optim)

        elif hyp_params.modalities == 'A':
            translator1 = getattr(models, 'TRANSLATEModel')(hyp_params, 'L')
            translator2 = getattr(models, 'TRANSLATEModel')(hyp_params, 'V')
            # 初始化判别器
            disc_l = models.Discriminator(hyp_params.orig_d_l,
                                          hyp_params.l_len).cuda() if hyp_params.use_cuda else models.Discriminator(
                hyp_params.orig_d_l, hyp_params.l_len)
            disc_v = models.Discriminator(hyp_params.orig_d_v,
                                          hyp_params.v_len).cuda() if hyp_params.use_cuda else models.Discriminator(
                hyp_params.orig_d_v, hyp_params.v_len)
            disc_l_optim = optim.Adam(disc_l.parameters(), lr=hyp_params.disc_lr, betas=(0.5, 0.999))
            disc_v_optim = optim.Adam(disc_v.parameters(), lr=hyp_params.disc_lr, betas=(0.5, 0.999))
            discriminators = (disc_l, disc_v)
            disc_optimizers = (disc_l_optim, disc_v_optim)

        elif hyp_params.modalities == 'V':
            translator1 = getattr(models, 'TRANSLATEModel')(hyp_params, 'L')
            translator2 = getattr(models, 'TRANSLATEModel')(hyp_params, 'A')
            # 初始化判别器
            disc_l = models.Discriminator(hyp_params.orig_d_l,
                                          hyp_params.l_len).cuda() if hyp_params.use_cuda else models.Discriminator(
                hyp_params.orig_d_l, hyp_params.l_len)
            disc_a = models.Discriminator(hyp_params.orig_d_a,
                                          hyp_params.a_len).cuda() if hyp_params.use_cuda else models.Discriminator(
                hyp_params.orig_d_a, hyp_params.a_len)
            disc_l_optim = optim.Adam(disc_l.parameters(), lr=hyp_params.disc_lr, betas=(0.5, 0.999))
            disc_a_optim = optim.Adam(disc_a.parameters(), lr=hyp_params.disc_lr, betas=(0.5, 0.999))
            discriminators = (disc_l, disc_a)
            disc_optimizers = (disc_l_optim, disc_a_optim)

        elif hyp_params.modalities == 'LA':
            translator = getattr(models, 'TRANSLATEModel')(hyp_params, 'V')
            # 初始化判别器
            disc_v = models.Discriminator(hyp_params.orig_d_v,
                                          hyp_params.v_len).cuda() if hyp_params.use_cuda else models.Discriminator(
                hyp_params.orig_d_v, hyp_params.v_len)
            disc_optim = optim.Adam(disc_v.parameters(), lr=hyp_params.disc_lr, betas=(0.5, 0.999))
            discriminators = disc_v
            disc_optimizers = disc_optim

        elif hyp_params.modalities == 'LV':
            translator = getattr(models, 'TRANSLATEModel')(hyp_params, 'A')
            # 初始化判别器
            disc_a = models.Discriminator(hyp_params.orig_d_a,
                                          hyp_params.a_len).cuda() if hyp_params.use_cuda else models.Discriminator(
                hyp_params.orig_d_a, hyp_params.a_len)
            disc_optim = optim.Adam(disc_a.parameters(), lr=hyp_params.disc_lr, betas=(0.5, 0.999))
            discriminators = disc_a
            disc_optimizers = disc_optim

        elif hyp_params.modalities == 'AV':
            translator = getattr(models, 'TRANSLATEModel')(hyp_params, 'L')
            # 初始化判别器
            disc_l = models.Discriminator(hyp_params.orig_d_l,
                                          hyp_params.l_len).cuda() if hyp_params.use_cuda else models.Discriminator(
                hyp_params.orig_d_l, hyp_params.l_len)
            disc_optim = optim.Adam(disc_l.parameters(), lr=hyp_params.disc_lr, betas=(0.5, 0.999))
            discriminators = disc_l
            disc_optimizers = disc_optim

        else:
            raise ValueError('Unknown modalities type')

        # 移动生成器到GPU
        if hyp_params.use_cuda:
            if hyp_params.modalities in ['L', 'A', 'V']:
                translator1 = translator1.cuda()
                translator2 = translator2.cuda()
            else:
                translator = translator.cuda()

        # 生成器优化器
        if hyp_params.modalities in ['L', 'A', 'V']:
            translator1_optimizer = getattr(optim, hyp_params.optim)(translator1.parameters(), lr=hyp_params.lr)
            translator2_optimizer = getattr(optim, hyp_params.optim)(translator2.parameters(), lr=hyp_params.lr)
        else:
            translator_optimizer = getattr(optim, hyp_params.optim)(translator.parameters(), lr=hyp_params.lr)

        trans_criterion = getattr(nn, 'MSELoss')()  # MSE损失
        adv_criterion = nn.BCELoss()  # 对抗损失

    # 初始化主模型
    model = getattr(models, hyp_params.model + 'Model')(hyp_params)
    if hyp_params.use_cuda:
        model = model.cuda()

    # 主模型优化器
    if hyp_params.use_bert:
        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(model.text_model.named_parameters())
        bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
        model_params_other = [p for n, p in list(model.named_parameters()) if 'text_model' not in n]
        optimizer_grouped_parameters = [
            {'params': bert_params_decay, 'weight_decay': hyp_params.weight_decay_bert, 'lr': hyp_params.lr_bert},
            {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': hyp_params.lr_bert},
            {'params': model_params_other, 'weight_decay': 0.0, 'lr': hyp_params.lr}
        ]
        optimizer = optim.Adam(optimizer_grouped_parameters)
    else:
        optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)

    criterion = getattr(nn, hyp_params.criterion)()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1)

    # 组装训练设置
    if hyp_params.modalities != 'LAV':
        if hyp_params.modalities in ['L', 'A', 'V']:
            settings = {
                'model': model,
                'translator1': translator1,
                'translator2': translator2,
                'translator1_optimizer': translator1_optimizer,
                'translator2_optimizer': translator2_optimizer,
                'discriminators': discriminators,
                'disc_optimizers': disc_optimizers,
                'trans_criterion': trans_criterion,
                'adv_criterion': adv_criterion,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler
            }
        else:
            settings = {
                'model': model,
                'translator': translator,
                'translator_optimizer': translator_optimizer,
                'discriminators': discriminators,
                'disc_optimizers': disc_optimizers,
                'trans_criterion': trans_criterion,
                'adv_criterion': adv_criterion,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler
            }
    else:
        settings = {
            'model': model,
            'optimizer': optimizer,
            'criterion': criterion,
            'scheduler': scheduler
        }

    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


####################################################################
#
# Training and evaluation scripts
#
####################################################################

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']

    # 提取生成器、判别器和损失函数
    translator = None
    discriminators = None
    disc_optimizers = None
    trans_criterion = None
    adv_criterion = None

    if hyp_params.modalities != 'LAV':
        trans_criterion = settings['trans_criterion']
        adv_criterion = settings['adv_criterion']
        discriminators = settings['discriminators']
        disc_optimizers = settings['disc_optimizers']

        if hyp_params.modalities in ['L', 'A', 'V']:
            translator1 = settings['translator1']
            translator2 = settings['translator2']
            translator1_optimizer = settings['translator1_optimizer']
            translator2_optimizer = settings['translator2_optimizer']
            translator = (translator1, translator2)
        else:
            translator = settings['translator']
            translator_optimizer = settings['translator_optimizer']

    def train(model, translator, optimizer, criterion):
        if isinstance(translator, tuple):
            translator1, translator2 = translator

        epoch_loss = 0
        model.train()

        if hyp_params.modalities != 'LAV':
            if hyp_params.modalities in ['L', 'A', 'V']:
                translator1.train()
                translator2.train()
            else:
                translator.train()
            # 确保判别器处于训练模式
            if isinstance(discriminators, tuple):
                for disc in discriminators:
                    disc.train()
            else:
                discriminators.train()

        num_batches = hyp_params.n_train // hyp_params.batch_size
        start_time = time.time()

        for i_batch, (batch_X, batch_Y, batch_META) in enumerate(train_loader):
            sample_ind, text, audio, vision = batch_X
            eval_attr = batch_Y.squeeze(-1)  # 标签处理

            # 梯度清零
            model.zero_grad()
            if hyp_params.modalities != 'LAV':
                if hyp_params.modalities in ['L', 'A', 'V']:
                    translator1.zero_grad()
                    translator2.zero_grad()
                else:
                    translator.zero_grad()
                # 判别器梯度清零
                if isinstance(disc_optimizers, tuple):
                    for opt in disc_optimizers:
                        opt.zero_grad()
                else:
                    disc_optimizers.zero_grad()

            # 数据移至GPU
            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
                    if hyp_params.dataset == 'iemocap':
                        eval_attr = eval_attr.long()

            batch_size = text.size(0)
            net = nn.DataParallel(model) if hyp_params.distribute else model
            trans_loss = 0  # 生成损失初始化
            contrast_loss = 0  # 对比损失初始化

            # 生成缺失模态并计算损失
            if hyp_params.modalities != 'LAV':
                if hyp_params.modalities in ['L', 'A', 'V']:
                    trans_net1 = nn.DataParallel(translator1) if hyp_params.distribute else translator1
                    trans_net2 = nn.DataParallel(translator2) if hyp_params.distribute else translator2

                    # 生成缺失模态
                    if hyp_params.modalities == 'L':
                        # 提取生成的音频和对比损失
                        trans_outputs1 = trans_net1(text, audio, 'train')
                        fake_a = trans_outputs1[0]
                        contrast_loss1 = trans_outputs1[1] if len(trans_outputs1) > 1 else 0

                        # 提取生成的视觉和对比损失
                        trans_outputs2 = trans_net2(text, vision, 'train')
                        fake_v = trans_outputs2[0]
                        contrast_loss2 = trans_outputs2[1] if len(trans_outputs2) > 1 else 0

                        # 计算总对比损失
                        contrast_loss = (contrast_loss1 + contrast_loss2) * 0.1  # 对比损失权重

                        # 计算MSE损失
                        mse_loss = trans_criterion(fake_a, audio) + trans_criterion(fake_v, vision)

                        # 训练判别器
                        disc1, disc2 = discriminators
                        real_labels = torch.ones(batch_size, 1).cuda() if hyp_params.use_cuda else torch.ones(
                            batch_size, 1)
                        fake_labels = torch.zeros(batch_size, 1).cuda() if hyp_params.use_cuda else torch.zeros(
                            batch_size, 1)

                        # 真实样本损失
                        real_a_pred = disc1(audio)
                        real_v_pred = disc2(vision)
                        real_loss_a = adv_criterion(real_a_pred, real_labels)
                        real_loss_v = adv_criterion(real_v_pred, real_labels)
                        real_loss = (real_loss_a + real_loss_v) * 0.5

                        # 生成样本损失
                        fake_a_pred = disc1(fake_a.detach())  # 分离生成器，不更新生成器
                        fake_v_pred = disc2(fake_v.detach())
                        fake_loss_a = adv_criterion(fake_a_pred, fake_labels)
                        fake_loss_v = adv_criterion(fake_v_pred, fake_labels)
                        fake_loss = (fake_loss_a + fake_loss_v) * 0.5

                        # 判别器总损失
                        disc_loss = (real_loss + fake_loss) * 0.5
                        disc_loss.backward()
                        disc_optimizers[0].step()
                        disc_optimizers[1].step()

                        # 训练生成器的对抗损失
                        gen_a_pred = disc1(fake_a)
                        gen_v_pred = disc2(fake_v)
                        adv_loss = (adv_criterion(gen_a_pred, real_labels) +
                                    adv_criterion(gen_v_pred, real_labels)) * 0.5

                        # 组合损失：MSE + 对抗损失 + 对比损失
                        trans_loss = mse_loss * hyp_params.mse_weight + adv_loss * hyp_params.adv_weight + contrast_loss

                    elif hyp_params.modalities == 'A':
                        # 提取生成的文本和对比损失
                        trans_outputs1 = trans_net1(audio, text, 'train')
                        fake_l = trans_outputs1[0]
                        contrast_loss1 = trans_outputs1[1] if len(trans_outputs1) > 1 else 0

                        # 提取生成的视觉和对比损失
                        trans_outputs2 = trans_net2(audio, vision, 'train')
                        fake_v = trans_outputs2[0]
                        contrast_loss2 = trans_outputs2[1] if len(trans_outputs2) > 1 else 0

                        # 计算总对比损失
                        contrast_loss = (contrast_loss1 + contrast_loss2) * 0.1  # 对比损失权重

                        # MSE损失
                        mse_loss = trans_criterion(fake_l, text) + trans_criterion(fake_v, vision)

                        # 训练判别器
                        disc1, disc2 = discriminators
                        real_labels = torch.ones(batch_size, 1).cuda() if hyp_params.use_cuda else torch.ones(
                            batch_size, 1)
                        fake_labels = torch.zeros(batch_size, 1).cuda() if hyp_params.use_cuda else torch.zeros(
                            batch_size, 1)

                        # 真实样本损失
                        real_l_pred = disc1(text)
                        real_v_pred = disc2(vision)
                        real_loss_l = adv_criterion(real_l_pred, real_labels)
                        real_loss_v = adv_criterion(real_v_pred, real_labels)
                        real_loss = (real_loss_l + real_loss_v) * 0.5

                        # 生成样本损失
                        fake_l_pred = disc1(fake_l.detach())
                        fake_v_pred = disc2(fake_v.detach())
                        fake_loss_l = adv_criterion(fake_l_pred, fake_labels)
                        fake_loss_v = adv_criterion(fake_v_pred, fake_labels)
                        fake_loss = (fake_loss_l + fake_loss_v) * 0.5

                        # 判别器总损失
                        disc_loss = (real_loss + fake_loss) * 0.5
                        disc_loss.backward()
                        disc_optimizers[0].step()
                        disc_optimizers[1].step()

                        # 生成器对抗损失
                        gen_l_pred = disc1(fake_l)
                        gen_v_pred = disc2(fake_v)
                        adv_loss = (adv_criterion(gen_l_pred, real_labels) +
                                    adv_criterion(gen_v_pred, real_labels)) * 0.5

                        # 组合损失
                        trans_loss = mse_loss * hyp_params.mse_weight + adv_loss * hyp_params.adv_weight + contrast_loss

                    elif hyp_params.modalities == 'V':
                        # 提取生成的文本和对比损失
                        trans_outputs1 = trans_net1(vision, text, 'train')
                        fake_l = trans_outputs1[0]
                        contrast_loss1 = trans_outputs1[1] if len(trans_outputs1) > 1 else 0

                        # 提取生成的音频和对比损失
                        trans_outputs2 = trans_net2(vision, audio, 'train')
                        fake_a = trans_outputs2[0]
                        contrast_loss2 = trans_outputs2[1] if len(trans_outputs2) > 1 else 0

                        # 计算总对比损失
                        contrast_loss = (contrast_loss1 + contrast_loss2) * 0.1  # 对比损失权重

                        # MSE损失
                        mse_loss = trans_criterion(fake_l, text) + trans_criterion(fake_a, audio)

                        # 训练判别器
                        disc1, disc2 = discriminators
                        real_labels = torch.ones(batch_size, 1).cuda() if hyp_params.use_cuda else torch.ones(
                            batch_size, 1)
                        fake_labels = torch.zeros(batch_size, 1).cuda() if hyp_params.use_cuda else torch.zeros(
                            batch_size, 1)

                        # 真实样本损失
                        real_l_pred = disc1(text)
                        real_a_pred = disc2(audio)
                        real_loss_l = adv_criterion(real_l_pred, real_labels)
                        real_loss_a = adv_criterion(real_a_pred, real_labels)
                        real_loss = (real_loss_l + real_loss_a) * 0.5

                        # 生成样本损失
                        fake_l_pred = disc1(fake_l.detach())
                        fake_a_pred = disc2(fake_a.detach())
                        fake_loss_l = adv_criterion(fake_l_pred, fake_labels)
                        fake_loss_a = adv_criterion(fake_a_pred, fake_labels)
                        fake_loss = (fake_loss_l + fake_loss_a) * 0.5

                        # 判别器总损失
                        disc_loss = (real_loss + fake_loss) * 0.5
                        disc_loss.backward()
                        disc_optimizers[0].step()
                        disc_optimizers[1].step()

                        # 生成器对抗损失
                        gen_l_pred = disc1(fake_l)
                        gen_a_pred = disc2(fake_a)
                        adv_loss = (adv_criterion(gen_l_pred, real_labels) +
                                    adv_criterion(gen_a_pred, real_labels)) * 0.5

                        # 组合损失
                        trans_loss = mse_loss * hyp_params.mse_weight + adv_loss * hyp_params.adv_weight + contrast_loss

                else:  # 双模态输入情况
                    trans_net = nn.DataParallel(translator) if hyp_params.distribute else translator

                    if hyp_params.modalities == 'LA':
                        # 提取生成的视觉和对比损失
                        trans_outputs = trans_net((text, audio), vision, 'train')
                        fake_v = trans_outputs[0]
                        contrast_loss = trans_outputs[1] * 0.1 if len(trans_outputs) > 1 else 0

                        # MSE损失
                        mse_loss = trans_criterion(fake_v, vision)

                        # 训练判别器
                        disc = discriminators
                        real_labels = torch.ones(batch_size, 1).cuda() if hyp_params.use_cuda else torch.ones(
                            batch_size, 1)
                        fake_labels = torch.zeros(batch_size, 1).cuda() if hyp_params.use_cuda else torch.zeros(
                            batch_size, 1)

                        # 真实样本损失
                        real_pred = disc(vision)
                        real_loss = adv_criterion(real_pred, real_labels)

                        # 生成样本损失
                        fake_pred = disc(fake_v.detach())
                        fake_loss = adv_criterion(fake_pred, fake_labels)

                        # 判别器总损失
                        disc_loss = (real_loss + fake_loss) * 0.5
                        disc_loss.backward()
                        disc_optimizers.step()

                        # 生成器对抗损失
                        gen_pred = disc(fake_v)
                        adv_loss = adv_criterion(gen_pred, real_labels)

                        # 组合损失
                        trans_loss = mse_loss * hyp_params.mse_weight + adv_loss * hyp_params.adv_weight + contrast_loss

                    elif hyp_params.modalities == 'LV':
                        # 提取生成的音频和对比损失
                        trans_outputs = trans_net((text, vision), audio, 'train')
                        fake_a = trans_outputs[0]
                        contrast_loss = trans_outputs[1] * 0.1 if len(trans_outputs) > 1 else 0

                        # MSE损失
                        mse_loss = trans_criterion(fake_a, audio)

                        # 训练判别器
                        disc = discriminators
                        real_labels = torch.ones(batch_size, 1).cuda() if hyp_params.use_cuda else torch.ones(
                            batch_size, 1)
                        fake_labels = torch.zeros(batch_size, 1).cuda() if hyp_params.use_cuda else torch.zeros(
                            batch_size, 1)

                        # 真实样本损失
                        real_pred = disc(audio)
                        real_loss = adv_criterion(real_pred, real_labels)

                        # 生成样本损失
                        fake_pred = disc(fake_a.detach())
                        fake_loss = adv_criterion(fake_pred, fake_labels)

                        # 判别器总损失
                        disc_loss = (real_loss + fake_loss) * 0.5
                        disc_loss.backward()
                        disc_optimizers.step()

                        # 生成器对抗损失
                        gen_pred = disc(fake_a)
                        adv_loss = adv_criterion(gen_pred, real_labels)

                        # 组合损失
                        trans_loss = mse_loss * hyp_params.mse_weight + adv_loss * hyp_params.adv_weight + contrast_loss

                    elif hyp_params.modalities == 'AV':
                        # 提取生成的文本和对比损失
                        trans_outputs = trans_net((audio, vision), text, 'train')
                        fake_l = trans_outputs[0]
                        contrast_loss = trans_outputs[1] * 0.1 if len(trans_outputs) > 1 else 0

                        # MSE损失
                        mse_loss = trans_criterion(fake_l, text)

                        # 训练判别器
                        disc = discriminators
                        real_labels = torch.ones(batch_size, 1).cuda() if hyp_params.use_cuda else torch.ones(
                            batch_size, 1)
                        fake_labels = torch.zeros(batch_size, 1).cuda() if hyp_params.use_cuda else torch.zeros(
                            batch_size, 1)

                        # 真实样本损失
                        real_pred = disc(text)
                        real_loss = adv_criterion(real_pred, real_labels)

                        # 生成样本损失
                        fake_pred = disc(fake_l.detach())
                        fake_loss = adv_criterion(fake_pred, fake_labels)

                        # 判别器总损失
                        disc_loss = (real_loss + fake_loss) * 0.5
                        disc_loss.backward()
                        disc_optimizers.step()

                        # 生成器对抗损失
                        gen_pred = disc(fake_l)
                        adv_loss = adv_criterion(gen_pred, real_labels)

                        # 组合损失
                        trans_loss = mse_loss * hyp_params.mse_weight + adv_loss * hyp_params.adv_weight + contrast_loss

            # 模型前向传播
            if hyp_params.modalities != 'LAV':
                if hyp_params.modalities == 'L':
                    preds, _ = net(text, fake_a, fake_v)
                elif hyp_params.modalities == 'A':
                    preds, _ = net(fake_l, audio, fake_v)
                elif hyp_params.modalities == 'V':
                    preds, _ = net(fake_l, fake_a, vision)
                elif hyp_params.modalities == 'LA':
                    preds, _ = net(text, audio, fake_v)
                elif hyp_params.modalities == 'LV':
                    preds, _ = net(text, fake_a, vision)
                elif hyp_params.modalities == 'AV':
                    preds, _ = net(fake_l, audio, vision)
                else:
                    raise ValueError('Unknown modalities type')
            else:
                preds, _ = net(text, audio, vision)

            # 处理特定数据集的输出格式
            if hyp_params.dataset == 'iemocap':
                preds = preds.view(-1, 2)
                eval_attr = eval_attr.view(-1)

            # 计算主模型损失
            raw_loss = criterion(preds, eval_attr)

            # 组合损失
            if hyp_params.modalities != 'LAV':
                combined_loss = raw_loss + trans_loss
            else:
                combined_loss = raw_loss

            # 反向传播和参数更新
            combined_loss.backward()

            # 梯度裁剪
            if hyp_params.modalities != 'LAV':
                if hyp_params.modalities in ['L', 'A', 'V']:
                    torch.nn.utils.clip_grad_norm_(translator1.parameters(), hyp_params.clip)
                    torch.nn.utils.clip_grad_norm_(translator2.parameters(), hyp_params.clip)
                    translator1_optimizer.step()
                    translator2_optimizer.step()
                else:
                    torch.nn.utils.clip_grad_norm_(translator.parameters(), hyp_params.clip)
                    translator_optimizer.step()

            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()

            # 累计损失
            epoch_loss += combined_loss.item() * batch_size

        return epoch_loss / hyp_params.n_train

    def evaluate(model, translator, criterion, test=False):
        if isinstance(translator, tuple):
            translator1, translator2 = translator

        model.eval()
        if hyp_params.modalities != 'LAV':
            if hyp_params.modalities in ['L', 'A', 'V']:
                translator1.eval()
                translator2.eval()
            else:
                translator.eval()
            # 判别器在评估时不需要训练
            if isinstance(discriminators, tuple):
                for disc in discriminators:
                    disc.eval()
            else:
                discriminators.eval()

        loader = test_loader if test else valid_loader
        total_loss = 0.0
        results = []
        truths = []

        with torch.no_grad():
            for i_batch, (batch_X, batch_Y, batch_META) in enumerate(loader):
                sample_ind, text, audio, vision = batch_X
                eval_attr = batch_Y.squeeze(dim=-1)

                # 数据移至GPU
                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
                        if hyp_params.dataset == 'iemocap':
                            eval_attr = eval_attr.long()

                batch_size = text.size(0)
                net = nn.DataParallel(model) if hyp_params.distribute else model
                fake_l, fake_a, fake_v = None, None, None
                trans_loss = 0

                # 生成缺失模态
                if hyp_params.modalities != 'LAV':
                    if not test:  # 验证模式
                        if hyp_params.modalities in ['L', 'A', 'V']:
                            trans_net1 = nn.DataParallel(translator1) if hyp_params.distribute else translator1
                            trans_net2 = nn.DataParallel(translator2) if hyp_params.distribute else translator2

                            if hyp_params.modalities == 'L':
                                # 提取生成的音频和视觉
                                trans_outputs1 = trans_net1(text, audio, 'valid')
                                fake_a = trans_outputs1[0]

                                trans_outputs2 = trans_net2(text, vision, 'valid')
                                fake_v = trans_outputs2[0]

                                trans_loss = trans_criterion(fake_a, audio) + trans_criterion(fake_v, vision)
                            elif hyp_params.modalities == 'A':
                                # 提取生成的文本和视觉
                                trans_outputs1 = trans_net1(audio, text, 'valid')
                                fake_l = trans_outputs1[0]

                                trans_outputs2 = trans_net2(audio, vision, 'valid')
                                fake_v = trans_outputs2[0]

                                trans_loss = trans_criterion(fake_l, text) + trans_criterion(fake_v, vision)
                            elif hyp_params.modalities == 'V':
                                # 提取生成的文本和音频
                                trans_outputs1 = trans_net1(vision, text, 'valid')
                                fake_l = trans_outputs1[0]

                                trans_outputs2 = trans_net2(vision, audio, 'valid')
                                fake_a = trans_outputs2[0]

                                trans_loss = trans_criterion(fake_l, text) + trans_criterion(fake_a, audio)
                        else:
                            trans_net = nn.DataParallel(translator) if hyp_params.distribute else translator

                            if hyp_params.modalities == 'LA':
                                # 提取生成的视觉
                                trans_outputs = trans_net((text, audio), vision, 'valid')
                                fake_v = trans_outputs[0]
                                trans_loss = trans_criterion(fake_v, vision)
                            elif hyp_params.modalities == 'LV':
                                # 提取生成的音频
                                trans_outputs = trans_net((text, vision), audio, 'valid')
                                fake_a = trans_outputs[0]
                                trans_loss = trans_criterion(fake_a, audio)
                            elif hyp_params.modalities == 'AV':
                                # 提取生成的文本
                                trans_outputs = trans_net((audio, vision), text, 'valid')
                                fake_l = trans_outputs[0]
                                trans_loss = trans_criterion(fake_l, text)
                    else:  # 测试模式，自回归生成
                        if hyp_params.modalities in ['L', 'A', 'V']:
                            trans_net1 = nn.DataParallel(translator1) if hyp_params.distribute else translator1
                            trans_net2 = nn.DataParallel(translator2) if hyp_params.distribute else translator2

                            if hyp_params.modalities == 'L':
                                # 生成音频模态
                                fake_a = torch.Tensor().cuda() if hyp_params.use_cuda else torch.Tensor()
                                for i in range(hyp_params.a_len):
                                    if i == 0:
                                        output = trans_net1(text, audio, 'test', eval_start=True)
                                        fake_a_token = output[0][:, [-1]]  # 只取生成数据部分
                                    else:
                                        output = trans_net1(text, fake_a, 'test')
                                        fake_a_token = output[0][:, [-1]]  # 只取生成数据部分
                                    fake_a = torch.cat((fake_a, fake_a_token), dim=1)

                                # 生成视觉模态
                                fake_v = torch.Tensor().cuda() if hyp_params.use_cuda else torch.Tensor()
                                for i in range(hyp_params.v_len):
                                    if i == 0:
                                        output = trans_net2(text, vision, 'test', eval_start=True)
                                        fake_v_token = output[0][:, [-1]]  # 只取生成数据部分
                                    else:
                                        output = trans_net2(text, fake_v, 'test')
                                        fake_v_token = output[0][:, [-1]]  # 只取生成数据部分
                                    fake_v = torch.cat((fake_v, fake_v_token), dim=1)

                            elif hyp_params.modalities == 'A':
                                # 生成文本模态
                                fake_l = torch.Tensor().cuda() if hyp_params.use_cuda else torch.Tensor()
                                for i in range(hyp_params.l_len):
                                    if i == 0:
                                        output = trans_net1(audio, text, 'test', eval_start=True)
                                        fake_l_token = output[0][:, [-1]]  # 只取生成数据部分
                                    else:
                                        output = trans_net1(audio, fake_l, 'test')
                                        fake_l_token = output[0][:, [-1]]  # 只取生成数据部分
                                    fake_l = torch.cat((fake_l, fake_l_token), dim=1)

                                # 生成视觉模态
                                fake_v = torch.Tensor().cuda() if hyp_params.use_cuda else torch.Tensor()
                                for i in range(hyp_params.v_len):
                                    if i == 0:
                                        output = trans_net2(audio, vision, 'test', eval_start=True)
                                        fake_v_token = output[0][:, [-1]]  # 只取生成数据部分
                                    else:
                                        output = trans_net2(audio, fake_v, 'test')
                                        fake_v_token = output[0][:, [-1]]  # 只取生成数据部分
                                    fake_v = torch.cat((fake_v, fake_v_token), dim=1)

                            elif hyp_params.modalities == 'V':
                                # 生成文本模态
                                fake_l = torch.Tensor().cuda() if hyp_params.use_cuda else torch.Tensor()
                                for i in range(hyp_params.l_len):
                                    if i == 0:
                                        output = trans_net1(vision, text, 'test', eval_start=True)
                                        fake_l_token = output[0][:, [-1]]  # 只取生成数据部分
                                    else:
                                        output = trans_net1(vision, fake_l, 'test')
                                        fake_l_token = output[0][:, [-1]]  # 只取生成数据部分
                                    fake_l = torch.cat((fake_l, fake_l_token), dim=1)

                                # 生成音频模态
                                fake_a = torch.Tensor().cuda() if hyp_params.use_cuda else torch.Tensor()
                                for i in range(hyp_params.a_len):
                                    if i == 0:
                                        output = trans_net2(vision, audio, 'test', eval_start=True)
                                        fake_a_token = output[0][:, [-1]]  # 只取生成数据部分
                                    else:
                                        output = trans_net2(vision, fake_a, 'test')
                                        fake_a_token = output[0][:, [-1]]  # 只取生成数据部分
                                    fake_a = torch.cat((fake_a, fake_a_token), dim=1)
                        else:
                            trans_net = nn.DataParallel(translator) if hyp_params.distribute else translator

                            if hyp_params.modalities == 'LA':
                                # 生成视觉模态
                                fake_v = torch.Tensor().cuda() if hyp_params.use_cuda else torch.Tensor()
                                for i in range(hyp_params.v_len):
                                    if i == 0:
                                        output = trans_net((text, audio), vision, 'test', eval_start=True)
                                        fake_v_token = output[0][:, [-1]]  # 只取生成数据部分
                                    else:
                                        output = trans_net((text, audio), fake_v, 'test')
                                        fake_v_token = output[0][:, [-1]]  # 只取生成数据部分
                                    fake_v = torch.cat((fake_v, fake_v_token), dim=1)

                            elif hyp_params.modalities == 'LV':
                                # 生成音频模态
                                fake_a = torch.Tensor().cuda() if hyp_params.use_cuda else torch.Tensor()
                                for i in range(hyp_params.a_len):
                                    if i == 0:
                                        output = trans_net((text, vision), audio, 'test', eval_start=True)
                                        fake_a_token = output[0][:, [-1]]  # 只取生成数据部分
                                    else:
                                        output = trans_net((text, vision), fake_a, 'test')
                                        fake_a_token = output[0][:, [-1]]  # 只取生成数据部分
                                    fake_a = torch.cat((fake_a, fake_a_token), dim=1)

                            elif hyp_params.modalities == 'AV':
                                # 生成文本模态
                                fake_l = torch.Tensor().cuda() if hyp_params.use_cuda else torch.Tensor()
                                for i in range(hyp_params.l_len):
                                    if i == 0:
                                        output = trans_net((audio, vision), text, 'test', eval_start=True)
                                        fake_l_token = output[0][:, [-1]]  # 只取生成数据部分
                                    else:
                                        output = trans_net((audio, vision), fake_l, 'test')
                                        fake_l_token = output[0][:, [-1]]  # 只取生成数据部分
                                    fake_l = torch.cat((fake_l, fake_l_token), dim=1)

                # 模型预测
                if hyp_params.modalities != 'LAV':
                    if hyp_params.modalities == 'L':
                        preds, _ = net(text, fake_a, fake_v)
                    elif hyp_params.modalities == 'A':
                        preds, _ = net(fake_l, audio, fake_v)
                    elif hyp_params.modalities == 'V':
                        preds, _ = net(fake_l, fake_a, vision)
                    elif hyp_params.modalities == 'LA':
                        preds, _ = net(text, audio, fake_v)
                    elif hyp_params.modalities == 'LV':
                        preds, _ = net(text, fake_a, vision)
                    elif hyp_params.modalities == 'AV':
                        preds, _ = net(fake_l, audio, vision)
                    else:
                        raise ValueError('Unknown modalities type')
                else:
                    preds, _ = net(text, audio, vision)

                # 处理特定数据集的输出格式
                if hyp_params.dataset == 'iemocap':
                    preds = preds.view(-1, 2)
                    eval_attr = eval_attr.view(-1)

                # 计算损失
                raw_loss = criterion(preds, eval_attr)
                if hyp_params.modalities != 'LAV' and not test:
                    combined_loss = raw_loss + trans_loss
                else:
                    combined_loss = raw_loss

                total_loss += combined_loss.item() * batch_size

                # 收集结果
                results.append(preds)
                truths.append(eval_attr)

        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)
        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths

    # 打印模型参数信息
    if hyp_params.modalities != 'LAV':
        if hyp_params.modalities in ['L', 'A', 'V']:
            mgm_parameter1 = sum([param.nelement() for param in translator1.parameters()])
            mgm_parameter2 = sum([param.nelement() for param in translator2.parameters()])
            mgm_parameter = mgm_parameter1 + mgm_parameter2
        else:
            mgm_parameter = sum([param.nelement() for param in translator.parameters()])
        print(f'Trainable Parameters for Multimodal Generation Model (MGM): {mgm_parameter}...')

        # 判别器参数
        if isinstance(discriminators, tuple):
            disc_parameter = sum([sum([p.nelement() for p in disc.parameters()]) for disc in discriminators])
        else:
            disc_parameter = sum([param.nelement() for param in discriminators.parameters()])
        print(f'Trainable Parameters for Discriminators: {disc_parameter}...')

    mum_parameter = sum([param.nelement() for param in model.parameters()])
    print(f'Trainable Parameters for Multimodal Understanding Model (MUM): {mum_parameter}...')

    # 训练循环
    best_valid = 1e8
    loop = tqdm(range(1, hyp_params.num_epochs + 1), leave=False)
    for epoch in loop:
        loop.set_description(f'Epoch {epoch:2d}/{hyp_params.num_epochs}')
        start = time.time()

        # 训练模型
        train(model, translator, optimizer, criterion)

        # 验证模型
        val_loss, _, _ = evaluate(model, translator, criterion, test=False)
        end = time.time()
        duration = end - start

        # 学习率调度
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_valid:
            if hyp_params.modalities in ['L', 'A', 'V']:
                save_model(hyp_params, translator1, name='TRANSLATOR_1')
                save_model(hyp_params, translator2, name='TRANSLATOR_2')
                # 保存判别器
                save_model(hyp_params, discriminators[0], name='DISCRIMINATOR_1')
                save_model(hyp_params, discriminators[1], name='DISCRIMINATOR_2')
            else:
                save_model(hyp_params, translator, name='TRANSLATOR')
                # 保存判别器
                save_model(hyp_params, discriminators, name='DISCRIMINATOR')

            save_model(hyp_params, model, name=hyp_params.name)
            best_valid = val_loss

    # 加载最佳模型进行测试
    if hyp_params.modalities in ['L', 'A', 'V']:
        translator1 = load_model(hyp_params, name='TRANSLATOR_1')
        translator2 = load_model(hyp_params, name='TRANSLATOR_2')
        translator = (translator1, translator2)
        # 加载判别器
        discriminators = (
            load_model(hyp_params, name='DISCRIMINATOR_1'),
            load_model(hyp_params, name='DISCRIMINATOR_2')
        )
    else:
        translator = load_model(hyp_params, name='TRANSLATOR')
        # 加载判别器
        discriminators = load_model(hyp_params, name='DISCRIMINATOR')

    model = load_model(hyp_params, name=hyp_params.name)
    _, results, truths = evaluate(model, translator, criterion, test=True)

    # 评估结果
    if hyp_params.dataset == "mosei_senti" or hyp_params.dataset == 'mosei-bert':
        acc = eval_mosei_senti(results, truths, True)
    elif hyp_params.dataset == 'mosi' or hyp_params.dataset == 'mosi-bert':
        acc = eval_mosi(results, truths, True)
    elif hyp_params.dataset == 'iemocap':
        acc = eval_iemocap(results, truths)
    elif hyp_params.dataset == 'sims':
        acc = eval_sims(results, truths)

    return acc