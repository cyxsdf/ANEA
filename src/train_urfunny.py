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
    # 初始化生成器、判别器及相关优化器和损失函数
    translator = None
    discriminator = None
    disc_optimizer = None
    trans_criterion = None
    adv_criterion = None
    translator_optimizer = None

    if hyp_params.modalities != 'AV':
        # 生成器初始化（支持L、A、V三种单模态）
        if hyp_params.modalities == 'A':
            translator = getattr(models, 'TRANSLATEModel')(hyp_params, 'V')  # A模态生成V
        elif hyp_params.modalities == 'V':
            translator = getattr(models, 'TRANSLATEModel')(hyp_params, 'A')  # V模态生成A
        elif hyp_params.modalities == 'L':
            # L模态需要生成两种缺失模态（A和V）
            translator1 = getattr(models, 'TRANSLATEModel')(hyp_params, 'A')  # L生成A
            translator2 = getattr(models, 'TRANSLATEModel')(hyp_params, 'V')  # L生成V
            translator = (translator1, translator2)
        else:
            raise ValueError(f'不支持的模态类型: {hyp_params.modalities}')

        # 移动生成器到GPU
        if hyp_params.use_cuda:
            if hyp_params.modalities == 'L':
                translator = (translator[0].cuda(), translator[1].cuda())
            else:
                translator = translator.cuda()

        # 生成器优化器
        if hyp_params.modalities == 'L':
            translator_optimizer = (
                getattr(optim, hyp_params.optim)(translator[0].parameters(), lr=hyp_params.lr),
                getattr(optim, hyp_params.optim)(translator[1].parameters(), lr=hyp_params.lr)
            )
        else:
            translator_optimizer = getattr(optim, hyp_params.optim)(translator.parameters(), lr=hyp_params.lr)

        # 判别器初始化（单模态1个判别器，L模态需要2个）
        if hyp_params.modalities == 'L':
            discriminator1 = getattr(models, 'Discriminator')(hyp_params, modal='A')  # 判别A模态
            discriminator2 = getattr(models, 'Discriminator')(hyp_params, modal='V')  # 判别V模态
            discriminator = (discriminator1, discriminator2)
            if hyp_params.use_cuda:
                discriminator = (discriminator[0].cuda(), discriminator[1].cuda())
            disc_optimizer = (
                getattr(optim, hyp_params.optim)(discriminator[0].parameters(), lr=hyp_params.lr),
                getattr(optim, hyp_params.optim)(discriminator[1].parameters(), lr=hyp_params.lr)
            )
        else:
            target_modal = 'V' if hyp_params.modalities == 'A' else 'A'
            discriminator = getattr(models, 'Discriminator')(hyp_params, modal=target_modal)
            if hyp_params.use_cuda:
                discriminator = discriminator.cuda()
            disc_optimizer = getattr(optim, hyp_params.optim)(discriminator.parameters(), lr=hyp_params.lr)

        # 损失函数
        trans_criterion = getattr(nn, 'MSELoss')()  # 生成损失（MSE）
        adv_criterion = getattr(nn, 'BCEWithLogitsLoss')()  # 对抗损失（BCE）

    # 主模型初始化
    model = getattr(models, hyp_params.model + 'Model')(hyp_params)
    if hyp_params.use_cuda:
        model = model.cuda()

    # 主模型优化器和损失函数
    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1)

    # 配置训练设置
    if hyp_params.modalities != 'AV':
        settings = {
            'model': model,
            'translator': translator,
            'translator_optimizer': translator_optimizer,
            'discriminator': discriminator,
            'disc_optimizer': disc_optimizer,
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

    # 提取GAN相关组件
    translator = None
    discriminator = None
    disc_optimizer = None
    trans_criterion = None
    adv_criterion = None
    translator_optimizer = None

    if hyp_params.modalities != 'AV':
        trans_criterion = settings['trans_criterion']
        adv_criterion = settings['adv_criterion']
        discriminator = settings['discriminator']
        disc_optimizer = settings['disc_optimizer']
        translator = settings['translator']
        translator_optimizer = settings['translator_optimizer']

    # 对比损失权重
    contrast_weight = 0.1 if hasattr(hyp_params, 'contrast_weight') else 0.1

    def train(model, translator, optimizer, criterion):
        epoch_loss = 0
        model.train()
        if hyp_params.modalities != 'AV':
            # 生成器和判别器设为训练模式
            if hyp_params.modalities == 'L':
                translator[0].train()
                translator[1].train()
                discriminator[0].train()
                discriminator[1].train()
            else:
                translator.train()
                discriminator.train()

        num_batches = hyp_params.n_train // hyp_params.batch_size
        start_time = time.time()

        for i_batch, (batch_data, masks, labels) in enumerate(train_loader):
            # 解析批次数据（根据模态类型适配输入）
            if hyp_params.modalities == 'L':
                text, audio, video = batch_data  # L模态输入文本，需生成A和V
            elif hyp_params.modalities == 'A':
                audio, video = batch_data  # A模态输入音频，需生成V
                text = None
            elif hyp_params.modalities == 'V':
                video, audio = batch_data  # V模态输入视频，需生成A
                text = None
            else:  # AV模态
                audio, video = batch_data
                text = None

            # 梯度清零
            model.zero_grad()
            if hyp_params.modalities != 'AV':
                if hyp_params.modalities == 'L':
                    translator[0].zero_grad()
                    translator[1].zero_grad()
                    disc_optimizer[0].zero_grad()
                    disc_optimizer[1].zero_grad()
                else:
                    translator.zero_grad()
                    disc_optimizer.zero_grad()

            # 数据移至GPU
            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    if text is not None:
                        text = text.cuda()
                    audio = audio.cuda() if audio is not None else None
                    video = video.cuda() if video is not None else None
                    masks = masks.cuda()
                    labels = labels.cuda()

            # 应用掩码（根据模态类型处理）
            batch_size = labels.size(0)
            if audio is not None:
                masks_audio = masks.unsqueeze(-1).expand(-1, hyp_params.a_len, hyp_params.orig_d_a)
                audio = audio * masks_audio
            if video is not None:
                masks_video = masks.unsqueeze(-1).expand(-1, hyp_params.v_len, hyp_params.orig_d_v)
                video = video * masks_video

            net = nn.DataParallel(model) if hyp_params.distribute else model
            trans_loss = torch.tensor(0.0, device=labels.device)
            contrast_loss = torch.tensor(0.0, device=labels.device)
            fake_v, fake_a = None, None

            # GAN训练流程
            if hyp_params.modalities != 'AV':
                if hyp_params.modalities == 'L':
                    # L模态：生成A和V两种模态
                    trans_net1 = nn.DataParallel(translator[0]) if hyp_params.distribute else translator[0]
                    trans_net2 = nn.DataParallel(translator[1]) if hyp_params.distribute else translator[1]

                    # 生成缺失模态
                    trans_outputs1 = trans_net1(text, audio, 'train')  # L生成A
                    trans_outputs2 = trans_net2(text, video, 'train')  # L生成V
                    fake_a, fake_v = trans_outputs1[0], trans_outputs2[0]
                    contrast_loss = (trans_outputs1[1] if len(trans_outputs1) > 1 else 0) + \
                                    (trans_outputs2[1] if len(trans_outputs2) > 1 else 0)
                    mse_loss = trans_criterion(fake_a, audio) + trans_criterion(fake_v, video)

                    # 训练判别器
                    real_labels = torch.ones(batch_size, 1, device=labels.device)
                    fake_labels = torch.zeros(batch_size, 1, device=labels.device)

                    # 判别器1（A模态）
                    real_pred_a = discriminator[0](audio)
                    fake_pred_a = discriminator[0](fake_a.detach())
                    real_loss_a = adv_criterion(real_pred_a, real_labels)
                    fake_loss_a = adv_criterion(fake_pred_a, fake_labels)
                    disc_loss_a = (real_loss_a + fake_loss_a) * 0.5

                    # 判别器2（V模态）
                    real_pred_v = discriminator[1](video)
                    fake_pred_v = discriminator[1](fake_v.detach())
                    real_loss_v = adv_criterion(real_pred_v, real_labels)
                    fake_loss_v = adv_criterion(fake_pred_v, fake_labels)
                    disc_loss_v = (real_loss_v + fake_loss_v) * 0.5

                    # 总判别器损失及更新
                    disc_loss = (disc_loss_a + disc_loss_v) * 0.5
                    disc_loss.backward(retain_graph=True)
                    disc_optimizer[0].step()
                    disc_optimizer[1].step()

                    # 生成器对抗损失
                    gen_pred_a = discriminator[0](fake_a)
                    gen_pred_v = discriminator[1](fake_v)
                    adv_loss = (adv_criterion(gen_pred_a, real_labels) +
                                adv_criterion(gen_pred_v, real_labels)) * 0.5

                    # 组合生成器损失
                    trans_loss = mse_loss * hyp_params.mse_weight + \
                                 adv_loss * hyp_params.adv_weight + \
                                 contrast_weight * contrast_loss

                elif hyp_params.modalities == 'A':
                    # A模态：生成V模态
                    trans_net = nn.DataParallel(translator) if hyp_params.distribute else translator
                    trans_outputs = trans_net(audio, video, 'train')
                    fake_v = trans_outputs[0]
                    contrast_loss = trans_outputs[1] if len(trans_outputs) > 1 else 0
                    mse_loss = trans_criterion(fake_v, video)

                    # 训练判别器
                    real_labels = torch.ones(batch_size, 1, device=labels.device)
                    fake_labels = torch.zeros(batch_size, 1, device=labels.device)

                    real_pred = discriminator(video)
                    fake_pred = discriminator(fake_v.detach())
                    real_loss = adv_criterion(real_pred, real_labels)
                    fake_loss = adv_criterion(fake_pred, fake_labels)
                    disc_loss = (real_loss + fake_loss) * 0.5
                    disc_loss.backward(retain_graph=True)
                    disc_optimizer.step()

                    # 生成器对抗损失
                    gen_pred = discriminator(fake_v)
                    adv_loss = adv_criterion(gen_pred, real_labels)
                    trans_loss = mse_loss * hyp_params.mse_weight + \
                                 adv_loss * hyp_params.adv_weight + \
                                 contrast_weight * contrast_loss

                elif hyp_params.modalities == 'V':
                    # V模态：生成A模态
                    trans_net = nn.DataParallel(translator) if hyp_params.distribute else translator
                    trans_outputs = trans_net(video, audio, 'train')
                    fake_a = trans_outputs[0]
                    contrast_loss = trans_outputs[1] if len(trans_outputs) > 1 else 0
                    mse_loss = trans_criterion(fake_a, audio)

                    # 训练判别器
                    real_labels = torch.ones(batch_size, 1, device=labels.device)
                    fake_labels = torch.zeros(batch_size, 1, device=labels.device)

                    real_pred = discriminator(audio)
                    fake_pred = discriminator(fake_a.detach())
                    real_loss = adv_criterion(real_pred, real_labels)
                    fake_loss = adv_criterion(fake_pred, fake_labels)
                    disc_loss = (real_loss + fake_loss) * 0.5
                    disc_loss.backward(retain_graph=True)
                    disc_optimizer.step()

                    # 生成器对抗损失
                    gen_pred = discriminator(fake_a)
                    adv_loss = adv_criterion(gen_pred, real_labels)
                    trans_loss = mse_loss * hyp_params.mse_weight + \
                                 adv_loss * hyp_params.adv_weight + \
                                 contrast_weight * contrast_loss

            # 主模型前向传播
            if hyp_params.modalities == 'L':
                outputs = net(text, fake_a, fake_v)  # L模态输入文本+生成的A和V
                preds = outputs[0]
            elif hyp_params.modalities == 'A':
                outputs = net(audio, fake_v)  # A模态输入音频+生成的V
                preds = outputs[0]
            elif hyp_params.modalities == 'V':
                outputs = net(fake_a, video)  # V模态输入视频+生成的A
                preds = outputs[0]
            else:  # AV模态
                outputs = net(audio, video)
                preds = outputs[0]

            # 计算主模型损失
            raw_loss = criterion(preds.transpose(1, 2), labels)

            # 组合总损失
            if hyp_params.modalities != 'AV':
                combined_loss = raw_loss + trans_loss
            else:
                combined_loss = raw_loss

            # 反向传播和参数更新
            combined_loss.backward()

            # 梯度裁剪与优化器更新
            if hyp_params.modalities != 'AV':
                if hyp_params.modalities == 'L':
                    torch.nn.utils.clip_grad_norm_(translator[0].parameters(), hyp_params.clip)
                    torch.nn.utils.clip_grad_norm_(translator[1].parameters(), hyp_params.clip)
                    translator_optimizer[0].step()
                    translator_optimizer[1].step()
                else:
                    torch.nn.utils.clip_grad_norm_(translator.parameters(), hyp_params.clip)
                    translator_optimizer.step()

            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()

            # 累计损失
            epoch_loss += combined_loss.item() * batch_size

        return epoch_loss / hyp_params.n_train

    def evaluate(model, translator, criterion, test=False):
        model.eval()
        if hyp_params.modalities != 'AV':
            if hyp_params.modalities == 'L':
                translator[0].eval()
                translator[1].eval()
                discriminator[0].eval()
                discriminator[1].eval()
            else:
                translator.eval()
                discriminator.eval()

        loader = test_loader if test else valid_loader
        total_loss = 0.0

        results = []
        truths = []
        mask = []

        with torch.no_grad():
            for i_batch, (batch_data, masks, labels) in enumerate(loader):
                # 解析批次数据
                if hyp_params.modalities == 'L':
                    text, audio, video = batch_data
                elif hyp_params.modalities == 'A':
                    audio, video = batch_data
                    text = None
                elif hyp_params.modalities == 'V':
                    video, audio = batch_data
                    text = None
                else:
                    audio, video = batch_data
                    text = None

                # 数据移至GPU
                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        if text is not None:
                            text = text.cuda()
                        audio = audio.cuda() if audio is not None else None
                        video = video.cuda() if video is not None else None
                        masks = masks.cuda()
                        labels = labels.cuda()

                # 应用掩码
                batch_size = labels.size(0)
                if audio is not None:
                    masks_audio = masks.unsqueeze(-1).expand(-1, hyp_params.a_len, hyp_params.orig_d_a)
                    audio = audio * masks_audio
                if video is not None:
                    masks_video = masks.unsqueeze(-1).expand(-1, hyp_params.v_len, hyp_params.orig_d_v)
                    video = video * masks_video

                net = nn.DataParallel(model) if hyp_params.distribute else model
                trans_loss = torch.tensor(0.0, device=labels.device)
                fake_v, fake_a = None, None

                # 生成缺失模态
                if hyp_params.modalities != 'AV':
                    if hyp_params.modalities == 'L':
                        trans_net1 = nn.DataParallel(translator[0]) if hyp_params.distribute else translator[0]
                        trans_net2 = nn.DataParallel(translator[1]) if hyp_params.distribute else translator[1]

                        if not test:  # 验证模式
                            trans_outputs1 = trans_net1(text, audio, 'valid')
                            trans_outputs2 = trans_net2(text, video, 'valid')
                            fake_a, fake_v = trans_outputs1[0], trans_outputs2[0]
                            trans_loss = trans_criterion(fake_a, audio) + trans_criterion(fake_v, video)
                        else:  # 测试模式（自回归生成）
                            # 生成A模态
                            fake_a = torch.Tensor().cuda() if hyp_params.use_cuda else torch.Tensor()
                            for i in range(hyp_params.a_len):
                                if i == 0:
                                    outputs = trans_net1(text, audio, 'test', eval_start=True)
                                    fake_a_token = outputs[0][:, [-1]]
                                else:
                                    outputs = trans_net1(text, fake_a, 'test')
                                    fake_a_token = outputs[0][:, [-1]]
                                fake_a = torch.cat((fake_a, fake_a_token), dim=1)

                            # 生成V模态
                            fake_v = torch.Tensor().cuda() if hyp_params.use_cuda else torch.Tensor()
                            for i in range(hyp_params.v_len):
                                if i == 0:
                                    outputs = trans_net2(text, video, 'test', eval_start=True)
                                    fake_v_token = outputs[0][:, [-1]]
                                else:
                                    outputs = trans_net2(text, fake_v, 'test')
                                    fake_v_token = outputs[0][:, [-1]]
                                fake_v = torch.cat((fake_v, fake_v_token), dim=1)

                    elif hyp_params.modalities == 'A':
                        trans_net = nn.DataParallel(translator) if hyp_params.distribute else translator
                        if not test:
                            outputs = trans_net(audio, video, 'valid')
                            fake_v = outputs[0]
                            trans_loss = trans_criterion(fake_v, video)
                        else:
                            fake_v = torch.Tensor().cuda() if hyp_params.use_cuda else torch.Tensor()
                            for i in range(hyp_params.v_len):
                                if i == 0:
                                    outputs = trans_net(audio, video, 'test', eval_start=True)
                                    fake_v_token = outputs[0][:, [-1]]
                                else:
                                    outputs = trans_net(audio, fake_v, 'test')
                                    fake_v_token = outputs[0][:, [-1]]
                                fake_v = torch.cat((fake_v, fake_v_token), dim=1)

                    elif hyp_params.modalities == 'V':
                        trans_net = nn.DataParallel(translator) if hyp_params.distribute else translator
                        if not test:
                            outputs = trans_net(video, audio, 'valid')
                            fake_a = outputs[0]
                            trans_loss = trans_criterion(fake_a, audio)
                        else:
                            fake_a = torch.Tensor().cuda() if hyp_params.use_cuda else torch.Tensor()
                            for i in range(hyp_params.a_len):
                                if i == 0:
                                    outputs = trans_net(video, audio, 'test', eval_start=True)
                                    fake_a_token = outputs[0][:, [-1]]
                                else:
                                    outputs = trans_net(video, fake_a, 'test')
                                    fake_a_token = outputs[0][:, [-1]]
                                fake_a = torch.cat((fake_a, fake_a_token), dim=1)

                # 主模型预测
                if hyp_params.modalities == 'L':
                    outputs = net(text, fake_a, fake_v)
                    preds = outputs[0]
                elif hyp_params.modalities == 'A':
                    outputs = net(audio, fake_v)
                    preds = outputs[0]
                elif hyp_params.modalities == 'V':
                    outputs = net(fake_a, video)
                    preds = outputs[0]
                else:
                    outputs = net(audio, video)
                    preds = outputs[0]

                # 计算损失
                raw_loss = criterion(preds.transpose(1, 2), labels)
                if hyp_params.modalities != 'AV' and not test:
                    combined_loss = raw_loss + trans_loss
                else:
                    combined_loss = raw_loss

                total_loss += combined_loss.item() * batch_size

                # 收集结果
                results.append(preds)
                truths.append(labels)
                mask.append(masks)

        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)
        results = torch.cat(results)
        truths = torch.cat(truths)
        mask = torch.cat(mask)
        return avg_loss, results, truths, mask

    # 打印模型参数数量
    if hyp_params.modalities != 'AV':
        if hyp_params.modalities == 'L':
            mgm_parameter = sum(p.nelement() for p in translator[0].parameters()) + \
                            sum(p.nelement() for p in translator[1].parameters())
            disc_parameter = sum(p.nelement() for p in discriminator[0].parameters()) + \
                             sum(p.nelement() for p in discriminator[1].parameters())
        else:
            mgm_parameter = sum(p.nelement() for p in translator.parameters())
            disc_parameter = sum(p.nelement() for p in discriminator.parameters())
        print(f'Trainable Parameters for Multimodal Generation Model (MGM): {mgm_parameter}...')
        print(f'Trainable Parameters for Discriminator: {disc_parameter}...')
    mum_parameter = sum(p.nelement() for p in model.parameters())
    print(f'Trainable Parameters for Multimodal Understanding Model (MUM): {mum_parameter}...')

    # 训练循环
    best_valid = 1e8
    loop = tqdm(range(1, hyp_params.num_epochs + 1), leave=False)
    for epoch in loop:
        loop.set_description(f'Epoch {epoch:2d}/{hyp_params.num_epochs}')
        start = time.time()
        train(model, translator, optimizer, criterion)
        val_loss, _, _, _ = evaluate(model, translator, criterion, test=False)

        end = time.time()
        duration = end - start
        scheduler.step(val_loss)  # 根据验证损失调整学习率

        # 保存最佳模型
        if val_loss < best_valid:
            if hyp_params.modalities != 'AV':
                if hyp_params.modalities == 'L':
                    save_model(hyp_params, translator[0], name='TRANSLATOR_A')
                    save_model(hyp_params, translator[1], name='TRANSLATOR_V')
                    save_model(hyp_params, discriminator[0], name='DISCRIMINATOR_A')
                    save_model(hyp_params, discriminator[1], name='DISCRIMINATOR_V')
                else:
                    save_model(hyp_params, translator, name='TRANSLATOR')
                    save_model(hyp_params, discriminator, name='DISCRIMINATOR')
            save_model(hyp_params, model, name=hyp_params.name)
            best_valid = val_loss

    # 加载最佳模型并测试
    if hyp_params.modalities != 'AV':
        if hyp_params.modalities == 'L':
            translator = (
                load_model(hyp_params, name='TRANSLATOR_A'),
                load_model(hyp_params, name='TRANSLATOR_V')
            )
            discriminator = (
                load_model(hyp_params, name='DISCRIMINATOR_A'),
                load_model(hyp_params, name='DISCRIMINATOR_V')
            )
        else:
            translator = load_model(hyp_params, name='TRANSLATOR')
            discriminator = load_model(hyp_params, name='DISCRIMINATOR')
    model = load_model(hyp_params, name=hyp_params.name)
    _, results, truths, mask = evaluate(model, translator, criterion, test=True)

    acc = eval_ur_funny(results, truths, mask)
    return acc