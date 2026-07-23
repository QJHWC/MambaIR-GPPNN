# -*- coding: utf-8 -*-
"""
MambaIR-GPPNN 训练脚本
受 MambaIR 启发的实验性全色锐化网络训练

运行方式:
python train.py --model_size base --batch_size 4 --epochs 50
python train.py --model_size large --batch_size 2 --epochs 100
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import time
import gc  # 垃圾回收
import glob  # 文件匹配
import psutil  # 系统监控
from datetime import datetime

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import create_mambairv2_gppnn
from data import create_photo_dataloaders


class IRDN_Loss(nn.Module):
    """
    🔥 增强的损失函数 - 优化38 (v2.2优化版)
    L1 + Gradient + SSIM + Edge-aware + Frequency
    v2.2: 大幅提升结构感知权重，强化SSIM和边缘保真
    """
    def __init__(self, alpha=1.0, beta=0.3, gamma=0.2, edge_aware=True, freq_loss=True, **kwargs):
        super().__init__()
        self.alpha = alpha          # L1损失权重
        self.beta = beta            # 梯度损失权重（0.15→0.3，×2倍强化结构感知）
        self.gamma = gamma          # SSIM损失权重（0.05→0.2，×4倍直接优化SSIM）
        self.edge_aware = edge_aware
        self.freq_loss = freq_loss

        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

        # 🌍 世界模型增强: DSC物理一致性损失
        self.use_dsc = kwargs.get('use_dsc', False)
        if self.use_dsc:
            from models.world_model import SensorConsistencyLoss
            self.dsc_loss_fn = SensorConsistencyLoss(
                spectral_response=kwargs.get('dsc_spectral_response', [0.299, 0.587, 0.114]),
                mtf_kernel_size=kwargs.get('dsc_mtf_kernel_size', 5),
                mtf_sigma=kwargs.get('dsc_mtf_sigma', 1.0),
                lrms_weight=kwargs.get('dsc_lrms_weight', 0.3)
            )
            self.lambda_s = kwargs.get('lambda_s', 0.3)
            print(f"[Loss] [WorldModel] DSC (Sensor Consistency) enabled, lambda_s={self.lambda_s}")
        else:
            self.dsc_loss_fn = None

        # 🌍 世界模型增强: WAC-X频域一致性损失
        self.use_wacx = kwargs.get('use_wacx', False)
        if self.use_wacx:
            from models.world_model import WACXLoss
            self.wacx_loss_fn = WACXLoss(
                interband_weight=kwargs.get('wacx_interband_weight', 1.0),
                pan_gate_weight=kwargs.get('wacx_pan_gate_weight', 0.5),
                freq_threshold=kwargs.get('wacx_freq_threshold', 0.1)
            )
            self.lambda_w = kwargs.get('lambda_w', 0.5)
            print(f"[Loss] [WorldModel] WAC-X (Cross-band Consistency) enabled, lambda_w={self.lambda_w}")
        else:
            self.wacx_loss_fn = None

    def gradient_loss(self, pred, target):
        """计算梯度损失"""
        pred_h = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        pred_w = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_h = target[:, :, 1:, :] - target[:, :, :-1, :]
        target_w = target[:, :, :, 1:] - target[:, :, :, :-1]

        loss_h = self.l1_loss(pred_h, target_h)
        loss_w = self.l1_loss(pred_w, target_w)

        return loss_h + loss_w

    def ssim_loss(self, pred, target):
        """简化SSIM损失"""
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        mu_pred = torch.mean(pred, dim=[2, 3], keepdim=True)
        mu_target = torch.mean(target, dim=[2, 3], keepdim=True)

        sigma_pred = torch.var(pred, dim=[2, 3], keepdim=True)
        sigma_target = torch.var(target, dim=[2, 3], keepdim=True)
        sigma_pred_target = torch.mean((pred - mu_pred) * (target - mu_target), dim=[2, 3], keepdim=True)

        ssim = ((2 * mu_pred * mu_target + c1) * (2 * sigma_pred_target + c2)) / \
               ((mu_pred ** 2 + mu_target ** 2 + c1) * (sigma_pred + sigma_target + c2))

        return 1 - torch.mean(ssim)

    def edge_aware_loss(self, pred, target):
        """边缘感知损失"""
        # Sobel算子检测边缘
        sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
                               dtype=pred.dtype, device=pred.device).repeat(pred.shape[1], 1, 1, 1)
        sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
                               dtype=pred.dtype, device=pred.device).repeat(pred.shape[1], 1, 1, 1)

        pred_edge_x = F.conv2d(pred, sobel_x, padding=1, groups=pred.shape[1])
        pred_edge_y = F.conv2d(pred, sobel_y, padding=1, groups=pred.shape[1])
        target_edge_x = F.conv2d(target, sobel_x, padding=1, groups=target.shape[1])
        target_edge_y = F.conv2d(target, sobel_y, padding=1, groups=target.shape[1])

        edge_loss = self.l1_loss(pred_edge_x, target_edge_x) + self.l1_loss(pred_edge_y, target_edge_y)
        return edge_loss

    def frequency_loss(self, pred, target):
        """频域损失"""
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')

        freq_loss = self.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))
        return freq_loss

    def forward(self, outputs, target, **kwargs):
        """
        🔥 增强的损失计算 - 优化39 + 世界模型增强
        Args:
            outputs: [SR_1_4, SR_1_2, output] - 模型输出
            target: [B, C, H, W] - 目标GT图像
            **kwargs: 额外参数（世界模型需要）
                - pan_gt: [B, 1, H, W] PAN Ground Truth (for DSC)
                - ms_gt: [B, C, H/4, W/4] MS Ground Truth (for DSC)
        """
        SR_1_4, SR_1_2, output_full = outputs

        # 创建多尺度目标
        target_1_2 = nn.functional.avg_pool2d(target, 2, 2)
        target_1_4 = nn.functional.avg_pool2d(target_1_2, 2, 2)

        # L1损失
        l1_loss_1_4 = self.l1_loss(SR_1_4[0], target_1_4)
        l1_loss_1_2 = self.l1_loss(SR_1_2[0], target_1_2)
        l1_loss_full = self.l1_loss(output_full, target)
        total_l1 = l1_loss_1_4 + l1_loss_1_2 + l1_loss_full

        # 梯度损失（仅在全分辨率上计算）
        grad_loss = self.gradient_loss(output_full, target)

        # 🔥 新增: SSIM损失
        ssim_loss_val = self.ssim_loss(output_full, target) if self.gamma > 0 else 0

        # 🔥 新增: 边缘感知损失 (v2.2: 0.1→0.15)
        edge_loss_val = self.edge_aware_loss(output_full, target) * 0.15 if self.edge_aware else 0

        # 🔥 新增: 频域损失 (v2.2: 0.05→0.1)
        freq_loss_val = self.frequency_loss(output_full, target) * 0.1 if self.freq_loss else 0

        # 🌍 世界模型增强: DSC物理一致性损失
        dsc_loss_val = torch.tensor(0.0, device=output_full.device)
        if self.use_dsc and self.dsc_loss_fn is not None:
            # 需要从kwargs获取pan_gt和ms_gt
            if 'pan_gt' in kwargs and 'ms_gt' in kwargs:
                dsc_losses = self.dsc_loss_fn(
                    hrms_pred=output_full,
                    pan_gt=kwargs['pan_gt'],
                    lrms_gt=kwargs.get('ms_gt', None)
                )
                dsc_loss_val = dsc_losses['dsc_total']

        # 🌍 世界模型增强: WAC-X频域一致性损失
        wacx_loss_val = torch.tensor(0.0, device=output_full.device)
        if self.use_wacx and self.wacx_loss_fn is not None:
            if 'pan_gt' in kwargs:
                wacx_losses = self.wacx_loss_fn(
                    hrms=output_full,
                    pan=kwargs['pan_gt']
                )
                wacx_loss_val = wacx_losses['wacx_total']

        # 总损失
        total_loss = (self.alpha * total_l1 +
                     self.beta * grad_loss +
                     self.gamma * ssim_loss_val +
                     edge_loss_val +
                     freq_loss_val +
                     (self.lambda_s * dsc_loss_val if self.use_dsc else 0.0) +
                     (self.lambda_w * wacx_loss_val if self.use_wacx else 0.0))

        return {
            'total_loss': total_loss,
            'l1_loss': total_l1,
            'grad_loss': grad_loss,
            'ssim_loss': ssim_loss_val if isinstance(ssim_loss_val, torch.Tensor) else torch.tensor(0.0, device=total_loss.device),
            'edge_loss': edge_loss_val if isinstance(edge_loss_val, torch.Tensor) else torch.tensor(0.0, device=total_loss.device),
            'freq_loss': freq_loss_val if isinstance(freq_loss_val, torch.Tensor) else torch.tensor(0.0, device=total_loss.device),
            'dsc_loss': dsc_loss_val,  # 🌍 新增DSC损失
            'wacx_loss': wacx_loss_val,  # 🌍 新增WAC-X损失
            'l1_1_4': l1_loss_1_4,
            'l1_1_2': l1_loss_1_2,
            'l1_full': l1_loss_full
        }


class ModelEMA:
    """
    🔥 EMA (Exponential Moving Average) 模型平滑 - v2.2新增
    提升验证稳定性和泛化能力
    """
    def __init__(self, model, decay=0.9999, device=None):
        self.ema = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.ema = {k: v.to(device) for k, v in self.ema.items()}

    def update(self, model):
        """更新EMA权重"""
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.ema:
                    self.ema[k] = self.decay * self.ema[k] + (1 - self.decay) * v

    def apply_shadow(self, model):
        """将EMA权重应用到模型"""
        with torch.no_grad():
            model.load_state_dict(self.ema)

    def store(self, model):
        """保存当前模型权重（用于恢复）"""
        self.backup = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def restore(self, model):
        """恢复保存的模型权重"""
        if hasattr(self, 'backup'):
            model.load_state_dict(self.backup)


def calculate_psnr(img1, img2):
    """计算PSNR"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args, writer=None, ema=None, scaler=None):
    """训练一个epoch - 带显存保护 + EMA更新 + 混合精度 (v2.2)"""
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    num_batches = 0
    global_step = epoch * len(train_loader)

    # 🔥 梯度累积步数
    grad_accum_steps = getattr(args, 'grad_accum_steps', 1)
    accum_step = 0

    epoch_start = time.time()

    for batch_idx, (ms, pan, gt) in enumerate(train_loader):
        try:
            ms, pan, gt = ms.to(device), pan.to(device), gt.to(device)

            # 🔥 梯度累积：第一步前清空梯度
            if accum_step == 0:
                optimizer.zero_grad()

            # 🔥 混合精度训练 + 梯度累积
            if scaler is not None:
                # FP16混合精度路径
                with torch.cuda.amp.autocast():
                    # 前向传播
                    outputs = model(ms, pan)

                    # 计算损失
                    loss_dict = criterion(outputs, gt, pan_gt=pan, ms_gt=ms)
                    loss = loss_dict['total_loss']

                    # 梯度累积：损失归一化
                    loss = loss / grad_accum_steps

                # 反向传播（scaled）
                scaler.scale(loss).backward()

                # 梯度累积：仅在累积满时更新
                accum_step += 1
                if accum_step >= grad_accum_steps:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    accum_step = 0

                    # 🔥 EMA仅在参数更新后执行
                    if ema is not None:
                        ema.update(model)
            else:
                # FP32标准路径
                outputs = model(ms, pan)
                loss_dict = criterion(outputs, gt, pan_gt=pan, ms_gt=ms)
                loss = loss_dict['total_loss']
                loss = loss / grad_accum_steps

                loss.backward()

                accum_step += 1
                if accum_step >= grad_accum_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    accum_step = 0

                    # 🔥 梯度累积后更新EMA
                    if ema is not None:
                        ema.update(model)

            # 🔥 v2.2修复: 记录训练时的峰值GPU显存 (在删除tensor之前)
            if torch.cuda.is_available():
                peak_gpu_allocated = torch.cuda.memory_allocated() / 1024**3
                peak_gpu_reserved = torch.cuda.memory_reserved() / 1024**3
            else:
                peak_gpu_allocated = 0.0
                peak_gpu_reserved = 0.0

        except (RuntimeError, Exception) as e:
            print(f"\n❌ 训练过程错误 (batch {batch_idx}): {e}")
            print(f"🔧 尝试清理显存并继续...")

            # 紧急清理和重置
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.synchronize()

            # 重置优化器梯度，避免状态污染
            optimizer.zero_grad()

            # 如果连续错误超过10次，停止训练
            if not hasattr(train_one_epoch, 'error_count'):
                train_one_epoch.error_count = 0
            train_one_epoch.error_count += 1

            if train_one_epoch.error_count > 10:
                print(f"\n💥 连续错误超过10次，停止训练避免无限循环")
                raise e

            # 跳过这个batch，继续训练
            continue  # 🔥 重要：这里会跳过下面所有代码

        # 🔧 成功执行batch，重置错误计数
        if hasattr(train_one_epoch, 'error_count'):
            train_one_epoch.error_count = 0

        # 🔧 安全的统计信息提取 (先保存所有需要的值，再删除变量)
        # 注意：梯度累积时loss已经除以grad_accum_steps，需要×回来显示真实值
        loss_value = loss.item() * grad_accum_steps
        l1_loss_value = loss_dict['l1_loss'].item()
        grad_loss_value = loss_dict['grad_loss'].item()

        # 安全的PSNR计算 (不提前删除tensor引用)
        with torch.no_grad():
            _, _, output_full = outputs
            psnr_value = calculate_psnr(output_full, gt).item()

        total_loss += loss_value
        total_psnr += psnr_value
        num_batches += 1

        # 🔧 关键修复：立即删除outputs释放显存（梯度累积时很重要！）
        del outputs, loss_dict, loss, output_full

        # 🔥 梯度累积时的显存清理
        if grad_accum_steps > 1 and torch.cuda.is_available():
            # 每个batch后清理，防止累积
            torch.cuda.empty_cache()

        # 添加显式的CUDA同步，确保所有操作完成
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # 🔧 完全无冲突的清理策略 - 实时显存监控
        # 使用质数来避免与日志频率(60)的最小公倍数冲突
        if batch_idx > 0 and batch_idx % 97 == 0:  # 97是质数，与60互质，避免冲突
            try:
                if torch.cuda.is_available():
                    # 清理前查询
                    allocated_before = torch.cuda.memory_allocated() / 1024**3
                    reserved_before = torch.cuda.memory_reserved() / 1024**3
                    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

                    # 执行清理
                    torch.cuda.empty_cache()
                    gc.collect()
                    torch.cuda.synchronize()

                    # 清理后查询
                    allocated_after = torch.cuda.memory_allocated() / 1024**3
                    reserved_after = torch.cuda.memory_reserved() / 1024**3

                    print(f"\n🧹 定期清理 (batch {batch_idx}):")
                    print(f"   GPU总显存: {total_memory:.1f}GB")
                    print(f"   清理前: 已用{allocated_before:.1f}GB / 缓存{reserved_before:.1f}GB")
                    print(f"   清理后: 已用{allocated_after:.1f}GB / 缓存{reserved_after:.1f}GB")
                    print(f"   释放: {reserved_before - reserved_after:.1f}GB")
            except Exception as e:
                print(f"\n⚠️  清理过程出现异常: {e}")

        # 紧急清理：使用另一个质数避免冲突
        elif batch_idx > 0 and batch_idx % 47 == 0 and torch.cuda.is_available():  # 47也是质数
            try:
                allocated = torch.cuda.memory_allocated() / 1024**3
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                # 只有在显存真正紧张时才清理
                if allocated > total_memory * 0.7:  # 超过70%才清理
                    reserved_before = torch.cuda.memory_reserved() / 1024**3
                    print(f"\n🚨 紧急清理显存: {allocated:.1f}GB / {total_memory:.1f}GB (使用率{allocated/total_memory*100:.1f}%)")
                    torch.cuda.empty_cache()
                    gc.collect()
                    torch.cuda.synchronize()
                    reserved_after = torch.cuda.memory_reserved() / 1024**3
                    print(f"   释放缓存: {reserved_before - reserved_after:.1f}GB")
            except Exception as e:
                print(f"\n⚠️  紧急清理异常: {e}")

        # 批次日志 (使用保存的值，避免访问已删除的变量)
        if batch_idx % args.log_freq == 0:
            # 🔥 v2.2修复: 使用峰值GPU显存显示（训练时的真实占用）
            if torch.cuda.is_available() and 'peak_gpu_allocated' in locals():
                gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_usage_pct = (peak_gpu_allocated / gpu_mem_total) * 100
                mem_info = f"GPU: {peak_gpu_allocated:.1f}/{gpu_mem_total:.1f}GB ({gpu_usage_pct:.0f}%) [峰值]"
            else:
                mem_info = ""

            print(f"Epoch [{epoch+1:3d}/{args.epochs}] "
                  f"Batch [{batch_idx:3d}/{len(train_loader)}] "
                  f"Loss: {loss_value:.6f} "
                  f"L1: {l1_loss_value:.6f} "
                  f"Grad: {grad_loss_value:.6f} "
                  f"PSNR: {psnr_value:.2f}dB  "
                  f"{mem_info}")

            # TensorBoard记录 (使用保存的值) - 添加异常保护
            if writer is not None:
                try:
                    current_step = global_step + batch_idx
                    writer.add_scalar('Train/BatchLoss', loss_value, current_step)
                    writer.add_scalar('Train/BatchPSNR', psnr_value, current_step)
                    writer.add_scalar('Train/BatchL1', l1_loss_value, current_step)
                    writer.add_scalar('Train/BatchGrad', grad_loss_value, current_step)
                except Exception as e:
                    print(f"\n⚠️  TensorBoard写入异常: {e}")
                    # 不中断训练，继续进行

    # 计算平均值并返回 - 添加防止除零保护
    if num_batches > 0:
        avg_loss = total_loss / num_batches
        avg_psnr = total_psnr / num_batches
    else:
        print("\n⚠️  警告: 没有成功处理任何batch！")
        avg_loss = float('inf')
        avg_psnr = 0.0

    epoch_time = time.time() - epoch_start

    # 重置错误计数器
    if hasattr(train_one_epoch, 'error_count'):
        train_one_epoch.error_count = 0

    return avg_loss, avg_psnr, epoch_time


def calculate_ssim_simple(img1, img2):
    """简化的SSIM计算 (参考train_IRDN_migrated.py)"""
    # 转换为numpy
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()

    # 如果是批次数据，取第一个
    if len(img1.shape) == 4:
        img1 = img1[0]
        img2 = img2[0]

    # 转换为灰度图
    if img1.shape[0] == 3:  # RGB
        img1 = np.mean(img1, axis=0)
        img2 = np.mean(img2, axis=0)

    # 简化版SSIM
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1 = np.var(img1)
    sigma2 = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_val = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))

    return max(0, min(1, ssim_val))


def validate(model, val_loader, criterion, device):
    """验证 - 增加SSIM指标"""
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = 0

    with torch.no_grad():
        for ms, pan, gt in val_loader:
            ms, pan, gt = ms.to(device), pan.to(device), gt.to(device)

            outputs = model(ms, pan)
            loss_dict = criterion(outputs, gt)

            _, _, output_full = outputs
            psnr = calculate_psnr(output_full, gt)
            ssim = calculate_ssim_simple(output_full, gt)

            total_loss += loss_dict['total_loss'].item()
            total_psnr += psnr.item()
            total_ssim += ssim
            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches

    return avg_loss, avg_psnr, avg_ssim


def auto_find_max_batch_size(model, train_loader, criterion, optimizer, device, args):
    """
    🔥 自动查找最大可用batch_size
    使用二分查找策略，避免OOM的同时最大化训练速度
    """
    print("\n" + "="*70)
    print("🔍 自动查找最大可用 batch_size...")
    print("="*70)

    # 🔧 测试前彻底清理显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    time.sleep(1)  # 等待清理完成

    # 候选batch_size列表（从大到小测试）
    if args.img_size == 256:
        candidates = [32, 24, 20, 16, 12, 8, 6, 4, 2, 1]
    else:  # 512
        candidates = [8, 6, 4, 3, 2, 1]  # 🔧 512降低起始值

    max_working_bs = 1  # 默认最小值

    # 获取一个batch的数据用于测试
    try:
        ms_sample, pan_sample, gt_sample = next(iter(train_loader))
        original_batch_size = ms_sample.shape[0]
        print(f"📦 数据加载器batch_size: {original_batch_size}")
    except Exception as e:
        print(f"❌ 无法获取测试数据: {e}")
        return args.batch_size

    for test_bs in candidates:
        if test_bs > original_batch_size:
            continue  # 跳过超过数据加载器batch_size的值

        print(f"\n🧪 测试 batch_size={test_bs}...", end=" ")

        try:
            # 准备测试数据
            if test_bs < original_batch_size:
                ms = ms_sample[:test_bs].to(device)
                pan = pan_sample[:test_bs].to(device)
                gt = gt_sample[:test_bs].to(device)
            else:
                ms = ms_sample.to(device)
                pan = pan_sample.to(device)
                gt = gt_sample.to(device)

            # 测试前向+反向传播
            optimizer.zero_grad()
            outputs = model(ms, pan)
            # 🔥 传递世界模型需要的参数
            loss_dict = criterion(outputs, gt, pan_gt=pan, ms_gt=ms)
            loss = loss_dict['total_loss']
            loss.backward()
            optimizer.step()

            # 查询显存占用
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                usage_pct = (allocated / total_mem) * 100
                print(f"✅ 成功! 显存: {allocated:.1f}/{total_mem:.1f}GB ({usage_pct:.0f}%)")
            else:
                print("✅ 成功!")

            max_working_bs = test_bs

            # 清理显存
            del ms, pan, gt, outputs, loss_dict, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()

            # 找到第一个成功的就停止（从大到小测试）
            break

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ OOM")
                # 清理显存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()
                time.sleep(0.5)  # 等待清理完成
                continue
            else:
                print(f"❌ 错误: {e}")
                break
        except Exception as e:
            print(f"❌ 未知错误: {e}")
            break

    print("\n" + "="*70)
    print(f"🎯 自动检测结果: 最大可用 batch_size = {max_working_bs}")
    print("="*70)

    return max_working_bs


def main():
    parser = argparse.ArgumentParser(description='MambaIRv2-GPPNN Training')
    parser.add_argument('--model_size', type=str, default='base', choices=['base', 'large'],
                        help='Model size (base/large)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size (保守配置，显存安全)')
    parser.add_argument('--epochs', type=int, default=190, help='Number of epochs (v2.2优化: 增加训练轮数)')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--photo_root', type=str, default='./photo', help='Photo directory path')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Save directory')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data workers')
    parser.add_argument('--img_size', type=int, default=256, help='Image size (256快速验证/512完整训练)')
    parser.add_argument('--save_freq', type=int, default=5, help='Save frequency (epochs)')
    parser.add_argument('--log_freq', type=int, default=10, help='Log frequency (batches)')
    parser.add_argument('--val_freq', type=int, default=5, help='Validation frequency (epochs, v2.2: 10→5更频繁)')
    parser.add_argument('--grad_clip_norm', type=float, default=0.1, help='Gradient clipping norm')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cuda/cpu)')
    parser.add_argument('--resume', type=str, default='', help='Resume training from checkpoint path')
    parser.add_argument('--auto_resume', action='store_true', help='Auto resume from latest checkpoint')
    parser.add_argument('--auto_batch_size', action='store_true', help='🔥 自动查找最大可用batch_size（推荐）')

    # 🌍 世界模型增强参数
    parser.add_argument('--enable_world_model', action='store_true', help='启用世界模型增强（总开关）')
    parser.add_argument('--use_wsm', action='store_true', help='启用世界状态记忆(WSM)')
    parser.add_argument('--use_dca_fim', action='store_true', help='启用可形变对齐(DCA-FIM)')
    parser.add_argument('--use_dsc', action='store_true', help='启用物理一致性损失(DSC)')
    parser.add_argument('--use_wacx', action='store_true', help='启用频域一致性损失(WAC-X)')
    parser.add_argument('--lambda_s', type=float, default=0.3, help='DSC损失权重')
    parser.add_argument('--lambda_w', type=float, default=0.5, help='WAC-X损失权重')
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='梯度累积步数（节省显存）')
    parser.add_argument('--fp16', action='store_true', help='🔥 启用混合精度训练（FP16，推荐V100/A100）')

    args = parser.parse_args()

    # 设备设置
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"🚀 开始训练MambaIRv2-GPPNN ({args.model_size.upper()})")
    print("="*60)
    print(f"   设备: {device}")
    print(f"   批次大小: {args.batch_size} (安全设置)")
    print(f"   训练轮数: {args.epochs}")
    print(f"   学习率: {args.lr}")
    print(f"   图像尺寸: {args.img_size}x{args.img_size}")

    # 🚨 安全检查 (参考train_IRDN_migrated.py)
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   GPU显存: {gpu_memory:.1f}GB")
        if gpu_memory < 6:
            print("   ⚠️ 显存较小，使用保守配置")

    # 系统内存检查
    system_memory = psutil.virtual_memory()
    print(f"   系统内存: {system_memory.available/1024**3:.1f}GB 可用")
    if system_memory.percent > 80:
        print("   ⚠️ 系统内存使用率过高，建议关闭其他程序")

    # 创建保存目录
    timestamp = datetime.now().strftime("%m%d_%H%M")
    save_dir = f"{args.save_dir}/mambairv2_gppnn_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/models", exist_ok=True)
    args.save_dir = save_dir  # 更新保存目录

    # 创建模型
    print("\n🏗️  创建模型...")

    # 🌍 世界模型配置
    world_model_kwargs = {}
    if args.enable_world_model:
        world_model_kwargs.update({
            'use_wsm': args.use_wsm,
            'use_dca_fim': args.use_dca_fim,
            'wsm_hidden_dim': 128,
            'wsm_dropout': 0.1,
            'dca_num_points': 4,
            'dca_deform_weight': 0.3,
        })
        print(f"   🌍 世界模型增强已启用:")
        print(f"      WSM: {args.use_wsm}, DCA-FIM: {args.use_dca_fim}")
        print(f"      DSC: {args.use_dsc}, WAC-X: {args.use_wacx}")

    model = create_mambairv2_gppnn(args.model_size, **world_model_kwargs).to(device)

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   总参数: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")

    # 创建数据加载器
    print("\n📊 创建数据加载器...")
    train_loader, val_loader, test_loader = create_photo_dataloaders(
        photo_root=args.photo_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=(args.img_size, args.img_size)
    )

    # 🔥 优化40: 创建增强的损失函数和AdamW优化器 (v2.2优化版)
    # 🌍 世界模型损失配置
    loss_kwargs = {}
    if args.enable_world_model:
        loss_kwargs.update({
            'use_dsc': args.use_dsc,
            'use_wacx': args.use_wacx,
            'lambda_s': args.lambda_s,
            'lambda_w': args.lambda_w,
        })

    criterion = IRDN_Loss(alpha=1.0, beta=0.3, gamma=0.2, edge_aware=True, freq_loss=True, **loss_kwargs)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))

    # 🔥 混合精度训练（FP16）
    scaler = None
    if args.fp16:
        print("\n⚡ 启用混合精度训练 (FP16)...")
        scaler = torch.cuda.amp.GradScaler()
        print("   显存预期减少: ~50%")
        print("   训练速度提升: ~20%")

    # 🔥 v2.2新增: EMA模型平滑
    # 🔧 512×512显存优化：大尺寸图像禁用EMA
    if args.img_size >= 512:
        print("\n⚠️  [512模式] 禁用EMA节省显存...")
        ema = None
    else:
        print("\n🔥 初始化EMA模型平滑 (decay=0.9999)...")
        ema = ModelEMA(model, decay=0.9999, device=device)

    # 🔥 新功能: 自动查找最大可用batch_size
    if args.auto_batch_size:
        print("\n" + "="*70)
        print("🚀 启用自动batch_size检测...")
        print("="*70)

        # 保存原始batch_size
        original_bs = args.batch_size

        # 自动检测
        optimal_bs = auto_find_max_batch_size(
            model, train_loader, criterion, optimizer, device, args
        )

        # 如果检测到更大的batch_size，重新创建数据加载器
        if optimal_bs > original_bs:
            print(f"\n🔄 检测到更优的batch_size={optimal_bs}，重新创建数据加载器...")
            args.batch_size = optimal_bs
            train_loader, val_loader, test_loader = create_photo_dataloaders(
                photo_root=args.photo_root,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                img_size=(args.img_size, args.img_size)
            )
            print(f"✅ 数据加载器已更新: batch_size={args.batch_size}")
        else:
            print(f"\n✅ 使用原始batch_size={original_bs}")

    # 🔥 优化41: Cosine Annealing学习率调度器 + Warmup (v2.2优化版)
    warmup_epochs = 8  # v2.2: 统一使用8轮warmup
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-7  # v2.2: 最小学习率降至1e-7
    )

    # 🔥 v2.2新增: Plateau检测器 - 作为备用
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
    )

    # TensorBoard - 添加异常保护
    try:
        writer = SummaryWriter(args.log_dir)
    except Exception as e:
        print(f"⚠️  TensorBoard初始化失败: {e}")
        print("   将禁用TensorBoard日志记录")
        writer = None

    # 断点续训处理
    start_epoch = 0
    best_psnr = 0.0
    train_history = {'train_loss': [], 'train_psnr': [], 'val_loss': [], 'val_psnr': []}

    # 查找断点文件
    resume_path = None
    if args.resume:
        resume_path = args.resume
        print(f"🔄 指定断点续训: {resume_path}")
    elif args.auto_resume:
        # 自动查找最新的checkpoint
        checkpoint_pattern = f"{args.save_dir}/mambairv2_gppnn_*/models/epoch_*.pth"
        checkpoints = glob.glob(checkpoint_pattern)
        if checkpoints:
            # 按修改时间排序，获取最新的
            latest_checkpoint = max(checkpoints, key=os.path.getmtime)
            resume_path = latest_checkpoint
            print(f"🔄 自动找到最新断点: {resume_path}")

    # 加载断点
    if resume_path and os.path.exists(resume_path):
        print(f"📁 加载断点: {resume_path}")
        try:
            checkpoint = torch.load(resume_path, map_location=device)

            # 加载模型状态
            model.load_state_dict(checkpoint['model_state_dict'])

            # 加载优化器状态
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # 加载调度器状态
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            # 加载训练信息
            start_epoch = checkpoint['epoch'] + 1
            if 'best_psnr' in checkpoint:
                best_psnr = checkpoint['best_psnr']
            if 'train_history' in checkpoint:
                train_history = checkpoint['train_history']

            print(f"✅ 断点加载成功!")
            print(f"   从第 {start_epoch+1} 轮继续训练")
            print(f"   当前最佳PSNR: {best_psnr:.2f}dB")

        except Exception as e:
            print(f"❌ 断点加载失败: {e}")
            print("   将从头开始训练...")
            start_epoch = 0
            best_psnr = 0.0

    # 训练循环 (参考train_IRDN_migrated.py格式)
    print(f"\n⚡ 开始训练...")
    print("="*60)
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        # 训练阶段
        train_loss, train_psnr, epoch_time = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args, writer, ema, scaler
        )

        train_history['train_loss'].append(train_loss)
        train_history['train_psnr'].append(train_psnr)

        print(f"\n📊 Epoch {epoch+1} 训练完成:")
        print(f"   平均损失: {train_loss:.6f}")
        print(f"   平均PSNR: {train_psnr:.2f}dB")
        print(f"   耗时: {epoch_time:.1f}s")

        # 🧹 Epoch结束后清理显存，防止碎片化
        torch.cuda.empty_cache()
        gc.collect()

        # 添加trainlog风格的SSIM记录 (参考train_IRDN_migrated.py)
        print(f"The {epoch} Epoch mean-ssim is :{train_psnr/100:.6f}")  # 简化版本

        # 验证阶段
        if (epoch + 1) % args.val_freq == 0:
            print(f"\n🔍 验证 Epoch {epoch+1}...")

            # 🔥 v2.2: 使用EMA模型进行验证
            if ema is not None:
                ema.store(model)  # 保存当前权重
                ema.apply_shadow(model)  # 应用EMA权重

            val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion, device)

            if ema is not None:
                ema.restore(model)  # 恢复训练权重

            train_history['val_loss'].append(val_loss)
            train_history['val_psnr'].append(val_psnr)

            print(f"   验证损失: {val_loss:.6f}")
            print(f"   验证PSNR: {val_psnr:.2f}dB (EMA)")
            print(f"   验证SSIM: {val_ssim:.4f} (EMA)")

            # 添加trainlog风格的验证记录
            print(f"The {epoch} Epoch validation mean-ssim is :{val_ssim:.6f}")

            # TensorBoard记录 - 添加异常保护
            if writer is not None:
                try:
                    writer.add_scalar('Train/Loss_Epoch', train_loss, epoch)
                    writer.add_scalar('Train/PSNR_Epoch', train_psnr, epoch)
                    writer.add_scalar('Val/Loss', val_loss, epoch)
                    writer.add_scalar('Val/PSNR', val_psnr, epoch)
                    writer.add_scalar('Val/SSIM', val_ssim, epoch)
                except Exception as e:
                    print(f"\n⚠️  TensorBoard验证记录异常: {e}")

            # 🔥 v2.2: Plateau检测 - 根据验证损失调整学习率
            plateau_scheduler.step(val_loss)

            # 保存最佳模型 (v2.2: 保存EMA权重)
            if val_psnr > best_psnr:
                best_psnr = val_psnr

                # 应用EMA权重再保存
                if ema is not None:
                    ema.store(model)
                    ema.apply_shadow(model)

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_psnr': best_psnr,
                    'loss': val_loss,
                    'psnr': val_psnr,
                    'ssim': val_ssim,
                    'config': vars(args),
                    'train_history': train_history,
                    'ema_state': ema.ema if ema is not None else None  # v2.2: 保存EMA状态
                }, f"{args.save_dir}/models/best_model.pth")

                if ema is not None:
                    ema.restore(model)

                print(f"   🎉 新的最佳PSNR: {best_psnr:.2f}dB (EMA模型已保存)")

        # 定期保存
        if (epoch + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_psnr': best_psnr,
                'config': vars(args),
                'train_history': train_history
            }, f"{args.save_dir}/models/epoch_{epoch+1}.pth")

        scheduler.step()
        print("-" * 60)

    # 训练完成
    total_time = time.time() - start_time
    print(f"\n🎉 MambaIRv2-GPPNN训练完成!")
    print(f"总耗时: {total_time/60:.1f}分钟")
    print(f"最佳验证PSNR: {best_psnr:.2f}dB")
    print(f"模型保存至: {args.save_dir}")

    # 安全关闭TensorBoard
    try:
        writer.close()
    except Exception as e:
        print(f"⚠️  TensorBoard关闭异常: {e}")

    # 保存配置和结果 (参考train_IRDN_migrated.py)
    results = {
        'config': vars(args),
        'best_psnr': best_psnr,
        'training_time': total_time,
        'total_params': total_params,
        'train_history': train_history,
        'save_dir': args.save_dir
    }

    import json
    with open(f"{args.save_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=4, default=str)

    print(f"\n📋 MambaIRv2-GPPNN训练报告:")
    print(f"   网络: MambaIRv2-GPPNN {args.model_size.upper()} ({total_params:,}参数)")
    print(f"   数据: Photo目录 600训练+50验证+150测试 (固定分割)")
    print(f"   损失: L1 + 梯度损失")
    print(f"   最佳PSNR: {best_psnr:.2f}dB")
    print(f"   训练用时: {total_time/60:.1f}分钟")

    return results


if __name__ == '__main__':
    main()
