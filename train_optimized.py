#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化训练脚本 - 保留完整IRDN损失，优化显存管理
基于train.py但移除导致OOM的功能
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import time
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import create_mambairv2_gppnn
from data import create_photo_dataloaders


# 从train.py复制完整的IRDN_Loss（保留所有损失项）
class IRDN_Loss_Optimized(nn.Module):
    """完整的IRDN损失函数"""
    def __init__(self, alpha=1.0, beta=0.3, gamma=0.2, use_dsc=False, use_wacx=False, lambda_s=0.3, lambda_w=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.l1_loss = nn.L1Loss()

        # 世界模型损失
        self.use_dsc = use_dsc
        self.use_wacx = use_wacx
        self.lambda_s = lambda_s
        self.lambda_w = lambda_w

        if use_dsc:
            from models.world_model import SensorConsistencyLoss
            self.dsc_loss_fn = SensorConsistencyLoss()

        if use_wacx:
            from models.world_model import WACXLoss
            self.wacx_loss_fn = WACXLoss()

    def gradient_loss(self, pred, target):
        """梯度损失"""
        pred_h = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        pred_w = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_h = target[:, :, 1:, :] - target[:, :, :-1, :]
        target_w = target[:, :, :, 1:] - target[:, :, :, :-1]
        return self.l1_loss(pred_h, target_h) + self.l1_loss(pred_w, target_w)

    def ssim_loss(self, pred, target):
        """简化SSIM损失"""
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        mu_pred = torch.mean(pred, dim=[2, 3], keepdim=True)
        mu_target = torch.mean(target, dim=[2, 3], keepdim=True)
        sigma_pred = torch.var(pred, dim=[2, 3], keepdim=True)
        sigma_target = torch.var(target, dim=[2, 3], keepdim=True)
        sigma_pred_target = torch.mean((pred - mu_pred) * (target - mu_target), dim=[2, 3], keepdim=True)

        ssim = ((2 * mu_pred * mu_target + c1) * (2 * sigma_pred_target + c2)) / \
               ((mu_pred ** 2 + mu_target ** 2 + c1) * (sigma_pred + sigma_target + c2))
        return 1 - torch.mean(ssim)

    def forward(self, outputs, target, pan_gt=None, ms_gt=None):
        """完整损失计算"""
        SR_1_4, SR_1_2, output_full = outputs

        # 多尺度L1
        target_1_2 = F.avg_pool2d(target, 2, 2)
        target_1_4 = F.avg_pool2d(target_1_2, 2, 2)

        l1_1_4 = self.l1_loss(SR_1_4[0], target_1_4)
        l1_1_2 = self.l1_loss(SR_1_2[0], target_1_2)
        l1_full = self.l1_loss(output_full, target)
        total_l1 = l1_1_4 + l1_1_2 + l1_full

        # 梯度损失
        grad_loss = self.gradient_loss(output_full, target)

        # SSIM损失
        ssim_loss_val = self.ssim_loss(output_full, target)

        # 基础损失
        total_loss = (self.alpha * total_l1 +
                     self.beta * grad_loss +
                     self.gamma * ssim_loss_val)

        # 世界模型DSC
        if self.use_dsc and pan_gt is not None and ms_gt is not None:
            dsc_dict = self.dsc_loss_fn(output_full, pan_gt, ms_gt)
            total_loss = total_loss + self.lambda_s * dsc_dict['dsc_total']

        # 世界模型WAC-X
        if self.use_wacx and pan_gt is not None:
            wacx_dict = self.wacx_loss_fn(output_full, pan_gt)
            total_loss = total_loss + self.lambda_w * wacx_dict['wacx_total']

        return total_loss


def calculate_psnr(img1, img2):
    """计算PSNR"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, args):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_psnr = 0.0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')

    for ms, pan, gt in pbar:
        ms, pan, gt = ms.to(device), pan.to(device), gt.to(device)

        optimizer.zero_grad()

        # Forward
        outputs = model(ms, pan)

        # Loss
        loss = criterion(outputs, gt, pan_gt=pan, ms_gt=ms)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        # PSNR
        with torch.no_grad():
            _, _, output_full = outputs
            psnr = calculate_psnr(output_full, gt)

        total_loss += loss.item()
        total_psnr += psnr.item()

        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'PSNR': f'{psnr.item():.2f}dB'
        })

        # 立即清理
        del outputs, output_full, loss

    return total_loss / len(train_loader), total_psnr / len(train_loader)


def validate(model, val_loader, device):
    """验证"""
    model.eval()
    total_psnr = 0.0

    with torch.no_grad():
        for ms, pan, gt in val_loader:
            ms, pan, gt = ms.to(device), pan.to(device), gt.to(device)
            _, _, output = model(ms, pan)
            psnr = calculate_psnr(output, gt)
            total_psnr += psnr.item()

    return total_psnr / len(val_loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', type=str, default='base')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--num_workers', type=int, default=2)  # 🔥 降低到2
    parser.add_argument('--save_dir', type=str, default='./checkpoints_optimized')
    parser.add_argument('--val_freq', type=int, default=5)
    parser.add_argument('--save_freq', type=int, default=10)

    # 世界模型
    parser.add_argument('--use_wsm', action='store_true')
    parser.add_argument('--use_dca_fim', action='store_true')
    parser.add_argument('--use_dsc', action='store_true')
    parser.add_argument('--use_wacx', action='store_true')
    parser.add_argument('--lambda_s', type=float, default=0.3)
    parser.add_argument('--lambda_w', type=float, default=0.5)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*70)
    print("优化训练脚本 - 完整IRDN损失 + 世界模型")
    print("="*70)
    print(f"Device: {device}")
    print(f"Model: {args.model_size}")
    print(f"Image Size: {args.img_size}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Num Workers: {args.num_workers}")  # 显示worker数
    print(f"WSM: {args.use_wsm}, DCA: {args.use_dca_fim}")
    print(f"DSC: {args.use_dsc}, WAC-X: {args.use_wacx}")
    print()

    # 创建保存目录
    timestamp = datetime.now().strftime("%m%d_%H%M")
    save_dir = f"{args.save_dir}/{args.model_size}_{args.img_size}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # 创建模型
    print("创建模型...")
    model = create_mambairv2_gppnn(
        args.model_size,
        use_wsm=args.use_wsm,
        use_dca_fim=args.use_dca_fim,
        wsm_hidden_dim=128,
        dca_num_points=4
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {params:,}\n")

    # 数据加载器
    print("创建数据加载器...")
    train_loader, val_loader, test_loader = create_photo_dataloaders(
        photo_root='./photo',
        batch_size=args.batch_size,
        num_workers=args.num_workers,  # 使用较小的worker数
        img_size=(args.img_size, args.img_size)
    )
    print()

    # 完整IRDN损失
    print("创建损失函数...")
    criterion = IRDN_Loss_Optimized(
        alpha=1.0,
        beta=0.3,
        gamma=0.2,
        use_dsc=args.use_dsc,
        use_wacx=args.use_wacx,
        lambda_s=args.lambda_s,
        lambda_w=args.lambda_w
    )

    if args.use_dsc or args.use_wacx:
        print("世界模型损失已启用")
    print()

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 训练
    print("开始训练...")
    print("="*70)
    print()

    best_psnr = 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        # 训练
        train_loss, train_psnr = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args
        )

        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, PSNR: {train_psnr:.2f}dB")

        # 验证
        if (epoch + 1) % args.val_freq == 0:
            val_psnr = validate(model, val_loader, device)
            print(f"  Val PSNR: {val_psnr:.2f}dB")

            # 保存最佳
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'psnr': val_psnr,
                    'config': vars(args)
                }, f"{save_dir}/best_model.pth")
                print(f"  ✅ Best: {best_psnr:.2f}dB")

        # 定期保存
        if (epoch + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"{save_dir}/epoch_{epoch+1}.pth")

        scheduler.step()

        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 完成
    total_time = time.time() - start_time

    print()
    print("="*70)
    print("训练完成!")
    print("="*70)
    print(f"最佳PSNR: {best_psnr:.2f}dB")
    print(f"总时间: {total_time/3600:.1f}小时")
    print(f"保存目录: {save_dir}")
    print("="*70)

    # 保存结果
    with open(f"{save_dir}/results.json", 'w') as f:
        json.dump({
            'best_psnr': best_psnr,
            'total_time_hours': total_time/3600,
            'config': vars(args)
        }, f, indent=4)


if __name__ == '__main__':
    main()

