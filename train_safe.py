#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全训练脚本 - 基于验证成功的train_minimal.py扩展
包含：世界模型、checkpoint保存、验证集评估
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


def calculate_psnr(img1, img2):
    """计算PSNR"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def train_epoch(model, train_loader, criterion_l1, criterion_dsc, optimizer, device, epoch, args):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
    
    for batch_idx, (ms, pan, gt) in enumerate(pbar):
        ms, pan, gt = ms.to(device), pan.to(device), gt.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        _, _, output = model(ms, pan)
        
        # Loss
        l1_loss = criterion_l1(output, gt)
        
        if args.use_dsc and criterion_dsc is not None:
            dsc_dict = criterion_dsc(output, pan, ms)
            dsc_loss = dsc_dict['dsc_total']
            loss = l1_loss + args.lambda_s * dsc_loss
        else:
            loss = l1_loss
        
        # Backward
        loss.backward()
        
        # Update
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        
        # Metrics
        with torch.no_grad():
            psnr = calculate_psnr(output, gt)
        
        total_loss += loss.item()
        total_psnr += psnr.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'PSNR': f'{psnr.item():.2f}dB'
        })
        
        # 清理
        del output, loss, l1_loss
        if torch.cuda.is_available() and batch_idx % 50 == 0:
            torch.cuda.empty_cache()
    
    avg_loss = total_loss / num_batches
    avg_psnr = total_psnr / num_batches
    
    return avg_loss, avg_psnr


def validate(model, val_loader, device):
    """验证"""
    model.eval()
    total_psnr = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for ms, pan, gt in val_loader:
            ms, pan, gt = ms.to(device), pan.to(device), gt.to(device)
            
            _, _, output = model(ms, pan)
            psnr = calculate_psnr(output, gt)
            
            total_psnr += psnr.item()
            num_batches += 1
    
    avg_psnr = total_psnr / num_batches
    return avg_psnr


def main():
    parser = argparse.ArgumentParser(description='Safe Training Script')
    parser.add_argument('--model_size', type=str, default='base')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--save_dir', type=str, default='./checkpoints_safe')
    parser.add_argument('--val_freq', type=int, default=5)
    parser.add_argument('--save_freq', type=int, default=10)
    
    # 世界模型参数
    parser.add_argument('--use_wsm', action='store_true')
    parser.add_argument('--use_dsc', action='store_true')
    parser.add_argument('--lambda_s', type=float, default=0.3)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("安全训练模式")
    print("="*70)
    print(f"Device: {device}")
    print(f"Model: {args.model_size}")
    print(f"Image Size: {args.img_size}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"WSM: {args.use_wsm}")
    print(f"DSC: {args.use_dsc}")
    print()
    
    # 创建保存目录
    timestamp = datetime.now().strftime("%m%d_%H%M")
    save_dir = f"{args.save_dir}/{args.model_size}_{args.img_size}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建模型
    print("创建模型...")
    model = create_mambairv2_gppnn(
        args.model_size,
        use_wsm=args.use_wsm
    ).to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {params:,}\n")
    
    # 数据加载器
    print("创建数据加载器...")
    train_loader, val_loader, test_loader = create_photo_dataloaders(
        photo_root='./photo',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=(args.img_size, args.img_size)
    )
    print()
    
    # 损失和优化器
    criterion_l1 = nn.L1Loss()
    
    criterion_dsc = None
    if args.use_dsc:
        from models.world_model import SensorConsistencyLoss
        criterion_dsc = SensorConsistencyLoss().to(device)
        print(f"DSC Loss enabled, lambda_s={args.lambda_s}\n")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # 训练循环
    print("开始训练...")
    print("="*70)
    
    best_psnr = 0.0
    
    for epoch in range(args.epochs):
        # 训练
        train_loss, train_psnr = train_epoch(
            model, train_loader, criterion_l1, criterion_dsc,
            optimizer, device, epoch, args
        )
        
        print(f"\nEpoch {epoch+1}: Loss={train_loss:.4f}, PSNR={train_psnr:.2f}dB")
        
        # 验证
        if (epoch + 1) % args.val_freq == 0:
            val_psnr = validate(model, val_loader, device)
            print(f"Validation PSNR: {val_psnr:.2f}dB")
            
            # 保存最佳模型
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'psnr': val_psnr,
                    'config': vars(args)
                }, f"{save_dir}/best_model.pth")
                print(f"✅ 新的最佳模型: {best_psnr:.2f}dB")
        
        # 定期保存
        if (epoch + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"{save_dir}/epoch_{epoch+1}.pth")
        
        scheduler.step()
        print("-"*70)
    
    print()
    print("="*70)
    print("训练完成!")
    print(f"最佳PSNR: {best_psnr:.2f}dB")
    print(f"模型保存: {save_dir}")
    print("="*70)
    
    # 保存训练结果
    results = {
        'best_psnr': best_psnr,
        'config': vars(args),
        'save_dir': save_dir
    }
    
    with open(f"{save_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    main()

