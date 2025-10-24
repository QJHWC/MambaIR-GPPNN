# -*- coding: utf-8 -*-
"""
极简训练脚本 - 排查显存问题
移除所有可选功能，只保留核心训练逻辑
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import create_mambairv2_gppnn
from data import create_photo_dataloaders


def train_minimal():
    """极简训练 - 无EMA，无TensorBoard，无复杂损失"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*70)
    print("极简训练模式 - 显存优化版")
    print("="*70)
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print()
    
    # 创建模型（无世界模型）
    print("创建模型...")
    model = create_mambairv2_gppnn('base').to(device)
    model.train()
    
    params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {params:,}\n")
    
    # 创建数据加载器（小batch）
    print("创建数据加载器...")
    train_loader, _, _ = create_photo_dataloaders(
        photo_root='./photo',
        batch_size=1,  # 极小batch
        num_workers=2,
        img_size=(256, 256)
    )
    print()
    
    # 简单优化器（无weight_decay）
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # 极简损失函数
    criterion = nn.L1Loss()
    
    print("开始训练 (10个batch测试)...")
    print("="*70)
    
    for batch_idx, (ms, pan, gt) in enumerate(train_loader):
        if batch_idx >= 10:
            break
        
        try:
            ms, pan, gt = ms.to(device), pan.to(device), gt.to(device)
            
            # 清空梯度
            optimizer.zero_grad()
            
            # Forward
            _, _, output = model(ms, pan)
            
            # Loss
            loss = criterion(output, gt)
            
            # Backward
            loss.backward()
            
            # Update
            optimizer.step()
            
            # 显存统计
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                peak = torch.cuda.max_memory_allocated() / 1024**3
                print(f"Batch {batch_idx}: Loss={loss.item():.4f}, Mem={allocated:.2f}GB, Peak={peak:.2f}GB")
            else:
                print(f"Batch {batch_idx}: Loss={loss.item():.4f}")
            
            # 清理
            del output, loss
            
            # 定期清理显存
            if batch_idx % 3 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        except RuntimeError as e:
            print(f"\n❌ Batch {batch_idx} 失败: {e}")
            
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"   显存: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
                
                torch.cuda.empty_cache()
            
            break
    
    print("\n" + "="*70)
    print("测试完成")
    print("="*70)


if __name__ == "__main__":
    train_minimal()

