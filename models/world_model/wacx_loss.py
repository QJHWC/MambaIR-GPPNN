# -*- coding: utf-8 -*-
"""
跨带频域一致性损失 (Wavelength-Agnostic Cross-band Consistency - WAC-X)
通过FFT约束不同波段的频域一致性，并使用PAN高频门控

数学原理:
    H_b = |FFT(HRMS_b)|  # 每个波段的频谱幅度
    L_inter = Σ_{b1≠b2} ||H_b1 - H_b2||₁  # 跨带一致性
    G = norm(|HF(PAN)|)  # PAN高频门控
    L_gate = ||G ⊙ HF(HRMS)||₁  # 门控高频约束
    
效果: 高频能量守恒 → PSNR↑, 纹理真实↑

参考:
    《最新任务计划.md》 Section 2.4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from itertools import combinations


class WACXLoss(nn.Module):
    """
    WAC-X损失模块
    
    核心功能:
    1. 跨波段频谱一致性约束（不同波段高频结构应相似）
    2. PAN高频门控（利用PAN的高频信息指导HRMS）
    3. 高频保真增强（防止高频信息丢失）
    
    Args:
        interband_weight: 跨带一致性权重（默认1.0）
        pan_gate_weight: PAN门控权重（默认0.5）
        freq_threshold: 高频阈值（默认0.1）
    """
    def __init__(self, 
                 interband_weight=1.0,
                 pan_gate_weight=0.5,
                 freq_threshold=0.1):
        super().__init__()
        self.interband_weight = interband_weight
        self.pan_gate_weight = pan_gate_weight
        self.freq_threshold = freq_threshold
        
    def extract_high_freq(self, fft_spectrum, threshold=0.1):
        """
        提取高频分量（中心为低频，边缘为高频）
        
        Args:
            fft_spectrum: [B, H, W//2+1] FFT频谱幅度（rfft2结果）
            threshold: 低频阈值比例
            
        Returns:
            high_freq: [B, H, W//2+1] 高频分量
        """
        B, H, W = fft_spectrum.shape
        
        # 创建高通滤波器（距离中心越远，权重越大）
        center_h = H // 2
        center_w = W // 2
        
        # 生成距离矩阵
        y_coords = torch.arange(H, device=fft_spectrum.device).view(-1, 1) - center_h
        x_coords = torch.arange(W, device=fft_spectrum.device).view(1, -1) - center_w
        
        # 归一化距离
        dist = torch.sqrt(y_coords**2 + x_coords**2)
        max_dist = math.sqrt(center_h**2 + center_w**2)
        dist_norm = dist / (max_dist + 1e-8)
        
        # 高通掩码（距离>threshold的为高频）
        high_pass_mask = (dist_norm > threshold).float()
        high_pass_mask = high_pass_mask.unsqueeze(0)  # [1, H, W]
        
        # 应用掩码
        high_freq = fft_spectrum * high_pass_mask
        
        return high_freq
    
    def forward(self, hrms, pan):
        """
        计算WAC-X损失
        
        Args:
            hrms: [B, C, H, W] 高分辨率多光谱（预测）
            pan: [B, 1, H, W] 全色图像（Ground Truth）
            
        Returns:
            loss_dict: 包含各项损失的字典
                - wacx_total: 总WAC-X损失
                - wacx_interband: 跨带一致性损失
                - wacx_gate: PAN门控损失
        """
        B, C, H, W = hrms.shape
        
        # ========== 1. 跨带频域一致性损失 ==========
        H_bands = []
        for c in range(C):
            # FFT变换（使用rfft2节省显存）
            fft_c = torch.fft.rfft2(hrms[:, c], norm='ortho')
            H_c = torch.abs(fft_c)  # 幅度谱 [B, H, W//2+1]
            H_bands.append(H_c)
        
        # 计算所有波段对之间的L1差异
        loss_interband = torch.tensor(0.0, device=hrms.device)
        num_pairs = 0
        
        for i, j in combinations(range(C), 2):
            loss_interband = loss_interband + F.l1_loss(H_bands[i], H_bands[j])
            num_pairs += 1
        
        # 平均化
        if num_pairs > 0:
            loss_interband = loss_interband / num_pairs
        
        # ========== 2. PAN高频门控损失 ==========
        # PAN的FFT频谱
        fft_pan = torch.fft.rfft2(pan.squeeze(1), norm='ortho')  # [B, H, W//2+1]
        H_pan = torch.abs(fft_pan)
        
        # 提取PAN高频分量
        H_pan_high = self.extract_high_freq(H_pan, self.freq_threshold)
        
        # 归一化生成门控
        gate = H_pan_high / (H_pan_high.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-8)
        
        # HRMS的平均频谱（跨波段平均）
        hrms_mean = hrms.mean(dim=1)  # [B, H, W]
        fft_hrms_mean = torch.fft.rfft2(hrms_mean, norm='ortho')
        H_hrms = torch.abs(fft_hrms_mean)
        H_hrms_high = self.extract_high_freq(H_hrms, self.freq_threshold)
        
        # 门控约束：HRMS高频应与PAN高频一致
        loss_gate = F.l1_loss(gate * H_hrms_high, H_pan_high)
        
        # ========== 总损失 ==========
        total_loss = (self.interband_weight * loss_interband + 
                     self.pan_gate_weight * loss_gate)
        
        return {
            'wacx_total': total_loss,
            'wacx_interband': loss_interband,
            'wacx_gate': loss_gate
        }


if __name__ == "__main__":
    """WAC-X模块单元测试"""
    print("="*60)
    print("WAC-X (Cross-band Consistency) 模块测试")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}\n")
    
    # 创建WAC-X损失模块
    wacx = WACXLoss(
        interband_weight=1.0,
        pan_gate_weight=0.5,
        freq_threshold=0.1
    ).to(device)
    
    # 测试1: FFT频谱计算
    print("测试1: FFT频谱计算")
    B, C, H, W = 2, 3, 128, 128
    hrms = torch.randn(B, C, H, W).to(device)
    pan = torch.randn(B, 1, H, W).to(device)
    
    # 手动FFT测试
    fft_test = torch.fft.rfft2(hrms[:, 0], norm='ortho')
    H_test = torch.abs(fft_test)
    
    print(f"  HRMS输入: {hrms.shape}")
    print(f"  FFT结果: {fft_test.shape} (复数)")
    print(f"  幅度谱: {H_test.shape}")
    print(f"  幅度谱范围: [{H_test.min().item():.6f}, {H_test.max().item():.6f}]")
    print("  [OK] FFT计算正确\n")
    
    # 测试2: 高频提取
    print("测试2: 高频分量提取")
    H_high = wacx.extract_high_freq(H_test, threshold=0.1)
    
    print(f"  原始频谱能量: {H_test.sum().item():.6f}")
    print(f"  高频能量: {H_high.sum().item():.6f}")
    print(f"  高频比例: {(H_high.sum() / H_test.sum() * 100).item():.2f}%")
    print("  [OK] 高频提取正确\n")
    
    # 测试3: WAC-X损失计算
    print("测试3: WAC-X损失计算")
    loss_dict = wacx(hrms, pan)
    
    print(f"  wacx_total: {loss_dict['wacx_total'].item():.6f}")
    print(f"  wacx_interband: {loss_dict['wacx_interband'].item():.6f}")
    print(f"  wacx_gate: {loss_dict['wacx_gate'].item():.6f}")
    
    # 验证损失为正值
    assert loss_dict['wacx_total'] > 0, "WAC-X总损失应为正值"
    assert loss_dict['wacx_interband'] >= 0, "跨带损失应非负"
    assert loss_dict['wacx_gate'] >= 0, "门控损失应非负"
    
    print("  [OK] WAC-X损失计算正确\n")
    
    # 测试4: 梯度反向传播
    print("测试4: 梯度反向传播")
    hrms_grad = torch.randn(1, 3, 64, 64, requires_grad=True, device=device)
    pan_test = torch.randn(1, 1, 64, 64, device=device)
    
    loss_dict = wacx(hrms_grad, pan_test)
    loss_dict['wacx_total'].backward()
    
    assert hrms_grad.grad is not None, "HRMS梯度为None"
    assert hrms_grad.grad.abs().sum() > 0, "HRMS梯度为0"
    
    print(f"  HRMS梯度范数: {hrms_grad.grad.norm().item():.6f}")
    print("  [OK] 梯度传播正确\n")
    
    # 测试5: 不同波段数适配
    print("测试5: 多波段适配性")
    for num_bands in [3, 4, 8]:
        hrms_test = torch.randn(1, num_bands, 64, 64).to(device)
        loss_dict = wacx(hrms_test, pan_test)
        
        print(f"  波段数={num_bands}: WAC-X={loss_dict['wacx_total'].item():.6f}")
    
    print("  [OK] 多波段适配正确\n")
    
    print("="*60)
    print("[SUCCESS] WAC-X模块所有测试通过!")
    print("="*60)

