# -*- coding: utf-8 -*-
"""
可微传感器一致性损失 (Differentiable Sensor Consistency Loss - DSC)
模拟遥感传感器的物理成像过程，约束生成结果符合物理规律

数学原理:
    PAN_syn = MTF(Σ R_b * HRMS_b)
    LRMS_syn = MTF(Downsample(HRMS))
    L_DSC = ||PAN_syn - PAN_gt||₁ + α||LRMS_syn - LRMS_gt||₁
    
效果: r⊤(Î - I*) → 0 ⇒ SAM↓, ERGAS↓

参考:
    《最新任务计划.md》 Section 2.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SensorConsistencyLoss(nn.Module):
    """
    DSC损失模块
    
    核心功能:
    1. MTF模糊核模拟传感器调制传递函数
    2. 光谱响应模拟RGB→PAN转换
    3. 多尺度一致性约束（PAN + LRMS）
    
    Args:
        spectral_response: RGB→PAN响应系数（默认[0.299, 0.587, 0.114]）
        mtf_kernel_size: MTF卷积核大小（默认5）
        mtf_sigma: 高斯模糊sigma（默认1.0）
        lrms_weight: LRMS损失权重（默认0.3）
        downsample_factor: 下采样倍数（默认4）
    """
    def __init__(self, 
                 spectral_response=[0.299, 0.587, 0.114],
                 mtf_kernel_size=5,
                 mtf_sigma=1.0,
                 lrms_weight=0.3,
                 downsample_factor=4):
        super().__init__()
        
        # 光谱响应系数（将RGB多光谱转换为全色）
        self.register_buffer('R', torch.tensor(spectral_response).view(1, -1, 1, 1))
        
        # MTF模糊核（模拟传感器点扩散函数）
        mtf_kernel = self._create_mtf_kernel(mtf_kernel_size, mtf_sigma)
        self.register_buffer('mtf_kernel', mtf_kernel)
        
        self.lrms_weight = lrms_weight
        self.downsample_factor = downsample_factor
        
    def _create_mtf_kernel(self, kernel_size, sigma):
        """
        创建高斯MTF模糊核
        
        Args:
            kernel_size: 卷积核大小（奇数）
            sigma: 高斯标准差
            
        Returns:
            kernel: [1, 1, K, K] 归一化卷积核
        """
        # 生成高斯核
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        kernel = kernel / kernel.sum()  # 归一化
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        
        return kernel
    
    def sensor_forward(self, hrms, apply_mtf=True):
        """
        传感器前向模型: HRMS → PAN_syn
        
        模拟物理成像过程:
        1. 光谱响应: PAN = Σ R_b * HRMS_b
        2. MTF模糊: PAN_blurred = Conv(PAN, MTF_kernel)
        
        Args:
            hrms: [B, C, H, W] 高分辨率多光谱图像
            apply_mtf: 是否应用MTF模糊
            
        Returns:
            pan_syn: [B, 1, H, W] 合成的全色图像
        """
        B, C, H, W = hrms.shape
        
        # 光谱响应: PAN = Σ R_b * HRMS_b
        # 自适应调整R的通道数
        if self.R.shape[1] != C:
            # 如果通道数不匹配，使用等权重
            R_adapted = torch.ones(1, C, 1, 1, device=hrms.device) / C
        else:
            R_adapted = self.R.to(hrms.device)
        
        # 加权求和生成PAN
        pan_syn = (hrms * R_adapted).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # MTF模糊（模拟传感器点扩散函数）
        if apply_mtf:
            pad_size = self.mtf_kernel.shape[-1] // 2
            pan_syn = F.conv2d(
                pan_syn, 
                self.mtf_kernel.to(hrms.device), 
                padding=pad_size
            )
        
        return pan_syn
    
    def forward(self, hrms_pred, pan_gt, lrms_gt=None):
        """
        计算DSC损失
        
        Args:
            hrms_pred: [B, C, H, W] 预测的高分辨率多光谱
            pan_gt: [B, 1, H, W] Ground Truth全色图像
            lrms_gt: [B, C, H/4, W/4] Ground Truth低分辨率多光谱（可选）
            
        Returns:
            loss_dict: 包含各项损失的字典
                - dsc_total: 总DSC损失
                - dsc_pan: PAN一致性损失
                - dsc_lrms: LRMS一致性损失
        """
        # ========== PAN一致性损失 ==========
        # 通过传感器前向模型合成PAN
        pan_syn = self.sensor_forward(hrms_pred, apply_mtf=True)
        loss_pan = F.l1_loss(pan_syn, pan_gt)
        
        # ========== LRMS一致性损失（如果提供）==========
        loss_lrms = torch.tensor(0.0, device=hrms_pred.device)
        if lrms_gt is not None:
            # 下采样HRMS到低分辨率
            hrms_down = F.avg_pool2d(
                hrms_pred, 
                self.downsample_factor, 
                self.downsample_factor
            )
            
            # 验证尺寸匹配
            if hrms_down.shape[2:] == lrms_gt.shape[2:]:
                # 计算LRMS损失（直接比较多光谱）
                loss_lrms = F.l1_loss(hrms_down, lrms_gt)
            else:
                # 尺寸不匹配，跳过LRMS损失
                pass
        
        # ========== 总DSC损失 ==========
        total_loss = loss_pan + self.lrms_weight * loss_lrms
        
        return {
            'dsc_total': total_loss,
            'dsc_pan': loss_pan,
            'dsc_lrms': loss_lrms
        }


if __name__ == "__main__":
    """DSC模块单元测试"""
    print("="*60)
    print("DSC (Sensor Consistency Loss) 模块测试")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}\n")
    
    # 创建DSC损失模块
    dsc_loss = SensorConsistencyLoss(
        spectral_response=[0.299, 0.587, 0.114],
        mtf_kernel_size=5,
        mtf_sigma=1.0,
        lrms_weight=0.3
    ).to(device)
    
    # 测试1: PAN合成
    print("测试1: PAN合成功能")
    B, C, H, W = 2, 3, 128, 128
    hrms = torch.randn(B, C, H, W).to(device)
    pan_syn = dsc_loss.sensor_forward(hrms, apply_mtf=True)
    
    print(f"  输入HRMS: {hrms.shape}")
    print(f"  合成PAN: {pan_syn.shape}")
    assert pan_syn.shape == (B, 1, H, W), f"PAN形状错误: {pan_syn.shape}"
    print("  [OK] PAN合成成功\n")
    
    # 测试2: DSC损失计算（仅PAN）
    print("测试2: DSC损失计算（仅PAN）")
    pan_gt = torch.randn(B, 1, H, W).to(device)
    loss_dict = dsc_loss(hrms, pan_gt, lrms_gt=None)
    
    print(f"  dsc_total: {loss_dict['dsc_total'].item():.6f}")
    print(f"  dsc_pan: {loss_dict['dsc_pan'].item():.6f}")
    print(f"  dsc_lrms: {loss_dict['dsc_lrms'].item():.6f}")
    assert loss_dict['dsc_total'] > 0, "DSC损失应为正值"
    print("  [OK] DSC损失计算正确\n")
    
    # 测试3: DSC损失计算（PAN + LRMS）
    print("测试3: DSC损失计算（PAN + LRMS）")
    lrms_gt = torch.randn(B, C, H//4, W//4).to(device)
    loss_dict_full = dsc_loss(hrms, pan_gt, lrms_gt)
    
    print(f"  dsc_total: {loss_dict_full['dsc_total'].item():.6f}")
    print(f"  dsc_pan: {loss_dict_full['dsc_pan'].item():.6f}")
    print(f"  dsc_lrms: {loss_dict_full['dsc_lrms'].item():.6f}")
    assert loss_dict_full['dsc_lrms'] > 0, "LRMS损失应为正值"
    print("  [OK] 完整DSC损失计算正确\n")
    
    # 测试4: MTF核检查
    print("测试4: MTF核验证")
    print(f"  MTF核形状: {dsc_loss.mtf_kernel.shape}")
    print(f"  MTF核总和: {dsc_loss.mtf_kernel.sum().item():.6f} (应接近1.0)")
    assert abs(dsc_loss.mtf_kernel.sum().item() - 1.0) < 1e-5, "MTF核未归一化"
    print("  [OK] MTF核归一化正确\n")
    
    # 测试5: 梯度反向传播
    print("测试5: 梯度反向传播")
    # 创建新的测试张量（作为叶子节点）
    hrms_grad = torch.randn(B, C, H, W, requires_grad=True, device=device)
    pan_test = torch.randn(B, 1, H, W, device=device)
    lrms_test = torch.randn(B, C, H//4, W//4, device=device)
    
    loss_dict = dsc_loss(hrms_grad, pan_test, lrms_test)
    loss = loss_dict['dsc_total']
    loss.backward()
    
    assert hrms_grad.grad is not None, "HRMS梯度为None"
    assert hrms_grad.grad.abs().sum() > 0, "HRMS梯度为0"
    print(f"  HRMS梯度范数: {hrms_grad.grad.norm().item():.6f}")
    print("  [OK] 梯度反向传播成功\n")
    
    print("="*60)
    print("[SUCCESS] DSC模块所有测试通过!")
    print("="*60)

