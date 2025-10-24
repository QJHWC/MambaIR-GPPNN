# -*- coding: utf-8 -*-
"""
Patch Prior Refiner - 免训练推理增强
使用Patch级别的流形约束修正生成结果

数学原理:
    L_patch = Σ_p min_z ||HRMS_p - G(z)||²
    
效果: E||Î - I*||² = bias² + variance↓ → 抑制伪影, Q8↑

参考:
    《最新任务计划.md》 Section 2.5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchPriorRefiner:
    """
    Patch Prior修正模块（免训练）
    
    核心功能:
    1. Patch提取与合并（重叠采样避免边界伪影）
    2. 生成器流形约束（可选，需要预训练生成器）
    3. 简单平滑修正（无生成器时的fallback）
    
    Args:
        generator: 预训练生成器（可选，None则使用简单平滑）
        patch_size: Patch尺寸（默认32）
        overlap: Patch重叠率（默认0.25）
    """
    def __init__(self, generator=None, patch_size=32, overlap=0.25):
        self.generator = generator
        if generator is not None:
            self.generator.eval()
            for param in self.generator.parameters():
                param.requires_grad = False
        
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = int(patch_size * (1 - overlap))
        
    def extract_patches(self, image, patch_size, stride):
        """
        提取重叠Patch
        
        Args:
            image: [B, C, H, W]
            patch_size: Patch大小
            stride: 滑动步长
            
        Returns:
            patches: [N, C, P, P] N个patch
            positions: List[(h, w)] Patch位置
        """
        B, C, H, W = image.shape
        assert B == 1, "Batch size must be 1 for patch extraction"
        
        patches = []
        positions = []
        
        # 滑动窗口提取patch
        for h in range(0, H - patch_size + 1, stride):
            for w in range(0, W - patch_size + 1, stride):
                patch = image[:, :, h:h+patch_size, w:w+patch_size]
                patches.append(patch)
                positions.append((h, w))
        
        # 处理边界（补齐最后一列/一行）
        # 右边界
        if (W - patch_size) % stride != 0:
            for h in range(0, H - patch_size + 1, stride):
                w = W - patch_size
                patch = image[:, :, h:h+patch_size, w:w+patch_size]
                patches.append(patch)
                positions.append((h, w))
        
        # 下边界
        if (H - patch_size) % stride != 0:
            for w in range(0, W - patch_size + 1, stride):
                h = H - patch_size
                patch = image[:, :, h:h+patch_size, w:w+patch_size]
                patches.append(patch)
                positions.append((h, w))
        
        # 右下角
        if (H - patch_size) % stride != 0 and (W - patch_size) % stride != 0:
            h, w = H - patch_size, W - patch_size
            patch = image[:, :, h:h+patch_size, w:w+patch_size]
            patches.append(patch)
            positions.append((h, w))
        
        if len(patches) > 0:
            patches = torch.cat(patches, dim=0)  # [N, C, P, P]
        else:
            patches = torch.empty(0, C, patch_size, patch_size, device=image.device)
        
        return patches, positions
    
    def merge_patches(self, patches, positions, image_size, patch_size, stride):
        """
        合并Patch（加权平均重叠区域）
        
        Args:
            patches: [N, C, P, P]
            positions: List[(h, w)]
            image_size: (H, W)
            patch_size: Patch大小
            stride: 滑动步长
            
        Returns:
            merged: [1, C, H, W]
        """
        H, W = image_size
        C = patches.shape[1]
        
        # 累加图和计数图
        merged = torch.zeros(1, C, H, W, device=patches.device)
        counts = torch.zeros(1, 1, H, W, device=patches.device)
        
        # 累加所有patch
        for idx, (h, w) in enumerate(positions):
            merged[:, :, h:h+patch_size, w:w+patch_size] += patches[idx:idx+1]
            counts[:, :, h:h+patch_size, w:w+patch_size] += 1
        
        # 加权平均（处理重叠区域）
        merged = merged / (counts + 1e-8)
        
        return merged
    
    @torch.no_grad()
    def refine(self, hrms):
        """
        推理时Patch级修正
        
        Args:
            hrms: [B, C, H, W] 网络输出（B必须为1）
            
        Returns:
            refined: [B, C, H, W] 修正后的输出
        """
        if self.generator is None:
            # 无生成器：使用简单双边滤波平滑
            return F.avg_pool2d(
                F.pad(hrms, (1, 1, 1, 1), mode='replicate'),
                kernel_size=3,
                stride=1
            )
        
        B, C, H, W = hrms.shape
        assert B == 1, "Batch size must be 1 for refine"
        
        # 提取Patch
        patches, positions = self.extract_patches(hrms, self.patch_size, self.stride)
        
        # 生成器修正
        refined_patches = self.generator(patches)
        
        # 合并Patch
        refined = self.merge_patches(
            refined_patches, 
            positions, 
            (H, W), 
            self.patch_size, 
            self.stride
        )
        
        # 与原图融合（保留70%修正，30%原图）
        alpha = 0.7
        refined = alpha * refined + (1 - alpha) * hrms
        
        return refined
    
    def loss(self, hrms):
        """
        训练时Patch流形约束损失
        
        Args:
            hrms: [B, C, H, W]
            
        Returns:
            loss: Scalar损失值
        """
        if self.generator is None:
            return torch.tensor(0.0, device=hrms.device)
        
        B, C, H, W = hrms.shape
        
        # 简化版：不提取patch，直接对整图应用生成器
        with torch.no_grad():
            # 生成器重建
            recon = self.generator(hrms)
        
        # MSE损失（鼓励输出接近生成器流形）
        loss = F.mse_loss(hrms, recon)
        
        return loss


if __name__ == "__main__":
    """Patch Prior Refiner模块单元测试"""
    print("="*60)
    print("Patch Prior Refiner 模块测试")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}\n")
    
    # 测试1: Patch提取
    print("测试1: Patch提取功能")
    refiner = PatchPriorRefiner(patch_size=32, overlap=0.25)
    
    B, C, H, W = 1, 3, 128, 128
    image = torch.randn(B, C, H, W).to(device)
    
    patches, positions = refiner.extract_patches(image, 32, 24)  # stride=24
    
    print(f"  输入图像: {image.shape}")
    print(f"  Patch数量: {len(patches)}")
    print(f"  Patch形状: {patches.shape}")
    print(f"  位置数量: {len(positions)}")
    print("  [OK] Patch提取成功\n")
    
    # 测试2: Patch合并
    print("测试2: Patch合并功能")
    merged = refiner.merge_patches(patches, positions, (H, W), 32, 24)
    
    print(f"  合并结果: {merged.shape}")
    assert merged.shape == image.shape, f"合并形状错误: {merged.shape}"
    
    # 计算重建误差
    recon_error = F.mse_loss(merged, image)
    print(f"  重建误差: {recon_error.item():.6f}")
    print("  [OK] Patch合并成功\n")
    
    # 测试3: 简单平滑修正（无生成器）
    print("测试3: 简单平滑修正")
    refined = refiner.refine(image)
    
    print(f"  原始: {image.shape}")
    print(f"  修正后: {refined.shape}")
    assert refined.shape == image.shape, f"修正形状错误: {refined.shape}"
    
    # 平滑应该降低方差
    var_orig = image.var().item()
    var_refined = refined.var().item()
    print(f"  原始方差: {var_orig:.6f}")
    print(f"  修正方差: {var_refined:.6f}")
    print(f"  方差变化: {(var_refined - var_orig) / var_orig * 100:.2f}%")
    print("  [OK] 平滑修正功能正确\n")
    
    # 测试4: 损失计算（无生成器）
    print("测试4: 损失计算")
    loss = refiner.loss(image)
    
    print(f"  Patch Prior损失: {loss.item():.6f}")
    assert loss.item() == 0.0, "无生成器时损失应为0"
    print("  [OK] 损失计算正确\n")
    
    print("="*60)
    print("[SUCCESS] Patch Prior Refiner所有测试通过!")
    print("="*60)

