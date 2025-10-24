# -*- coding: utf-8 -*-
"""
Patch Prior Refiner 单元测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from models.world_model import PatchPriorRefiner


def test_patch_extraction():
    """测试Patch提取"""
    print("测试1: Patch提取")
    refiner = PatchPriorRefiner(patch_size=32, overlap=0.25)
    
    image = torch.randn(1, 3, 128, 128)
    patches, positions = refiner.extract_patches(image, 32, 24)
    
    print(f"  输入: {image.shape}")
    print(f"  Patch数: {len(patches)}")
    print(f"  Patch形状: {patches.shape}")
    assert patches.shape[1:] == (3, 32, 32), "Patch形状错误"
    print("  [PASS]\n")


def test_patch_merge():
    """测试Patch合并"""
    print("测试2: Patch合并")
    refiner = PatchPriorRefiner(patch_size=32, overlap=0.25)
    
    image = torch.randn(1, 3, 128, 128)
    patches, positions = refiner.extract_patches(image, 32, 24)
    merged = refiner.merge_patches(patches, positions, (128, 128), 32, 24)
    
    assert merged.shape == image.shape, "合并形状错误"
    recon_error = F.mse_loss(merged, image).item()
    print(f"  重建误差: {recon_error:.6f}")
    assert recon_error < 1e-4, f"重建误差过大: {recon_error}"
    print("  [PASS]\n")


def test_refine():
    """测试refine功能"""
    print("测试3: Refine功能")
    refiner = PatchPriorRefiner(patch_size=32)
    
    image = torch.randn(1, 3, 128, 128)
    refined = refiner.refine(image)
    
    assert refined.shape == image.shape, "输出形状错误"
    print(f"  输入: {image.shape}")
    print(f"  输出: {refined.shape}")
    print("  [PASS]\n")


def run_all_tests():
    """运行所有测试"""
    print("="*60)
    print("Patch Prior Refiner 单元测试套件")
    print("="*60)
    print()
    
    try:
        test_patch_extraction()
        test_patch_merge()
        test_refine()
        
        print("="*60)
        print("[SUCCESS] Patch Prior所有单元测试通过!")
        print("="*60)
        return True
        
    except AssertionError as e:
        print(f"\n[FAIL] 测试失败: {e}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

