# -*- coding: utf-8 -*-
"""
DCA-FIM (Deformable Cross-Attention) 单元测试
验证几何对齐模块的功能正确性
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.world_model import DeformableCrossAttention


def test_dca_forward():
    """测试DCA-FIM前向传播"""
    print("测试1: DCA-FIM前向传播")
    B, C, H, W = 2, 96, 64, 64
    
    dca = DeformableCrossAttention(dim=C, num_points=4)
    query = torch.randn(B, C, H, W)
    key = torch.randn(B, C, H, W)
    
    output = dca(query, key)
    
    assert output.shape == query.shape, f"输出形状错误: {output.shape}"
    print(f"  输入: Query{query.shape}, Key{key.shape}")
    print(f"  输出: {output.shape}")
    print("  [PASS] 前向传播正确\n")


def test_dca_grid_sample():
    """测试grid_sample形变采样"""
    print("测试2: grid_sample形变采样")
    B, C, H, W = 1, 96, 32, 32
    
    dca = DeformableCrossAttention(dim=C, num_points=4)
    
    features = torch.randn(B, C, H, W)
    offsets = torch.randn(B, 8, H, W) * 0.1  # 小偏移
    weights = torch.randn(B, 4, H, W)
    
    with torch.no_grad():
        sampled = dca.deformable_sample(features, offsets, weights)
    
    assert sampled.shape == features.shape, "采样形状错误"
    print(f"  原始特征: {features.shape}")
    print(f"  采样结果: {sampled.shape}")
    print(f"  采样差异: {(sampled - features).abs().mean().item():.6f}")
    print("  [PASS] grid_sample功能正确\n")


def test_dca_offset_init():
    """测试offset初始化为0"""
    print("测试3: Offset初始化为0（避免初期形变过大）")
    dca = DeformableCrossAttention(dim=96, num_points=4)
    
    query = torch.randn(1, 96, 32, 32)
    
    with torch.no_grad():
        offsets = dca.offset_net(query)
    
    # Offset应该接近0（初始化策略）
    offset_mean = offsets.abs().mean().item()
    print(f"  Offset绝对值均值: {offset_mean:.6f}")
    assert offset_mean < 0.01, f"Offset初始值过大: {offset_mean}"
    print("  [PASS] Offset初始化正确\n")


def test_dca_parameters():
    """测试DCA-FIM参数量"""
    print("测试4: 参数量统计")
    dca = DeformableCrossAttention(dim=96, num_points=4)
    
    total_params = sum(p.numel() for p in dca.parameters())
    print(f"  总参数量: {total_params:,}")
    print(f"  显存估算: ~{total_params * 4 / 1024**2:.2f} MB")
    
    # 验证参数量合理（不超过200K）
    assert total_params < 200_000, f"参数量过大: {total_params}"
    print("  [PASS] 参数量合理\n")


def run_all_tests():
    """运行所有测试"""
    print("="*60)
    print("DCA-FIM (Deformable Cross-Attention) 单元测试套件")
    print("="*60)
    print()
    
    try:
        test_dca_forward()
        test_dca_grid_sample()
        test_dca_offset_init()
        test_dca_parameters()
        
        print("="*60)
        print("[SUCCESS] DCA-FIM所有单元测试通过!")
        print("="*60)
        return True
        
    except AssertionError as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n[ERROR] 测试错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

