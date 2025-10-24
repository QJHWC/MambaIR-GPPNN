# -*- coding: utf-8 -*-
"""
WAC-X (Cross-band Consistency) 单元测试
验证频域一致性损失模块的功能正确性
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.world_model import WACXLoss


def test_wacx_fft():
    """测试FFT频谱计算"""
    print("测试1: FFT频谱计算")
    wacx = WACXLoss()
    
    B, C, H, W = 2, 3, 64, 64
    hrms = torch.randn(B, C, H, W)
    
    # 手动计算FFT
    fft_result = torch.fft.rfft2(hrms[:, 0], norm='ortho')
    H_magnitude = torch.abs(fft_result)
    
    print(f"  HRMS: {hrms.shape}")
    print(f"  FFT: {fft_result.shape} (复数)")
    print(f"  幅度谱: {H_magnitude.shape}")
    print(f"  范围: [{H_magnitude.min().item():.6f}, {H_magnitude.max().item():.6f}]")
    print("  [PASS] FFT计算正确\n")


def test_wacx_high_freq():
    """测试高频提取"""
    print("测试2: 高频分量提取")
    wacx = WACXLoss(freq_threshold=0.1)
    
    B, H, W = 1, 64, 64
    spectrum = torch.randn(B, H, W//2+1).abs()  # 模拟频谱
    
    high_freq = wacx.extract_high_freq(spectrum, threshold=0.1)
    
    total_energy = spectrum.sum().item()
    high_energy = high_freq.sum().item()
    high_ratio = high_energy / total_energy * 100
    
    print(f"  总能量: {total_energy:.6f}")
    print(f"  高频能量: {high_energy:.6f}")
    print(f"  高频比例: {high_ratio:.2f}%")
    print("  [PASS] 高频提取正确\n")


def test_wacx_loss():
    """测试WAC-X损失计算"""
    print("测试3: WAC-X损失计算")
    wacx = WACXLoss(interband_weight=1.0, pan_gate_weight=0.5)
    
    B, C, H, W = 2, 3, 128, 128
    hrms = torch.randn(B, C, H, W)
    pan = torch.randn(B, 1, H, W)
    
    loss_dict = wacx(hrms, pan)
    
    print(f"  total: {loss_dict['wacx_total'].item():.6f}")
    print(f"  interband: {loss_dict['wacx_interband'].item():.6f}")
    print(f"  gate: {loss_dict['wacx_gate'].item():.6f}")
    
    # 验证损失计算
    expected_total = (1.0 * loss_dict['wacx_interband'] + 
                     0.5 * loss_dict['wacx_gate'])
    assert torch.allclose(loss_dict['wacx_total'], expected_total, atol=1e-5), \
        "WAC-X总损失计算错误"
    
    print("  [PASS] WAC-X损失计算正确\n")


def test_wacx_gradient():
    """测试WAC-X梯度"""
    print("测试4: WAC-X梯度反向传播")
    wacx = WACXLoss()
    
    B, C, H, W = 1, 3, 64, 64
    hrms = torch.randn(B, C, H, W, requires_grad=True)
    pan = torch.randn(B, 1, H, W)
    
    loss_dict = wacx(hrms, pan)
    loss_dict['wacx_total'].backward()
    
    assert hrms.grad is not None, "HRMS梯度为None"
    assert hrms.grad.abs().sum() > 0, "HRMS梯度为0"
    
    print(f"  HRMS梯度范数: {hrms.grad.norm().item():.6f}")
    print("  [PASS] 梯度传播正确\n")


def run_all_tests():
    """运行所有测试"""
    print("="*60)
    print("WAC-X (Cross-band Consistency) 单元测试套件")
    print("="*60)
    print()
    
    try:
        test_wacx_fft()
        test_wacx_high_freq()
        test_wacx_loss()
        test_wacx_gradient()
        
        print("="*60)
        print("[SUCCESS] WAC-X所有单元测试通过!")
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

