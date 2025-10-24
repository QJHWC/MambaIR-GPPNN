# -*- coding: utf-8 -*-
"""
DSC (Differentiable Sensor Consistency) 单元测试
验证物理一致性损失模块的功能正确性
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from models.world_model import SensorConsistencyLoss


def test_dsc_mtf_kernel():
    """测试MTF核生成"""
    print("测试1: MTF核生成与归一化")
    dsc = SensorConsistencyLoss(mtf_kernel_size=5, mtf_sigma=1.0)
    
    # 检查MTF核形状
    assert dsc.mtf_kernel.shape == (1, 1, 5, 5), f"MTF核形状错误: {dsc.mtf_kernel.shape}"
    
    # 检查归一化
    kernel_sum = dsc.mtf_kernel.sum().item()
    assert abs(kernel_sum - 1.0) < 1e-5, f"MTF核未归一化: sum={kernel_sum}"
    
    print(f"  MTF核形状: {dsc.mtf_kernel.shape}")
    print(f"  MTF核总和: {kernel_sum:.6f}")
    print("  [PASS] MTF核正确\n")


def test_dsc_pan_synthesis():
    """测试PAN合成功能"""
    print("测试2: PAN合成（光谱响应）")
    dsc = SensorConsistencyLoss(spectral_response=[0.299, 0.587, 0.114])
    
    B, C, H, W = 2, 3, 64, 64
    hrms = torch.randn(B, C, H, W)
    
    # 合成PAN（不应用MTF）
    pan_syn_no_mtf = dsc.sensor_forward(hrms, apply_mtf=False)
    assert pan_syn_no_mtf.shape == (B, 1, H, W), f"PAN形状错误: {pan_syn_no_mtf.shape}"
    
    # 合成PAN（应用MTF）
    pan_syn_mtf = dsc.sensor_forward(hrms, apply_mtf=True)
    assert pan_syn_mtf.shape == (B, 1, H, W), f"PAN形状错误: {pan_syn_mtf.shape}"
    
    # MTF应该使图像更模糊（方差降低）
    var_no_mtf = pan_syn_no_mtf.var().item()
    var_mtf = pan_syn_mtf.var().item()
    
    print(f"  无MTF方差: {var_no_mtf:.6f}")
    print(f"  有MTF方差: {var_mtf:.6f}")
    print(f"  方差降低: {(var_no_mtf - var_mtf) / var_no_mtf * 100:.2f}%")
    print("  [PASS] PAN合成功能正确\n")


def test_dsc_loss_computation():
    """测试DSC损失计算"""
    print("测试3: DSC损失计算")
    dsc = SensorConsistencyLoss(lrms_weight=0.3)
    
    B, C, H, W = 2, 3, 128, 128
    hrms = torch.randn(B, C, H, W)
    pan_gt = torch.randn(B, 1, H, W)
    lrms_gt = torch.randn(B, C, H//4, W//4)
    
    # 仅PAN损失
    loss_dict_pan = dsc(hrms, pan_gt, lrms_gt=None)
    print(f"  仅PAN - total: {loss_dict_pan['dsc_total'].item():.6f}")
    print(f"          pan: {loss_dict_pan['dsc_pan'].item():.6f}")
    print(f"          lrms: {loss_dict_pan['dsc_lrms'].item():.6f}")
    
    # PAN + LRMS损失
    loss_dict_full = dsc(hrms, pan_gt, lrms_gt)
    print(f"  PAN+LRMS - total: {loss_dict_full['dsc_total'].item():.6f}")
    print(f"             pan: {loss_dict_full['dsc_pan'].item():.6f}")
    print(f"             lrms: {loss_dict_full['dsc_lrms'].item():.6f}")
    
    # 验证LRMS权重
    expected_total = (loss_dict_full['dsc_pan'] + 
                      0.3 * loss_dict_full['dsc_lrms'])
    assert abs(loss_dict_full['dsc_total'].item() - expected_total.item()) < 1e-5, \
        "DSC总损失计算错误"
    
    print("  [PASS] DSC损失计算正确\n")


def test_dsc_gradient():
    """测试DSC梯度"""
    print("测试4: DSC梯度反向传播")
    dsc = SensorConsistencyLoss()
    
    B, C, H, W = 1, 3, 64, 64
    hrms = torch.randn(B, C, H, W, requires_grad=True)
    pan_gt = torch.randn(B, 1, H, W)
    lrms_gt = torch.randn(B, C, H//4, W//4)
    
    loss_dict = dsc(hrms, pan_gt, lrms_gt)
    loss_dict['dsc_total'].backward()
    
    assert hrms.grad is not None, "HRMS梯度为None"
    assert hrms.grad.abs().sum() > 0, "HRMS梯度为0"
    
    print(f"  HRMS梯度范数: {hrms.grad.norm().item():.6f}")
    print(f"  梯度非零元素: {(hrms.grad.abs() > 1e-6).sum().item()}")
    print("  [PASS] 梯度传播正确\n")


def test_dsc_spectral_response():
    """测试光谱响应适配"""
    print("测试5: 光谱响应自适应")
    
    # 测试不同通道数
    for C in [3, 4, 8]:
        dsc = SensorConsistencyLoss(spectral_response=[0.3, 0.6, 0.1])  # 仅3通道
        hrms = torch.randn(1, C, 32, 32)
        pan_syn = dsc.sensor_forward(hrms, apply_mtf=False)
        
        assert pan_syn.shape == (1, 1, 32, 32), f"通道数{C}时PAN形状错误"
        print(f"  通道数={C}: PAN合成成功 {pan_syn.shape}")
    
    print("  [PASS] 光谱响应自适应正确\n")


def run_all_tests():
    """运行所有测试"""
    print("="*60)
    print("DSC (Sensor Consistency Loss) 单元测试套件")
    print("="*60)
    print()
    
    try:
        test_dsc_mtf_kernel()
        test_dsc_pan_synthesis()
        test_dsc_loss_computation()
        test_dsc_gradient()
        test_dsc_spectral_response()
        
        print("="*60)
        print("[SUCCESS] DSC所有单元测试通过!")
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

