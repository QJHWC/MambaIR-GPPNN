# -*- coding: utf-8 -*-
"""
WSM (World State Memory) 单元测试
验证世界状态记忆模块的功能正确性
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.world_model import WorldStateMemory


def test_wsm_shape():
    """测试WSM输出形状"""
    print("测试1: 输出形状验证")
    B, C, H, W = 2, 96, 64, 64
    feat = torch.randn(B, C, H, W)
    
    wsm = WorldStateMemory(feature_dim=C, hidden_dim=128)
    out, h, gamma, beta = wsm(feat, h_prev=None)
    
    assert out.shape == (B, C, H, W), f"输出形状错误: {out.shape}"
    assert h.shape == (B, 128), f"隐状态形状错误: {h.shape}"
    assert gamma.shape == (B, C), f"gamma形状错误: {gamma.shape}"
    assert beta.shape == (B, C), f"beta形状错误: {beta.shape}"
    
    print(f"  [PASS] 输出形状: {out.shape}")
    print(f"  [PASS] 隐状态: {h.shape}\n")


def test_wsm_state_update():
    """测试隐状态更新"""
    print("测试2: 隐状态更新验证")
    B, C, H, W = 2, 96, 64, 64
    feat = torch.randn(B, C, H, W)
    
    wsm = WorldStateMemory(feature_dim=C, hidden_dim=128)
    
    # 第一次前向传播
    out1, h1, _, _ = wsm(feat, h_prev=None)
    
    # 第二次前向传播（使用前一次的隐状态）
    out2, h2, _, _ = wsm(feat, h_prev=h1)
    
    # 隐状态应该更新
    assert not torch.equal(h1, h2), "隐状态未更新"
    
    # 输出应该不同（因为隐状态不同）
    assert not torch.equal(out1, out2), "输出未受隐状态影响"
    
    print(f"  [PASS] 隐状态成功更新")
    print(f"  [PASS] 输出受隐状态调制\n")


def test_wsm_modulation():
    """测试调制效果"""
    print("测试3: 特征调制验证")
    B, C, H, W = 2, 96, 64, 64
    feat = torch.randn(B, C, H, W)
    
    wsm = WorldStateMemory(feature_dim=C, hidden_dim=128)
    out, h, gamma, beta = wsm(feat, h_prev=None)
    
    # gamma和beta应该在合理范围内（Tanh输出范围[-1, 1]）
    assert gamma.abs().max() <= 1.0, "gamma超出范围"
    assert beta.abs().max() <= 1.0, "beta超出范围"
    
    # 输出应该与输入有差异（被调制）
    diff = (out - feat).abs().mean()
    assert diff > 1e-5, "调制效果不明显"
    
    print(f"  [PASS] gamma范围: [{gamma.min().item():.3f}, {gamma.max().item():.3f}]")
    print(f"  [PASS] beta范围: [{beta.min().item():.3f}, {beta.max().item():.3f}]")
    print(f"  [PASS] 平均调制幅度: {diff.item():.6f}\n")


def test_wsm_parameters():
    """测试参数量"""
    print("测试4: 参数量统计")
    C = 96
    hidden_dim = 128
    
    wsm = WorldStateMemory(feature_dim=C, hidden_dim=hidden_dim)
    
    total_params = sum(p.numel() for p in wsm.parameters())
    trainable_params = sum(p.numel() for p in wsm.parameters() if p.requires_grad)
    
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  显存估算: ~{total_params * 4 / 1024**2:.2f} MB")
    
    # 验证参数量在合理范围（不超过1M）
    assert total_params < 1_000_000, "参数量过大"
    
    print(f"  [PASS] 参数量合理\n")


def test_wsm_gradient():
    """测试梯度反向传播"""
    print("测试5: 梯度反向传播")
    B, C, H, W = 2, 96, 32, 32
    feat = torch.randn(B, C, H, W, requires_grad=True)
    
    wsm = WorldStateMemory(feature_dim=C, hidden_dim=128)
    
    out, h, gamma, beta = wsm(feat, h_prev=None)
    loss = out.sum()
    loss.backward()
    
    # 验证梯度存在
    assert feat.grad is not None, "输入特征梯度为None"
    assert feat.grad.abs().sum() > 0, "输入特征梯度为0"
    
    # 验证模型参数梯度
    grad_params = sum(p.grad.abs().sum().item() for p in wsm.parameters() if p.grad is not None)
    assert grad_params > 0, "模型参数梯度为0"
    
    print(f"  [PASS] 输入梯度范数: {feat.grad.norm().item():.6f}")
    print(f"  [PASS] 参数梯度总和: {grad_params:.6f}\n")


def run_all_tests():
    """运行所有测试"""
    print("="*60)
    print("WSM (World State Memory) 单元测试套件")
    print("="*60)
    print()
    
    try:
        test_wsm_shape()
        test_wsm_state_update()
        test_wsm_modulation()
        test_wsm_parameters()
        test_wsm_gradient()
        
        print("="*60)
        print("[SUCCESS] WSM所有单元测试通过!")
        print("="*60)
        return True
        
    except AssertionError as e:
        print(f"\n[FAIL] 测试失败: {e}")
        return False
    except Exception as e:
        print(f"\n[ERROR] 测试错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

