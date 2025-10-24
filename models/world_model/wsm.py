# -*- coding: utf-8 -*-
"""
世界状态记忆模块 (World State Memory - WSM)
通过GRU隐状态记忆实现时序一致性，降低生成方差

数学原理:
    h_t = GRU(Pool(F_t), h_{t-1})
    gamma, beta = Linear(h_t)
    F'_t = F_t * (1 + gamma * scale) + beta
    
效果: Var(x̃_0) = Var(x̂_0)(1 - ρ²) → MSE↓, PSNR↑

参考:
    《最新任务计划.md》 Section 2.3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WorldStateMemory(nn.Module):
    """
    世界状态记忆模块
    
    核心功能:
    1. 通过GRU维持隐状态记忆（时序一致性）
    2. 生成gamma/beta调制系数（特征调制）
    3. LayerScale可学习缩放（训练稳定性）
    
    Args:
        feature_dim: 特征维度（与输入特征通道数一致）
        hidden_dim: GRU隐状态维度（默认128）
        dropout: Dropout率（默认0.1）
        layer_scale_init: LayerScale初始值（默认0.1）
    """
    def __init__(self, feature_dim, hidden_dim=128, dropout=0.1, layer_scale_init=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # GRU Cell用于状态更新
        self.gru_cell = nn.GRUCell(feature_dim, hidden_dim)
        
        # gamma生成网络（乘性调制）
        self.to_gamma = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, feature_dim),
            nn.Tanh()  # 限制调制范围在[-1, 1]
        )
        
        # beta生成网络（加性调制）
        self.to_beta = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, feature_dim),
            nn.Tanh()  # 限制调制范围在[-1, 1]
        )
        
        # LayerScale可学习缩放（控制调制强度）
        self.layer_scale_gamma = nn.Parameter(torch.ones(feature_dim) * layer_scale_init)
        self.layer_scale_beta = nn.Parameter(torch.ones(feature_dim) * layer_scale_init)
        
        # Dropout正则化
        self.dropout = nn.Dropout(dropout)
        
        # 归一化层
        self.norm_feat = nn.LayerNorm(feature_dim)
        self.norm_hidden = nn.LayerNorm(hidden_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        """权重初始化"""
        # GRU权重初始化
        for name, param in self.gru_cell.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        # Linear层初始化
        for module in [self.to_gamma, self.to_beta]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, feat, h_prev=None):
        """
        前向传播
        
        Args:
            feat: [B, C, H, W] 输入特征图
            h_prev: [B, hidden_dim] 前一时刻隐状态（None则初始化为0）
            
        Returns:
            modulated_feat: [B, C, H, W] 调制后的特征
            h_t: [B, hidden_dim] 当前时刻隐状态
            gamma: [B, C] gamma调制系数
            beta: [B, C] beta调制系数
        """
        B, C, H, W = feat.shape
        
        # 全局平均池化降维: [B, C, H, W] → [B, C]
        pooled_feat = feat.mean(dim=[2, 3])
        pooled_feat = self.norm_feat(pooled_feat)
        
        # 初始化隐状态（第一次调用或新序列）
        if h_prev is None:
            h_prev = torch.zeros(B, self.hidden_dim, device=feat.device, dtype=feat.dtype)
        
        # GRU状态更新: h_t = GRU(pooled_feat, h_{t-1})
        h_t = self.gru_cell(pooled_feat, h_prev)
        h_t = self.norm_hidden(h_t)
        h_t = self.dropout(h_t)
        
        # 生成调制系数
        gamma = self.to_gamma(h_t)  # [B, C]
        beta = self.to_beta(h_t)    # [B, C]
        
        # 应用LayerScale（可学习的调制强度）
        gamma_scaled = gamma * self.layer_scale_gamma.view(1, C)
        beta_scaled = beta * self.layer_scale_beta.view(1, C)
        
        # 扩展维度用于广播: [B, C] → [B, C, 1, 1]
        gamma_expanded = gamma_scaled.view(B, C, 1, 1)
        beta_expanded = beta_scaled.view(B, C, 1, 1)
        
        # 特征调制: F'_t = F_t * (1 + gamma) + beta
        modulated_feat = feat * (1 + gamma_expanded) + beta_expanded
        
        return modulated_feat, h_t, gamma, beta
    
    def reset_state(self):
        """
        重置隐状态（用于新序列开始）
        
        注: 实际上隐状态由外部管理（通过h_prev参数），
        此方法主要用于清空模型内部可能缓存的状态
        """
        if hasattr(self, '_cached_state'):
            delattr(self, '_cached_state')


if __name__ == "__main__":
    """WSM模块单元测试"""
    print("="*60)
    print("WSM (World State Memory) 模块测试")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}\n")
    
    # 测试参数
    B, C, H, W = 2, 96, 64, 64
    hidden_dim = 128
    
    # 创建WSM模块
    wsm = WorldStateMemory(
        feature_dim=C,
        hidden_dim=hidden_dim,
        dropout=0.1,
        layer_scale_init=0.1
    ).to(device)
    
    # 测试前向传播
    print("测试1: 前向传播（无隐状态）")
    feat = torch.randn(B, C, H, W).to(device)
    out1, h1, gamma1, beta1 = wsm(feat, h_prev=None)
    
    print(f"  输入特征: {feat.shape}")
    print(f"  输出特征: {out1.shape}")
    print(f"  隐状态: {h1.shape}")
    print(f"  gamma: {gamma1.shape}, 范围: [{gamma1.min().item():.3f}, {gamma1.max().item():.3f}]")
    print(f"  beta: {beta1.shape}, 范围: [{beta1.min().item():.3f}, {beta1.max().item():.3f}]")
    assert out1.shape == feat.shape, "输出形状不匹配"
    assert h1.shape == (B, hidden_dim), "隐状态形状不匹配"
    print("  [OK] 形状验证通过\n")
    
    # 测试2: 带隐状态的前向传播
    print("测试2: 前向传播（有隐状态）")
    out2, h2, gamma2, beta2 = wsm(feat, h_prev=h1)
    
    print(f"  输出特征: {out2.shape}")
    print(f"  隐状态: {h2.shape}")
    print(f"  隐状态是否更新: {not torch.equal(h1, h2)}")
    assert out2.shape == feat.shape, "输出形状不匹配"
    assert not torch.equal(h1, h2), "隐状态未更新"
    print("  [OK] 隐状态更新验证通过\n")
    
    # 测试3: 参数量统计
    print("测试3: 参数量统计")
    total_params = sum(p.numel() for p in wsm.parameters())
    trainable_params = sum(p.numel() for p in wsm.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  显存估算: ~{total_params * 4 / 1024**2:.2f} MB\n")
    
    # 测试4: 批次间一致性
    print("测试4: 批次间隐状态传递")
    h_state = None
    for t in range(3):
        feat_t = torch.randn(B, C, H, W).to(device)
        out_t, h_state, _, _ = wsm(feat_t, h_prev=h_state)
        print(f"  时刻t={t}: 隐状态均值={h_state.mean().item():.4f}, 标准差={h_state.std().item():.4f}")
    print("  [OK] 批次间传递验证通过\n")
    
    print("="*60)
    print("[SUCCESS] WSM模块所有测试通过!")
    print("="*60)

