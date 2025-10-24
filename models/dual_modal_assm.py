# -*- coding: utf-8 -*-
"""
双模态注意力状态空间模块 (Dual Modal ASSM)
MambaIR-GPPNN 的核心组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def semantic_neighbor(x, index):
    """语义邻居对齐 - SGN预处理"""
    dim = index.dim()
    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)
    shuffled_x = torch.gather(x, dim=dim - 1, index=index)
    return shuffled_x


class DualModalSelectiveScan(nn.Module):
    """双模态选择性扫描 - 强化版Mamba特性 with 深度特征融合"""
    def __init__(self, d_model, d_state=16, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # 🔥 优化1: 多层特征映射，增强非线性表达
        self.ms_linear = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.pan_linear = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # 🔥 优化2: 分层融合策略 - 逐步深化特征交互
        self.fusion_linear1 = nn.Linear(d_model * 2, d_model * 2)
        self.fusion_linear2 = nn.Linear(d_model * 2, d_model)
        self.fusion_gelu = nn.GELU()

        # 状态归一化与门控
        self.norm_ms = nn.LayerNorm(d_model)
        self.norm_pan = nn.LayerNorm(d_model)
        self.merge_norm = nn.LayerNorm(d_model)

        # 🔥 优化3: 双路门控 - 分别控制MS和PAN的融合比例
        self.ms_gate = nn.Linear(d_model * 2, d_model)
        self.pan_gate = nn.Linear(d_model * 2, d_model)
        self.context_gate = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)

        # 🔥 优化4: 增强残差通路 - 使用LayerScale技术
        self.ms_residual = nn.Linear(d_model, d_model)
        self.pan_residual = nn.Linear(d_model, d_model)
        self.layer_scale_ms = nn.Parameter(torch.ones(d_model) * 0.1)
        self.layer_scale_pan = nn.Parameter(torch.ones(d_model) * 0.1)

    def forward(self, ms_seq, pan_seq, ms_prompt, pan_prompt):
        """双模态特征扫描并融合 - 优化版"""
        hidden_dim = ms_seq.shape[-1]
        prompt_dim = ms_prompt.shape[-1]

        # Prompt对齐（保持原逻辑）
        if hidden_dim > prompt_dim:
            repeat_factor = hidden_dim // prompt_dim
            ms_prompt_proj = ms_prompt.repeat(1, 1, repeat_factor)
            pan_prompt_proj = pan_prompt.repeat(1, 1, repeat_factor)

            if hidden_dim % prompt_dim != 0:
                padding_size = hidden_dim - ms_prompt_proj.shape[-1]
                ms_padding = ms_prompt[:, :, :padding_size]
                pan_padding = pan_prompt[:, :, :padding_size]
                ms_prompt_proj = torch.cat([ms_prompt_proj, ms_padding], dim=-1)
                pan_prompt_proj = torch.cat([pan_prompt_proj, pan_padding], dim=-1)
        else:
            ms_prompt_proj = ms_prompt[:, :, :hidden_dim]
            pan_prompt_proj = pan_prompt[:, :, :hidden_dim]

        # 🔥 优化5: 增强的特征映射 + Prompt注入
        ms_mapped = self.ms_linear(ms_seq)
        pan_mapped = self.pan_linear(pan_seq)
        ms_enhanced = self.norm_ms(ms_mapped + ms_prompt_proj)
        pan_enhanced = self.norm_pan(pan_mapped + pan_prompt_proj)

        # 🔥 优化6: 分层深度融合
        concat_feat = torch.cat([ms_enhanced, pan_enhanced], dim=-1)
        fusion_deep = self.fusion_gelu(self.fusion_linear1(concat_feat))
        fusion = self.fusion_linear2(fusion_deep)
        fusion = self.merge_norm(fusion)

        # 🔥 优化7: 双路自适应门控
        gate_input = torch.cat([ms_seq, pan_seq], dim=-1)
        ms_gate_weight = torch.sigmoid(self.ms_gate(gate_input))
        pan_gate_weight = torch.sigmoid(self.pan_gate(gate_input))
        context_gate_weight = torch.sigmoid(self.context_gate(gate_input))

        # 应用门控 + Dropout
        fusion_gated = self.dropout(fusion * context_gate_weight)

        # 🔥 优化8: 差异化融合 - MS和PAN使用不同的融合比例
        ms_out = ms_enhanced + fusion_gated * ms_gate_weight * 0.6
        pan_out = pan_enhanced + fusion_gated * pan_gate_weight * 0.6

        # 🔥 优化9: LayerScale残差 - 可学习的残差缩放
        ms_out = ms_out + self.ms_residual(ms_seq) * self.layer_scale_ms
        pan_out = pan_out + self.pan_residual(pan_seq) * self.layer_scale_pan

        return ms_out, pan_out


class DualModal_ASSM(nn.Module):
    """
    双模态注意力状态空间模块

    核心流程:
    1. 分别对 MS 和 PAN 进行投影
    2. 自适应局部上下文增强
    3. 语义路由 + prompt 检索
    4. 双模态选择性扫描
    5. 模态对齐输出
    """
    def __init__(self, dim, d_state=16, num_tokens=64, inner_rank=32, mlp_ratio=2.):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.num_tokens = num_tokens
        self.inner_rank = inner_rank

        self.expand = mlp_ratio
        hidden = int(self.dim * self.expand)

        # 双模态投影
        self.ms_proj = nn.Conv2d(self.dim, hidden, 1, 1, 0)
        self.pan_proj = nn.Conv2d(self.dim, hidden, 1, 1, 0)

        # 🔥 优化10: 多尺度局部增强 - 不同感受野捕获多层次细节
        self.local_enhance_3x3 = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden)
        self.local_enhance_5x5 = nn.Conv2d(hidden, hidden, 5, 1, 2, groups=hidden)
        self.local_fusion = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 1, 1, 0)
        )

        # 🔥 优化11: 频域增强 - 捕获全局频率信息
        self.freq_enhance = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, hidden // 4, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden // 4, hidden, 1, 1, 0),
            nn.Sigmoid()
        )

        self.modality_gate = nn.Conv2d(hidden * 2, hidden * 2, 1, 1, 0)
        self.CPE = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden)

        # 语义路由
        self.route = nn.Sequential(
            nn.Linear(self.dim, self.dim // 3),
            nn.GELU(),
            nn.Linear(self.dim // 3, self.num_tokens),
            nn.LogSoftmax(dim=-1)
        )

        # Prompt 嵌入
        self.ms_embedding = nn.Embedding(self.num_tokens, self.inner_rank)
        self.pan_embedding = nn.Embedding(self.num_tokens, self.inner_rank)
        self.token_proj = nn.Linear(self.inner_rank, self.d_state)

        # 双模态选择性扫描
        self.selective_scan = DualModalSelectiveScan(hidden, d_state)

        # 输出层
        self.out_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, dim)

        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        nn.init.uniform_(self.ms_embedding.weight, -1/self.num_tokens, 1/self.num_tokens)
        nn.init.uniform_(self.pan_embedding.weight, -1/self.num_tokens, 1/self.num_tokens)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, ms_feat, pan_feat, x_size):
        """
        Args:
            ms_feat: [B, C, H, W]
            pan_feat: [B, C, H, W]
            x_size: (H, W)
        Returns:
            ms_enhanced, pan_enhanced: 双模态增强特征
        """
        B, C, H, W = ms_feat.shape

        ms_hidden = self.ms_proj(ms_feat)
        pan_hidden = self.pan_proj(pan_feat)

        # 条件位置编码（保持原逻辑）
        ms_hidden = ms_hidden * torch.sigmoid(self.CPE(ms_hidden))
        pan_hidden = pan_hidden * torch.sigmoid(self.CPE(pan_hidden))

        # 🔥 优化12: 多尺度局部增强
        ms_local_3x3 = self.local_enhance_3x3(ms_hidden)
        ms_local_5x5 = self.local_enhance_5x5(ms_hidden)
        ms_local = self.local_fusion(torch.cat([ms_local_3x3, ms_local_5x5], dim=1))

        pan_local_3x3 = self.local_enhance_3x3(pan_hidden)
        pan_local_5x5 = self.local_enhance_5x5(pan_hidden)
        pan_local = self.local_fusion(torch.cat([pan_local_3x3, pan_local_5x5], dim=1))

        # 🔥 优化13: 频域全局增强
        ms_freq = self.freq_enhance(ms_hidden)
        pan_freq = self.freq_enhance(pan_hidden)
        ms_hidden = ms_hidden * (1 + ms_freq)
        pan_hidden = pan_hidden * (1 + pan_freq)

        # 🔥 优化14: 自适应模态门控
        gate_features = torch.sigmoid(self.modality_gate(torch.cat([ms_local, pan_local], dim=1)))
        gate_ms, gate_pan = torch.chunk(gate_features, 2, dim=1)
        ms_hidden = ms_hidden + ms_local * gate_ms * 0.7  # 增强局部增强的比例
        pan_hidden = pan_hidden + pan_local * gate_pan * 0.7

        ms_seq = ms_hidden.flatten(2).transpose(1, 2)
        pan_seq = pan_hidden.flatten(2).transpose(1, 2)

        ms_seq_orig = ms_feat.flatten(2).transpose(1, 2)
        pan_seq_orig = pan_feat.flatten(2).transpose(1, 2)
        fused_for_route = (ms_seq_orig + pan_seq_orig) / 2
        route_logits = self.route(fused_for_route)
        route_policy = F.gumbel_softmax(route_logits, hard=True, dim=-1)

        ms_prompt = torch.matmul(route_policy, self.ms_embedding.weight)
        pan_prompt = torch.matmul(route_policy, self.pan_embedding.weight)

        ms_prompt = self.token_proj(ms_prompt)
        pan_prompt = self.token_proj(pan_prompt)

        semantic_indices = torch.argmax(route_policy, dim=-1)
        sorted_indices = torch.argsort(semantic_indices, dim=-1)
        reverse_indices = torch.argsort(sorted_indices, dim=-1)

        ms_semantic = torch.gather(ms_seq, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, ms_seq.shape[-1]))
        pan_semantic = torch.gather(pan_seq, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, pan_seq.shape[-1]))
        ms_prompt_sorted = torch.gather(ms_prompt, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, ms_prompt.shape[-1]))
        pan_prompt_sorted = torch.gather(pan_prompt, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, pan_prompt.shape[-1]))

        ms_out, pan_out = self.selective_scan(ms_semantic, pan_semantic, ms_prompt_sorted, pan_prompt_sorted)

        ms_out = torch.gather(ms_out, 1, reverse_indices.unsqueeze(-1).expand(-1, -1, ms_out.shape[-1]))
        pan_out = torch.gather(pan_out, 1, reverse_indices.unsqueeze(-1).expand(-1, -1, pan_out.shape[-1]))

        ms_final = self.out_proj(self.out_norm(ms_out))
        pan_final = self.out_proj(self.out_norm(pan_out))

        ms_final = ms_final.transpose(1, 2).reshape(B, C, H, W)
        pan_final = pan_final.transpose(1, 2).reshape(B, C, H, W)

        return ms_final, pan_final


if __name__ == "__main__":
    # 测试DualModal_ASSM
    print("🚀 测试DualModal_ASSM模块...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size = 2
    channels = 64
    height, width = 64, 64

    ms_feat = torch.randn(batch_size, channels, height, width).to(device)
    pan_feat = torch.randn(batch_size, channels, height, width).to(device)

    assm = DualModal_ASSM(
        dim=channels,
        d_state=16,
        num_tokens=64,
        inner_rank=32
    ).to(device)

    with torch.no_grad():
        ms_out, pan_out = assm(ms_feat, pan_feat, x_size=(height, width))

    print(f"✅ 输入 - MS: {ms_feat.shape}, PAN: {pan_feat.shape}")
    print(f"✅ 输出 - MS: {ms_out.shape}, PAN: {pan_out.shape}")
    print(f"✅ 参数量: {sum(p.numel() for p in assm.parameters()):,}")

    print("🎉 DualModal_ASSM测试通过!")
