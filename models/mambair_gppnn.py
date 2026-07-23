# -*- coding: utf-8 -*-
"""
MambaIR-inspired GPPNN 多模态全色锐化网络。

公开类名和工厂函数名沿用早期实验版本以保持兼容。当前实现借鉴
MambaIRv2 的语义路由思路，但不包含 Mamba 状态空间层或 Selective
Scan 算子，也不是 MambaIR 或 GPPNN 作者发布的官方实现。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dual_modal_assm import DualModal_ASSM
from .cross_modal_attention import CrossModalAttention


class MambaIRv2_GPPNN(nn.Module):
    """
    MambaIR-inspired 实验架构，保持既有的 GPPNN 风格多尺度输出接口。
    """
    def __init__(self,
                 ms_channels=3,
                 pan_channels=1,
                 embed_dim=96,
                 d_state=16,
                 depths=[6, 6, 6, 6],
                 num_heads=[6, 6, 6, 6],
                 window_size=8,
                 inner_rank=32,
                 num_tokens=128,
                 mlp_ratio=2.0,
                 model_size='base',
                 **kwargs):  # 🌍 接受世界模型配置参数
        super().__init__()

        if model_size == 'large':
            embed_dim = 128
            d_state = 20
            depths = [8, 8, 8, 8]
            num_heads = [8, 8, 8, 8]
            inner_rank = 48
            num_tokens = 256

        self.embed_dim = embed_dim
        self.model_size = model_size

        print(f"[Init] MambaIR-inspired GPPNN ({model_size.upper()})")
        print(f"  embed_dim={embed_dim}, d_state={d_state}")
        print(f"  depths={depths}, num_heads={num_heads}")

        # 1. 浅层特征提取
        self.ms_conv_first = nn.Conv2d(ms_channels, embed_dim, 3, 1, 1)
        self.pan_conv_first = nn.Conv2d(pan_channels, embed_dim, 3, 1, 1)

        # 2. MambaIR-inspired 双模态特征阶段
        self.mamba_layers = nn.ModuleList()
        current_dim = embed_dim

        for i, (depth, num_head) in enumerate(zip(depths, num_heads)):
            layer_d_state = d_state + i * 2
            layer_tokens = num_tokens + i * 64
            layer_rank = inner_rank + i * 8

            layer = DualModal_ASSM(
                dim=current_dim,
                d_state=layer_d_state,
                num_tokens=layer_tokens,
                inner_rank=layer_rank,
                mlp_ratio=mlp_ratio
            )
            self.mamba_layers.append(layer)

        # 3. 跨模态注意力 - 🔥 优化24: 逐层增加注意力头数
        # 🌍 传递世界模型配置（DCA-FIM）
        cross_attn_kwargs = {
            'use_dca_fim': kwargs.get('use_dca_fim', False),
            'dca_num_points': kwargs.get('dca_num_points', 4),
            'dca_offset_groups': kwargs.get('dca_offset_groups', 1),
            'dca_deform_weight': kwargs.get('dca_deform_weight', 0.3),
        }

        self.cross_modal_layers = nn.ModuleList([
            CrossModalAttention(embed_dim, num_heads=6, attn_drop=0.1, proj_drop=0.1, **cross_attn_kwargs),
            CrossModalAttention(embed_dim, num_heads=8, attn_drop=0.1, proj_drop=0.1, **cross_attn_kwargs),
            CrossModalAttention(embed_dim, num_heads=8, attn_drop=0.1, proj_drop=0.1, **cross_attn_kwargs)
        ])
        # 🔥 优化25: 增强低层细节保留
        self.low_level_refine = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 1, 1, 0)
        )
        # 🔥 优化26: 边缘保护模块
        self.edge_preserve = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 4, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, embed_dim, 1, 1, 0),
            nn.Sigmoid()
        )

        # 4. 多尺度处理
        self.downsample = nn.AvgPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # 5. 特征融合 - 🔥 优化27: 使用GELU + 残差块
        self.fusion_conv1 = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(embed_dim * 2, embed_dim * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(embed_dim * 2, embed_dim * 2, 1, 1, 0)
        )
        self.fusion_conv2 = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim * 4, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(embed_dim * 4, embed_dim * 4, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(embed_dim * 4, embed_dim * 4, 1, 1, 0)
        )
        # 🔥 优化28: 双路全局上下文（频道+空间）
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim * 4, embed_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim * 4, 1, 1, 0),
            nn.Sigmoid()
        )
        # 🔥 优化29: 空间注意力
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(embed_dim * 4, 1, 7, 1, 3),
            nn.Sigmoid()
        )

        # 6. 解码
        self.deconv1 = nn.ConvTranspose2d(embed_dim * 4, embed_dim * 2, 2, 2, 0)
        self.deconv2 = nn.ConvTranspose2d(embed_dim * 3, embed_dim, 2, 2, 0)

        # 7. 残差强化
        self.residual_enhance = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim * 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim * 2, embed_dim * 2, 1, 1, 0)
        )

        # 8. 多尺度输出
        self.output_conv1 = nn.Sequential(
            nn.Conv2d(embed_dim * 4, ms_channels, 3, 1, 1),
            nn.Sigmoid()
        )
        self.output_conv2 = nn.Sequential(
            nn.Conv2d(embed_dim * 2, ms_channels, 3, 1, 1),
            nn.Sigmoid()
        )
        self.output_conv3 = nn.Sequential(
            nn.Conv2d(embed_dim, ms_channels, 3, 1, 1),
            nn.Sigmoid()
        )

        # 🌍 世界模型增强: WSM模块
        self.use_wsm = kwargs.get('use_wsm', False)
        if self.use_wsm:
            from .world_model import WorldStateMemory
            self.wsm = WorldStateMemory(
                feature_dim=embed_dim,
                hidden_dim=kwargs.get('wsm_hidden_dim', 128),
                dropout=kwargs.get('wsm_dropout', 0.1),
                layer_scale_init=kwargs.get('wsm_layer_scale_init', 0.1)
            )
            print(f"[Init] [WorldModel] WSM (World State Memory) enabled, hidden_dim={kwargs.get('wsm_hidden_dim', 128)}")
        else:
            self.wsm = None

        self._init_weights()

    def _init_weights(self):
        """初始化卷积、线性层和归一化层权重。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, ms, pan):
        """
        前向传播，输出保持既有训练代码使用的三个尺度。
        """
        if pan is None:
            raise Exception('PAN image is required for pansharpening!')

        B, _, H, W = ms.shape

        # 浅层特征
        ms_feat1 = F.relu(self.ms_conv_first(ms))
        pan_feat1 = F.relu(self.pan_conv_first(pan))

        # Stage 1: 全分辨率 - 🔥 优化30: 增强细节保护
        ms_enhanced1, pan_enhanced1 = self.mamba_layers[0](ms_feat1, pan_feat1, (H, W))
        ms_seq1 = ms_enhanced1.flatten(2).transpose(1, 2)
        pan_seq1 = pan_enhanced1.flatten(2).transpose(1, 2)
        fused_seq1 = self.cross_modal_layers[0](ms_seq1, pan_seq1)
        fused_feat1 = fused_seq1.transpose(1, 2).reshape(B, self.embed_dim, H, W)

        # 🌍 世界模型增强: WSM状态调制
        if self.use_wsm and self.wsm is not None:
            # 获取前一时刻隐状态（首次为None）
            h_prev = getattr(self, '_wsm_hidden_state', None)
            # WSM调制: F' = F * (1 + gamma) + beta
            fused_feat1, h_t, gamma, beta = self.wsm(fused_feat1, h_prev)
            # 保存隐状态供下一次使用
            if self.training:
                # 训练时detach避免长期梯度累积
                self._wsm_hidden_state = h_t.detach()
            else:
                # 推理时保持完整状态
                self._wsm_hidden_state = h_t

        # 🔥 优化31: 边缘保护 + 低层细化
        refined_feat1 = self.low_level_refine(fused_feat1)
        edge_weight = self.edge_preserve(fused_feat1)
        fused_feat1 = fused_feat1 + refined_feat1 * edge_weight

        # 🔥 优化32: 调整残差比例，增强融合
        ms_enhanced1 = ms_enhanced1 + fused_feat1 * 0.35
        pan_enhanced1 = pan_enhanced1 + fused_feat1 * 0.35

        # Stage 2: 1/2 分辨率
        ms_feat2 = self.downsample(ms_enhanced1)
        pan_feat2 = self.downsample(pan_enhanced1)
        target_h = max(H // 2, 1)
        target_w = max(W // 2, 1)
        if len(self.mamba_layers) > 1:
            ms_enhanced2, pan_enhanced2 = self.mamba_layers[1](ms_feat2, pan_feat2, (target_h, target_w))
        else:
            ms_enhanced2, pan_enhanced2 = ms_feat2, pan_feat2

        if len(self.cross_modal_layers) > 1:
            ms_seq2 = ms_enhanced2.flatten(2).transpose(1, 2)
            pan_seq2 = pan_enhanced2.flatten(2).transpose(1, 2)
            fused_seq2 = self.cross_modal_layers[1](ms_seq2, pan_seq2)
            fused_feat2_residual = fused_seq2.transpose(1, 2).reshape(B, self.embed_dim, target_h, target_w)
            ms_enhanced2 = ms_enhanced2 + fused_feat2_residual * 0.25
            pan_enhanced2 = pan_enhanced2 + fused_feat2_residual * 0.25

        fused_feat2 = torch.cat([ms_enhanced2, pan_enhanced2], dim=1)
        fused_feat2 = self.fusion_conv1(fused_feat2)

        # Stage 3/4: 1/4 分辨率
        low_h = max(H // 4, 1)
        low_w = max(W // 4, 1)
        low_size = (low_h, low_w)
        fused_feat4 = F.adaptive_avg_pool2d(fused_feat2, low_size)
        low_ms = F.adaptive_avg_pool2d(ms_enhanced2, low_size)
        low_pan = F.adaptive_avg_pool2d(pan_enhanced2, low_size)

        if len(self.mamba_layers) > 2:
            low_ms, low_pan = self.mamba_layers[2](low_ms, low_pan, low_size)
        if len(self.mamba_layers) > 3:
            low_ms, low_pan = self.mamba_layers[3](low_ms, low_pan, low_size)

        if len(self.cross_modal_layers) > 2:
            low_seq_ms = low_ms.flatten(2).transpose(1, 2)
            low_seq_pan = low_pan.flatten(2).transpose(1, 2)
            low_fused_seq = self.cross_modal_layers[2](low_seq_ms, low_seq_pan)
            low_fused_feat = low_fused_seq.transpose(1, 2).reshape(B, self.embed_dim, low_h, low_w)
            low_ms = low_ms + low_fused_feat * 0.25
            low_pan = low_pan + low_fused_feat * 0.25

        low_fused = torch.cat([low_ms, low_pan], dim=1)
        fused_feat4 = fused_feat4 + low_fused
        fused_feat4 = fused_feat4 + self.residual_enhance(fused_feat4) * 0.1
        fused_feat4 = self.fusion_conv2(fused_feat4)

        # 🔥 优化33: 双路注意力增强（频道 + 空间）
        context_weight = self.global_context(fused_feat4)
        spatial_weight = self.spatial_attn(fused_feat4)
        fused_feat4 = fused_feat4 * (1 + context_weight) * (1 + spatial_weight * 0.5)

        # 输出金字塔
        output_1_4 = self.output_conv1(fused_feat4)

        deconv1 = F.relu(self.deconv1(fused_feat4))
        deconv1_cat = torch.cat([deconv1, fused_feat2], dim=1)
        output_1_2 = self.output_conv2(deconv1_cat[:, :self.embed_dim * 2])

        deconv2 = F.relu(self.deconv2(deconv1_cat[:, :self.embed_dim * 3]))
        deconv2_cat = torch.cat([deconv2, fused_feat1], dim=1)
        output_full = self.output_conv3(deconv2_cat[:, :self.embed_dim])

        down_1_2 = F.avg_pool2d(output_full, 2, 2)
        down_1_4 = F.avg_pool2d(down_1_2, 2, 2)

        SR_1_4 = [output_1_4, down_1_4]
        SR_1_2 = [output_1_2, down_1_2]

        return SR_1_4, SR_1_2, output_full


def create_mambairv2_gppnn(model_size='base', **kwargs):
    """
    构建指定规模的 MambaIR-inspired GPPNN 模型。

    Args:
        model_size: 模型大小 ('base' or 'large')
        **kwargs: 额外参数（包括世界模型配置）
            - use_wsm: 启用世界状态记忆
            - wsm_hidden_dim: WSM隐状态维度
            - use_dca_fim: 启用可形变对齐
            - 其他世界模型参数...
    """
    if model_size == 'base':
        return MambaIRv2_GPPNN(
            embed_dim=96,
            d_state=16,
            depths=[6, 6, 6, 6],
            num_heads=[6, 6, 6, 6],
            model_size='base',
            **kwargs
        )
    elif model_size == 'large':
        return MambaIRv2_GPPNN(
            embed_dim=128,
            d_state=20,
            depths=[8, 8, 8, 8],
            num_heads=[8, 8, 8, 8],
            model_size='large',
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported model_size: {model_size}")


if __name__ == "__main__":
    print("==> 构建 MambaIR-inspired GPPNN 模型...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size = 2
    height = width = 256
    ms = torch.randn(batch_size, 3, height, width).to(device)
    pan = torch.randn(batch_size, 1, height, width).to(device)

    print("\n[Base] 模型:")
    model_base = create_mambairv2_gppnn('base').to(device)
    base_params = sum(p.numel() for p in model_base.parameters())
    with torch.no_grad():
        SR_1_4, SR_1_2, output = model_base(ms, pan)
    print(f"  参数量: {base_params:,}")
    print(f"  输出尺寸: {output.shape}")

    print("\n[Large] 模型:")
    model_large = create_mambairv2_gppnn('large').to(device)
    large_params = sum(p.numel() for p in model_large.parameters())
    with torch.no_grad():
        SR_1_4, SR_1_2, output = model_large(ms, pan)
    print(f"  参数量: {large_params:,}")
    print(f"  输出尺寸: {output.shape}")
    print(f"  参数增幅: {((large_params - base_params) / base_params * 100):+.1f}%")

    print("\nMambaIR-inspired GPPNN 初始化完成！")
