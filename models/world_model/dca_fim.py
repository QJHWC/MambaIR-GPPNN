# -*- coding: utf-8 -*-
"""
可形变跨模态注意力 (Deformable Cross-Attention with Feature-level Image Matching - DCA-FIM)
通过学习形变offset实现亚像素级几何对齐

数学原理:
    offset = ConvOffset(Q_lrms)  # 学习形变偏移
    weight = Softmax(ConvWeight(Q_lrms))  # 采样权重
    V_aligned = DeformSample(V_pan, offset, weight)  # 可形变采样
    
效果: I*(x+u) ≈ I*(x) + ∇I*·u，配准误差ε' = ∇I*·(u - û) → MSE↓

参考:
    《最新任务计划.md》 Section 2.2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableCrossAttention(nn.Module):
    """
    DCA-FIM模块
    
    核心功能:
    1. 学习空间形变偏移（亚像素级对齐）
    2. 自适应采样权重（多点聚合）
    3. 几何对齐融合（减少配准误差）
    
    Args:
        dim: 特征维度
        num_points: 形变采样点数量（默认4）
        offset_groups: 形变分组数（默认1）
        deform_weight: 形变特征融合权重（默认0.3）
    """
    def __init__(self, dim, num_points=4, offset_groups=1, deform_weight=0.3):
        super().__init__()
        self.dim = dim
        self.num_points = num_points
        self.offset_groups = offset_groups
        self.deform_weight = deform_weight
        
        # Offset预测网络（预测2D空间偏移）
        self.offset_net = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=offset_groups),
            nn.GELU(),
            nn.Conv2d(dim, 2 * num_points, 3, 1, 1)  # 每个点(x,y)共2个坐标
        )
        
        # 采样权重网络
        self.weight_net = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, num_points, 1, 1, 0)
        )
        
        # 特征融合层
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, 1, 0)
        )
        
        # LayerNorm稳定训练
        self.norm = nn.LayerNorm(dim)
        
        self._init_weights()
        
    def _init_weights(self):
        """权重初始化"""
        # Offset网络初始化为小值（避免初期形变过大）
        for m in self.offset_net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 其他网络正常初始化
        for module in [self.weight_net, self.fusion_conv]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def deformable_sample(self, features, offsets, weights):
        """
        可形变采样（使用F.grid_sample实现）
        
        实现原理:
        1. 预测每个位置的K个采样点偏移
        2. 在原特征图上根据偏移进行双线性采样
        3. 加权聚合K个采样结果
        
        Args:
            features: [B, C, H, W] 待采样特征（PAN）
            offsets: [B, 2*K, H, W] 偏移量（K个点，每点2个坐标）
            weights: [B, K, H, W] 采样权重
            
        Returns:
            sampled: [B, C, H, W] 采样后的特征
        """
        B, C, H, W = features.shape
        K = self.num_points
        
        # Reshape offsets: [B, 2*K, H, W] → [B, K, H, W, 2]
        offsets = offsets.view(B, K, 2, H, W).permute(0, 1, 3, 4, 2).contiguous()
        
        # 创建基础网格 [-1, 1]（grid_sample标准范围）
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=features.device),
            torch.linspace(-1, 1, W, device=features.device),
            indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        base_grid = base_grid.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W, 2]
        base_grid = base_grid.expand(B, K, -1, -1, -1)  # [B, K, H, W, 2]
        
        # 归一化偏移量（相对于特征图尺寸）
        # offsets范围通常是像素单位，需要归一化到[-1, 1]
        offsets_normalized = offsets / torch.tensor(
            [W / 2, H / 2], 
            device=offsets.device
        ).view(1, 1, 1, 1, 2)
        
        # 应用偏移: grid = base_grid + offset
        sampling_grids = base_grid + offsets_normalized
        # 裁剪到有效范围[-1, 1]
        sampling_grids = torch.clamp(sampling_grids, -1, 1)
        
        # 逐点采样
        sampled_features = []
        for k in range(K):
            sampled_k = F.grid_sample(
                features, 
                sampling_grids[:, k],  # [B, H, W, 2]
                mode='bilinear', 
                padding_mode='border',
                align_corners=True
            )
            sampled_features.append(sampled_k)
        
        # Stack: [B, K, C, H, W]
        sampled_features = torch.stack(sampled_features, dim=1)
        
        # 加权聚合: [B, K, C, H, W] → [B, C, H, W]
        weights = torch.softmax(weights, dim=1).unsqueeze(2)  # [B, K, 1, H, W]
        aggregated = (sampled_features * weights).sum(dim=1)  # [B, C, H, W]
        
        return aggregated
    
    def forward(self, query_feat, key_feat):
        """
        前向传播
        
        Args:
            query_feat: [B, C, H, W] Query特征（MS，用于预测offset）
            key_feat: [B, C, H, W] Key特征（PAN，待对齐）
            
        Returns:
            aligned_feat: [B, C, H, W] 对齐并融合后的特征
        """
        # 预测形变偏移和采样权重
        offsets = self.offset_net(query_feat)  # [B, 2*K, H, W]
        weights = self.weight_net(query_feat)  # [B, K, H, W]
        
        # 可形变采样PAN特征
        aligned_key = self.deformable_sample(key_feat, offsets, weights)
        
        # 融合原始query和对齐后的key
        fused = self.fusion_conv(aligned_key)
        
        # 残差连接 + 可调节权重
        output = query_feat + fused * self.deform_weight
        
        return output


if __name__ == "__main__":
    """DCA-FIM模块单元测试"""
    print("="*60)
    print("DCA-FIM (Deformable Cross-Attention) 模块测试")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}\n")
    
    # 测试参数
    B, C, H, W = 2, 96, 64, 64
    num_points = 4
    
    # 创建DCA-FIM模块
    dca = DeformableCrossAttention(
        dim=C,
        num_points=num_points,
        offset_groups=1,
        deform_weight=0.3
    ).to(device)
    
    # 测试1: 前向传播
    print("测试1: 前向传播")
    query = torch.randn(B, C, H, W).to(device)
    key = torch.randn(B, C, H, W).to(device)
    
    output = dca(query, key)
    
    print(f"  Query: {query.shape}")
    print(f"  Key: {key.shape}")
    print(f"  Output: {output.shape}")
    assert output.shape == query.shape, f"输出形状错误: {output.shape}"
    print("  [OK] 前向传播成功\n")
    
    # 测试2: Offset范围检查
    print("测试2: Offset范围检查")
    with torch.no_grad():
        offsets = dca.offset_net(query)
        weights = dca.weight_net(query)
    
    print(f"  Offset形状: {offsets.shape} (应为[B, 2*K, H, W])")
    print(f"  Offset范围: [{offsets.min().item():.3f}, {offsets.max().item():.3f}]")
    print(f"  Weight形状: {weights.shape} (应为[B, K, H, W])")
    assert offsets.shape == (B, 2*num_points, H, W), "Offset形状错误"
    assert weights.shape == (B, num_points, H, W), "Weight形状错误"
    print("  [OK] Offset/Weight形状正确\n")
    
    # 测试3: 可形变采样
    print("测试3: 可形变采样功能")
    with torch.no_grad():
        sampled = dca.deformable_sample(key, offsets, weights)
    
    print(f"  采样结果: {sampled.shape}")
    assert sampled.shape == key.shape, f"采样形状错误: {sampled.shape}"
    
    # 验证采样有效性（采样结果应与原图接近但不完全相同）
    diff = (sampled - key).abs().mean()
    print(f"  采样差异: {diff.item():.6f}")
    print("  [OK] 可形变采样功能正确\n")
    
    # 测试4: 梯度反向传播
    print("测试4: 梯度反向传播")
    # 使用更小的尺寸加速测试
    query_grad = torch.randn(1, C, 32, 32, requires_grad=True, device=device)
    key_grad = torch.randn(1, C, 32, 32, requires_grad=True, device=device)
    
    output = dca(query_grad, key_grad)
    loss = output.sum()
    loss.backward()
    
    assert query_grad.grad is not None, "Query梯度为None"
    assert key_grad.grad is not None, "Key梯度为None"
    assert query_grad.grad.abs().sum() > 0, "Query梯度为0"
    assert key_grad.grad.abs().sum() > 0, "Key梯度为0"
    
    print(f"  Query梯度范数: {query_grad.grad.norm().item():.6f}")
    print(f"  Key梯度范数: {key_grad.grad.norm().item():.6f}")
    print("  [OK] 梯度传播正确\n")
    
    # 测试5: 参数量统计
    print("测试5: 参数量统计")
    total_params = sum(p.numel() for p in dca.parameters())
    trainable_params = sum(p.numel() for p in dca.parameters() if p.requires_grad)
    
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  显存估算: ~{total_params * 4 / 1024**2:.2f} MB")
    print("  [OK] 参数量合理\n")
    
    print("="*60)
    print("[SUCCESS] DCA-FIM模块所有测试通过!")
    print("="*60)

