# -*- coding: utf-8 -*-
"""
MambaIR-inspired GPPNN Experimental v2.2 配置。

类名和配置键沿用早期版本以保持兼容。这些 Base/Large 预设是本项目的
实验默认值，不是 MambaIR 官方配置；设备与显存字段是未经系统基准
验证的运行参考，不构成性能或硬件保证。
"""

import argparse


class MambaIRv2_GPPNN_Config:
    """MambaIR-inspired GPPNN 的兼容配置类。"""

    # Experimental v2.2 基础预设
    BASE_CONFIG = {
        'model_size': 'base',
        'embed_dim': 96,
        'd_state': 16,
        'depths': [6, 6, 6, 6],
        'num_heads': [6, 6, 6, 6],
        'inner_rank': 32,
        'num_tokens': 128,
        'mlp_ratio': 2.0,
        'window_size': 8,

        # 项目训练默认值（尚未作为最优超参数完成系统验证）
        'batch_size': 12,  # 适度降低batch避免显存压力
        'epochs': 80,      # 增加训练轮数
        'learning_rate': 0.0002,  # 提升初始学习率
        'warmup_epochs': 5,       # 🔥 新增: 学习率warmup
        'min_lr': 1e-6,           # 🔥 新增: 最小学习率
        'weight_decay': 1e-4,
        'img_size': 512,

        # 设备运行参考（不是最低要求或显存保证）
        'min_gpu_memory': '6GB',
        'recommended_gpu': 'RTX 3070',
        'max_batch_size': 8
    }

    # Experimental v2.2 大模型预设
    LARGE_CONFIG = {
        'model_size': 'large',
        'embed_dim': 128,
        'd_state': 20,
        'depths': [8, 8, 8, 8],
        'num_heads': [8, 8, 8, 8],
        'inner_rank': 48,
        'num_tokens': 256,
        'mlp_ratio': 2.0,
        'window_size': 8,

        # 项目训练默认值（尚未作为最优超参数完成系统验证）
        'batch_size': 4,         # 适度提升batch
        'epochs': 100,           # 大模型需要更多轮次
        'learning_rate': 0.0001, # 提升学习率
        'warmup_epochs': 8,      # 更长的warmup
        'min_lr': 5e-7,          # 最小学习率
        'weight_decay': 1e-4,
        'img_size': 512,

        # 设备运行参考（不是最低要求或显存保证）
        'min_gpu_memory': '12GB',
        'recommended_gpu': 'RTX 4080 / A100',
        'max_batch_size': 6
    }

    # 数据配置
    DATA_CONFIG = {
        'photo_root': './photo',
        'train_split': 'dataset',      # 使用dataset目录作为训练集
        'test_split': 'testdateset',   # 使用testdateset目录作为测试集
        'train_ratio': 0.9,            # 训练集比例
        'num_workers': 2,
        'pin_memory': True,

        # 数据增强
        'horizontal_flip': True,
        'vertical_flip': True,
        'random_rotation': True,
        'color_jitter': False  # 全色锐化通常不需要颜色抖动
    }

    # 实验性损失组合默认值
    LOSS_CONFIG = {
        'alpha': 1.0,      # L1损失权重
        'beta': 0.15,      # 梯度损失权重
        'gamma': 0.05,     # 🔥 新增: SSIM损失权重
        'multi_scale': True,      # 多尺度损失
        'perceptual': False,      # 感知损失（可选）
        'edge_aware': True,       # 🔥 新增: 边缘感知损失
        'frequency_loss': True    # 🔥 新增: 频域损失
    }

    # 优化器与学习率调度默认值
    OPTIMIZER_CONFIG = {
        'optimizer': 'AdamW',  # 项目默认优化器
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'scheduler': 'CosineAnnealingWarmRestarts',  # Cosine退火 + 热重启
        'T_0': 20,        # 第一个周期长度
        'T_mult': 2,      # 周期倍增
        'eta_min': 1e-6,  # 最小学习率
        'warmup_epochs': 5  # warmup轮数
    }

    # 可选世界模型实验配置
    WORLD_MODEL_CONFIG = {
        # ========== 模块开关 ==========
        'enable_world_model': False,     # 总开关
        'use_wsm': False,                # 世界状态记忆 (World State Memory)
        'use_dca_fim': False,            # 可形变对齐 (Deformable Cross-Attention)
        'use_dsc': False,                # 物理一致性损失 (Sensor Consistency)
        'use_wacx': False,               # 频域一致性损失 (Cross-band Consistency)
        'use_patch_prior': False,        # Patch Prior修正

        # ========== 损失权重（参考任务计划默认值）==========
        'lambda_s': 0.3,                 # DSC权重
        'lambda_g': 0.05,                # DCA几何权重
        'lambda_w': 0.5,                 # WAC-X频域权重
        'lambda_p': 0.2,                 # Patch先验权重

        # ========== WSM参数 ==========
        'wsm_hidden_dim': 128,           # GRU隐状态维度
        'wsm_dropout': 0.1,              # Dropout率
        'wsm_layer_scale_init': 0.1,    # LayerScale初始值

        # ========== DCA-FIM参数 ==========
        'dca_num_points': 4,             # 形变采样点数量
        'dca_offset_groups': 1,          # 形变分组数
        'dca_deform_weight': 0.3,        # 形变特征融合权重

        # ========== DSC参数 ==========
        'dsc_mtf_kernel_size': 5,        # MTF卷积核大小
        'dsc_mtf_sigma': 1.0,            # 高斯模糊sigma
        'dsc_spectral_response': [0.299, 0.587, 0.114],  # RGB→PAN响应系数
        'dsc_lrms_weight': 0.3,          # LRMS损失权重

        # ========== WAC-X参数 ==========
        'wacx_interband_weight': 1.0,    # 跨带一致性权重
        'wacx_pan_gate_weight': 0.5,     # PAN门控权重
        'wacx_freq_threshold': 0.1,      # 高频阈值

        # ========== Patch Prior参数 ==========
        'patch_size': 32,                # Patch尺寸
        'patch_overlap': 0.25,           # Patch重叠率
        'patch_refiner_path': None,      # 预训练生成器路径（可选）
    }

    @classmethod
    def get_config(cls, model_size='base'):
        """获取指定模型大小的配置"""
        if model_size == 'base':
            config = cls.BASE_CONFIG.copy()
        elif model_size == 'large':
            config = cls.LARGE_CONFIG.copy()
        else:
            raise ValueError(f"不支持的模型大小: {model_size}")

        # 添加其他配置
        config.update(cls.DATA_CONFIG)
        config.update(cls.LOSS_CONFIG)
        config.update(cls.OPTIMIZER_CONFIG)

        return config

    @classmethod
    def print_config(cls, model_size='base'):
        """打印配置信息"""
        config = cls.get_config(model_size)

        print(f"🔧 MambaIR-inspired GPPNN {model_size.upper()} 配置:")
        print(f"   模型参数:")
        print(f"     embed_dim: {config['embed_dim']}")
        print(f"     d_state: {config['d_state']}")
        print(f"     depths: {config['depths']}")
        print(f"     num_heads: {config['num_heads']}")

        print(f"   训练参数:")
        print(f"     batch_size: {config['batch_size']}")
        print(f"     epochs: {config['epochs']}")
        print(f"     learning_rate: {config['learning_rate']}")
        print(f"     img_size: {config['img_size']}x{config['img_size']}")

        print(f"   设备参考（未经系统基准验证）:")
        print(f"     参考显存: {config['min_gpu_memory']}")
        print(f"     参考GPU: {config['recommended_gpu']}")
        print(f"     参考batch_size上限: {config['max_batch_size']}")

        print(f"   数据配置:")
        print(f"     photo_root: {config['photo_root']}")
        print(f"     num_workers: {config['num_workers']}")

        return config

    @classmethod
    def get_world_model_config(cls):
        """获取世界模型配置"""
        return cls.WORLD_MODEL_CONFIG.copy()

    @classmethod
    def print_world_model_config(cls):
        """打印世界模型配置信息"""
        config = cls.get_world_model_config()

        if not config['enable_world_model']:
            print("🌍 世界模型增强: 未启用")
            return

        print("🌍 世界模型增强配置:")
        print(f"   模块状态:")
        print(f"     WSM (世界状态记忆): {config['use_wsm']}")
        print(f"     DCA-FIM (可形变对齐): {config['use_dca_fim']}")
        print(f"     DSC (物理一致性): {config['use_dsc']} (λs={config['lambda_s']})")
        print(f"     WAC-X (频域一致性): {config['use_wacx']} (λw={config['lambda_w']})")
        print(f"     Patch Prior (流形修正): {config['use_patch_prior']} (λp={config['lambda_p']})")

        if config['use_wsm']:
            print(f"   WSM参数:")
            print(f"     hidden_dim={config['wsm_hidden_dim']}, dropout={config['wsm_dropout']}")

        if config['use_dca_fim']:
            print(f"   DCA-FIM参数:")
            print(f"     num_points={config['dca_num_points']}, deform_weight={config['dca_deform_weight']}")

        return config


def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(description='MambaIR-inspired GPPNN 训练/测试')

    # 模型配置
    parser.add_argument('--model_size', type=str, default='base',
                       choices=['base', 'large'],
                       help='模型大小 (base/large)')

    # 训练配置
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批次大小（自动根据模型大小设置）')
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数（自动根据模型大小设置）')
    parser.add_argument('--lr', type=float, default=None,
                       help='学习率（自动根据模型大小设置）')
    parser.add_argument('--img_size', type=int, default=256,
                       help='图像尺寸')

    # 数据配置
    parser.add_argument('--photo_root', type=str, default='../photo',
                       help='Photo目录路径')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='数据加载线程数')

    # 输出配置
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='日志目录')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='结果目录')

    # 设备配置
    parser.add_argument('--device', type=str, default='auto',
                       help='计算设备 (auto/cuda/cpu)')

    return parser


def update_args_with_config(args):
    """根据模型大小更新参数"""
    config = MambaIRv2_GPPNN_Config.get_config(args.model_size)

    # 如果参数没有设置，使用配置文件的默认值
    if args.batch_size is None:
        args.batch_size = config['batch_size']
    if args.epochs is None:
        args.epochs = config['epochs']
    if args.lr is None:
        args.lr = config['learning_rate']

    return args


if __name__ == "__main__":
    # 测试配置
    print("🧪 测试 MambaIR-inspired GPPNN 配置...")

    print("\n" + "="*50)
    MambaIRv2_GPPNN_Config.print_config('base')

    print("\n" + "="*50)
    MambaIRv2_GPPNN_Config.print_config('large')

    print("\n✅ 配置测试完成!")
