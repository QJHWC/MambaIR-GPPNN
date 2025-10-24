# -*- coding: utf-8 -*-
"""
🚀 MambaIRv2-GPPNN 统一训练脚本
支持 Base/Large 模型 + 256/512 分辨率
云端快速部署专用

命令行示例:
# Base模型 256分辨率
python train_unified.py --model_size base --img_size 256 --batch_size 16 --epochs 80

# Base模型 512分辨率
python train_unified.py --model_size base --img_size 512 --batch_size 12 --epochs 80

# Large模型 256分辨率
python train_unified.py --model_size large --img_size 256 --batch_size 8 --epochs 100

# Large模型 512分辨率
python train_unified.py --model_size large --img_size 512 --batch_size 4 --epochs 100
"""

import os
import sys
import argparse
import torch
from datetime import datetime

# 导入原始训练脚本的主函数
from train import main as train_main

def create_unified_args():
    """创建统一的参数解析器"""
    parser = argparse.ArgumentParser(description='🚀 MambaIRv2-GPPNN 统一训练脚本')

    # 🔥 核心参数
    parser.add_argument('--model_size', type=str, default='base', choices=['base', 'large'],
                       help='模型大小 (base/large)')
    parser.add_argument('--img_size', type=int, default=256, choices=[256, 512],
                       help='图像尺寸 (256/512)')

    # 🔥 自动适配参数（可选覆盖）
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批次大小（自动适配或手动指定）')
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数（自动适配或手动指定）')
    parser.add_argument('--lr', type=float, default=None,
                       help='学习率（自动适配或手动指定）')

    # 数据配置
    parser.add_argument('--photo_root', type=str, default='./photo',
                       help='Photo目录路径')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='数据加载线程数')

    # 输出配置
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='日志目录')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='结果目录')

    # 其他配置
    parser.add_argument('--save_freq', type=int, default=5,
                       help='保存频率（epochs）')
    parser.add_argument('--log_freq', type=int, default=10,
                       help='日志频率（batches）')
    parser.add_argument('--val_freq', type=int, default=10,
                       help='验证频率（epochs）')
    parser.add_argument('--grad_clip_norm', type=float, default=0.1,
                       help='梯度裁剪')
    parser.add_argument('--device', type=str, default='auto',
                       help='设备 (auto/cuda/cpu)')
    parser.add_argument('--resume', type=str, default='',
                       help='断点续训路径')
    parser.add_argument('--auto_resume', action='store_true',
                       help='自动断点续训')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='权重衰减')
    parser.add_argument('--auto_batch_size', action='store_true',
                       help='🔥 自动查找最大可用batch_size（推荐）')
    
    # 🌍 世界模型增强参数
    parser.add_argument('--enable_world_model', action='store_true',
                       help='启用世界模型增强（总开关）')
    parser.add_argument('--world_model_preset', type=str, default='full',
                       choices=['wsm_only', 'dsc_only', 'wsm_dsc', 'full', 'custom'],
                       help='世界模型预设配置')

    return parser

def auto_configure(args):
    """🔥 智能自动配置参数"""

    print("="*70)
    print("🚀 MambaIRv2-GPPNN 统一训练脚本")
    print("="*70)

    # 显示核心配置
    print(f"\n📋 核心配置:")
    print(f"   模型大小: {args.model_size.upper()}")
    print(f"   图像尺寸: {args.img_size}×{args.img_size}")

    # 🔥 智能适配batch_size (v2.2优化版: 更激进的默认值)
    if args.batch_size is None:
        if args.model_size == 'base':
            args.batch_size = 8 if args.img_size == 256 else 4  # v2.2: 提升默认batch
        else:  # large
            args.batch_size = 4 if args.img_size == 256 else 2  # v2.2: 提升默认batch
        print(f"   Batch Size: {args.batch_size} (v2.2优化-充分利用GPU)")
    else:
        print(f"   Batch Size: {args.batch_size} (手动指定)")

    # 🔥 智能适配epochs
    if args.epochs is None:
        args.epochs = 80 if args.model_size == 'base' else 100
        print(f"   训练轮数: {args.epochs} (自动适配)")
    else:
        print(f"   训练轮数: {args.epochs} (手动指定)")

    # 🔥 智能适配学习率
    if args.lr is None:
        if args.model_size == 'base':
            args.lr = 0.0002
        else:  # large
            args.lr = 0.0001
        print(f"   学习率: {args.lr} (自动适配)")
    else:
        print(f"   学习率: {args.lr} (手动指定)")

    # 生成唯一的保存目录
    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_name = f"{args.model_size}_{args.img_size}_{timestamp}"
    args.save_dir = os.path.join(args.save_dir, f"mambairv2_gppnn_{run_name}")
    args.log_dir = os.path.join(args.log_dir, run_name)

    print(f"\n💾 输出路径:")
    print(f"   保存目录: {args.save_dir}")
    print(f"   日志目录: {args.log_dir}")

    # 🌍 世界模型预设配置
    if args.enable_world_model:
        print(f"\n🌍 世界模型增强:")
        if args.world_model_preset == 'wsm_only':
            args.use_wsm = True
            args.use_dsc = False
            args.use_wacx = False
            args.use_dca_fim = False
            print("   预设: WSM Only (仅时序一致性)")
        elif args.world_model_preset == 'dsc_only':
            args.use_wsm = False
            args.use_dsc = True
            args.use_wacx = False
            args.use_dca_fim = False
            print("   预设: DSC Only (仅物理约束)")
        elif args.world_model_preset == 'wsm_dsc':
            args.use_wsm = True
            args.use_dsc = True
            args.use_wacx = False
            args.use_dca_fim = False
            print("   预设: WSM+DSC (核心功能)")
        elif args.world_model_preset == 'full':
            args.use_wsm = True
            args.use_dsc = True
            args.use_wacx = True
            args.use_dca_fim = True
            print("   预设: Full (全模块启用)")
        
        print(f"   模块状态: WSM={args.use_wsm}, DCA={args.use_dca_fim}, DSC={args.use_dsc}, WAC-X={args.use_wacx}")

    # 显存和性能预估 (v2.2更新)
    print(f"\n⚡ 性能预估 (v2.2优化版):")
    if args.model_size == 'base':
        if args.img_size == 256:
            print(f"   显存需求: ~6-8GB (batch_size={args.batch_size})")
            print(f"   训练速度: 快 (~2-3 sec/batch)")
            print(f"   预计时长: 4-6小时 (80 epochs)")
            print(f"   预期PSNR: 27-30dB | SSIM: 0.7-0.85")
        else:  # 512
            print(f"   显存需求: ~8-12GB (batch_size={args.batch_size})")
            print(f"   训练速度: 中 (~4-6 sec/batch)")
            print(f"   预计时长: 10-14小时 (80 epochs)")
            print(f"   预期PSNR: 28-31dB | SSIM: 0.75-0.9")
    else:  # large
        if args.img_size == 256:
            print(f"   显存需求: ~10-14GB (batch_size={args.batch_size})")
            print(f"   训练速度: 中 (~4-5 sec/batch)")
            print(f"   预计时长: 14-18小时 (100 epochs)")
            print(f"   预期PSNR: 29-32dB | SSIM: 0.8-0.9")
        else:  # 512
            print(f"   显存需求: ~16-20GB (batch_size={args.batch_size})")
            print(f"   训练速度: 慢 (~8-12 sec/batch)")
            print(f"   预计时长: 20-28小时 (100 epochs)")
            print(f"   预期PSNR: 30-33dB | SSIM: 0.85-0.95")

    print("\n" + "="*70)

    return args

def main():
    """主函数"""
    parser = create_unified_args()
    args = parser.parse_args()

    # 🔥 自动配置参数
    args = auto_configure(args)

    # 确认开始训练
    print("\n⏳ 3秒后开始训练...")
    print("   按 Ctrl+C 取消")
    import time
    try:
        time.sleep(3)
    except KeyboardInterrupt:
        print("\n❌ 训练已取消")
        return

    # 调用原始训练主函数（通过修改sys.argv传参）
    original_argv = sys.argv.copy()
    sys.argv = [
        'train.py',
        '--model_size', args.model_size,
        '--batch_size', str(args.batch_size),
        '--epochs', str(args.epochs),
        '--lr', str(args.lr),
        '--img_size', str(args.img_size),
        '--photo_root', args.photo_root,
        '--num_workers', str(args.num_workers),
        '--save_dir', args.save_dir,
        '--log_dir', args.log_dir,
        '--save_freq', str(args.save_freq),
        '--log_freq', str(args.log_freq),
        '--val_freq', str(args.val_freq),
        '--grad_clip_norm', str(args.grad_clip_norm),
        '--device', args.device,
        '--weight_decay', str(args.weight_decay)
    ]

    if args.resume:
        sys.argv.extend(['--resume', args.resume])
    if args.auto_resume:
        sys.argv.append('--auto_resume')
    if args.auto_batch_size:
        sys.argv.append('--auto_batch_size')
    
    # 🌍 世界模型参数传递
    if args.enable_world_model:
        sys.argv.append('--enable_world_model')
        if hasattr(args, 'use_wsm') and args.use_wsm:
            sys.argv.append('--use_wsm')
        if hasattr(args, 'use_dca_fim') and args.use_dca_fim:
            sys.argv.append('--use_dca_fim')
        if hasattr(args, 'use_dsc') and args.use_dsc:
            sys.argv.append('--use_dsc')
        if hasattr(args, 'use_wacx') and args.use_wacx:
            sys.argv.append('--use_wacx')

    try:
        # 调用train.py的main函数
        train_main()
    finally:
        sys.argv = original_argv

if __name__ == '__main__':
    main()
