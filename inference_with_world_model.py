# -*- coding: utf-8 -*-
"""
世界模型增强推理脚本
在标准推理基础上添加Patch Prior修正

使用方法:
    # 标准推理（无Patch Prior）
    python inference_with_world_model.py --model_path checkpoints/.../best_model.pth
    
    # 启用Patch Prior增强
    python inference_with_world_model.py --model_path checkpoints/.../best_model.pth --use_patch_prior
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from tqdm import tqdm
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import create_mambairv2_gppnn
from models.world_model import PatchPriorRefiner


def calculate_psnr(img1, img2):
    """计算PSNR"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def load_image(path, device, img_size=None):
    """加载图像"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if img_size is not None:
        img = cv2.resize(img, img_size)
    
    img = img.astype(np.float32) / 255.0
    img = np.clip(img, 0.0, 1.0)
    img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)
    
    return img


def main():
    parser = argparse.ArgumentParser(description='World Model Enhanced Inference')
    parser.add_argument('--model_path', type=str, required=True, help='模型checkpoint路径')
    parser.add_argument('--test_dir', type=str, default='photo/testdateset', help='测试数据目录')
    parser.add_argument('--output_dir', type=str, default='results_world_model', help='输出目录')
    parser.add_argument('--use_patch_prior', action='store_true', help='启用Patch Prior增强')
    parser.add_argument('--patch_size', type=int, default=32, help='Patch尺寸')
    parser.add_argument('--device', type=str, default='auto', help='设备 (auto/cuda/cpu)')
    parser.add_argument('--img_size', type=int, default=None, help='图像尺寸（None=原尺寸）')
    args = parser.parse_args()
    
    # 设备设置
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("="*70)
    print("世界模型增强推理")
    print("="*70)
    print(f"设备: {device}")
    print(f"模型: {args.model_path}")
    print(f"Patch Prior: {'启用' if args.use_patch_prior else '未启用'}")
    print()
    
    # 加载模型
    print("加载模型...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # 获取模型配置
    model_config = checkpoint.get('config', {})
    model_size = model_config.get('model_size', 'base')
    
    # 检查是否使用世界模型训练
    use_wsm = model_config.get('use_wsm', False)
    use_dca_fim = model_config.get('use_dca_fim', False)
    
    print(f"模型大小: {model_size}")
    print(f"训练时启用WSM: {use_wsm}")
    print(f"训练时启用DCA-FIM: {use_dca_fim}")
    
    # 创建模型（使用训练时相同的配置）
    model = create_mambairv2_gppnn(
        model_size,
        use_wsm=use_wsm,
        use_dca_fim=use_dca_fim
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    
    print("模型加载成功！\n")
    
    # 创建Patch Refiner
    refiner = None
    if args.use_patch_prior:
        refiner = PatchPriorRefiner(patch_size=args.patch_size, overlap=0.25)
        print(f"Patch Prior Refiner已创建 (patch_size={args.patch_size})\n")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取测试图像列表
    ms_dir = os.path.join(args.test_dir, 'MS')
    pan_dir = os.path.join(args.test_dir, 'PAN')
    gt_dir = os.path.join(args.test_dir, 'GT')
    
    img_names = sorted([f for f in os.listdir(ms_dir) if f.lower().endswith(('.jpg', '.png'))])
    
    print(f"测试图像数量: {len(img_names)}")
    print("="*70)
    print()
    
    # 推理
    total_psnr = 0.0
    total_time = 0.0
    
    with torch.no_grad():
        for idx, img_name in enumerate(tqdm(img_names, desc="推理进度")):
            # 加载图像
            ms_path = os.path.join(ms_dir, img_name)
            pan_path = os.path.join(pan_dir, img_name)
            gt_path = os.path.join(gt_dir, img_name)
            
            # 简化版：使用RGB图像
            ms = load_image(ms_path, device, None)
            pan_gray = cv2.imread(pan_path, cv2.IMREAD_GRAYSCALE)
            pan = torch.from_numpy(pan_gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
            gt = load_image(gt_path, device, None)
            
            # 确保尺寸一致
            if ms.shape[2:] != pan.shape[2:]:
                pan = F.interpolate(pan, size=ms.shape[2:], mode='bilinear', align_corners=False)
            
            # 推理
            start_time = time.time()
            _, _, output = model(ms, pan)
            
            # Patch Prior修正
            if refiner is not None:
                output = refiner.refine(output)
            
            infer_time = time.time() - start_time
            total_time += infer_time
            
            # 计算PSNR
            psnr = calculate_psnr(output, gt)
            total_psnr += psnr.item()
            
            # 保存结果（可选）
            # save_image(output, os.path.join(args.output_dir, img_name))
    
    # 统计结果
    avg_psnr = total_psnr / len(img_names)
    avg_time = total_time / len(img_names)
    
    print()
    print("="*70)
    print("推理完成！")
    print("="*70)
    print(f"平均PSNR: {avg_psnr:.2f} dB")
    print(f"平均推理时间: {avg_time:.3f} 秒/张")
    print(f"总图像数: {len(img_names)}")
    print(f"结果保存至: {args.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()

