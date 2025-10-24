# -*- coding: utf-8 -*-
"""
公平测试脚本 - 512对512
验证MambaIRv2-GPPNN架构在512×512分辨率下的真实性能
"""

import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import time

# 添加模型路径
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import create_mambairv2_gppnn


def calculate_psnr(img1, img2):
    """计算PSNR"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def calculate_ssim_simple(img1, img2):
    """简化SSIM计算"""
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()
    
    if len(img1.shape) == 4:
        img1 = img1[0]
        img2 = img2[0]
    
    # 转换为灰度图
    if img1.shape[0] == 3:
        img1 = np.mean(img1, axis=0)
        img2 = np.mean(img2, axis=0)
    
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1 = np.var(img1)
    sigma2 = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    ssim_val = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
    
    return max(0, min(1, ssim_val))


def load_and_resize_image(img_path, target_size=(512, 512), is_grayscale=False):
    """加载并resize图像到指定尺寸"""
    if is_grayscale:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, target_size)
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).unsqueeze(0)  # [1, H, W]
    else:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img.transpose(2, 0, 1))  # [3, H, W]


def test_fair_512(model_path, test_dir, output_dir, device):
    """公平的512×512测试"""
    
    print(f"🧪 开始公平测试 - 512×512对512×512")
    print("="*60)
    print(f"   模型路径: {model_path}")
    print(f"   测试目录: {test_dir}")
    print(f"   输出目录: {output_dir}")
    print(f"   设备: {device}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    print("\n🏗️ 加载模型...")
    checkpoint = torch.load(model_path, map_location=device)
    model_size = checkpoint.get('config', {}).get('model_size', 'base')
    print(f"   模型大小: {model_size}")
    
    model = create_mambairv2_gppnn(model_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"   最佳PSNR: {checkpoint.get('best_psnr', '未知'):.2f}dB")
    print(f"   训练轮数: {checkpoint.get('epoch', '未知')}")
    
    # 获取测试图像列表
    gt_dir = os.path.join(test_dir, 'GT')
    ms_dir = os.path.join(test_dir, 'MS') 
    pan_dir = os.path.join(test_dir, 'PAN')
    
    if not all(os.path.exists(d) for d in [gt_dir, ms_dir, pan_dir]):
        raise FileNotFoundError(f"测试目录不完整: {test_dir}")
    
    img_names = [f for f in os.listdir(gt_dir) if f.lower().endswith(('.jpg', '.png'))]
    img_names = sorted(img_names)
    print(f"\n📊 找到测试图像: {len(img_names)}张")
    
    # 测试统计
    total_psnr = 0.0
    total_ssim = 0.0
    valid_count = 0
    
    print(f"\n🔬 开始逐张测试...")
    start_time = time.time()
    
    with torch.no_grad():
        for i, img_name in enumerate(tqdm(img_names, desc="测试进度")):
            try:
                # 加载原始图像并resize到512×512
                gt_path = os.path.join(gt_dir, img_name)
                ms_path = os.path.join(ms_dir, img_name)
                pan_path = os.path.join(pan_dir, img_name)
                
                # 读取并resize为512×512 (关键步骤!)
                gt_512 = load_and_resize_image(gt_path, (512, 512), False)
                ms_512 = load_and_resize_image(ms_path, (512, 512), False) 
                pan_512 = load_and_resize_image(pan_path, (512, 512), True)
                
                # 添加batch维度并移到GPU
                gt_512 = gt_512.unsqueeze(0).to(device)    # [1, 3, 512, 512]
                ms_512 = ms_512.unsqueeze(0).to(device)    # [1, 3, 512, 512]
                pan_512 = pan_512.unsqueeze(0).to(device)  # [1, 1, 512, 512]
                
                # 模型推理
                outputs = model(ms_512, pan_512)
                _, _, output_full = outputs
                
                # 计算指标 (512×512 vs 512×512 - 完全公平!)
                psnr = calculate_psnr(output_full, gt_512)
                ssim = calculate_ssim_simple(output_full, gt_512)
                
                total_psnr += psnr.item()
                total_ssim += ssim
                valid_count += 1
                
                # 保存结果图像 (只保存前6张)
                if i < 6:
                    # 转换为numpy保存
                    output_np = output_full[0].cpu().numpy().transpose(1, 2, 0)
                    output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)
                    output_pil = Image.fromarray(output_np)
                    
                    # 保存结果
                    result_name = f"result_{i:03d}_{img_name}"
                    output_pil.save(os.path.join(output_dir, result_name))
                    
                    # 也保存GT对比
                    gt_np = gt_512[0].cpu().numpy().transpose(1, 2, 0)
                    gt_np = np.clip(gt_np * 255, 0, 255).astype(np.uint8)
                    gt_pil = Image.fromarray(gt_np)
                    gt_name = f"gt_{i:03d}_{img_name}"
                    gt_pil.save(os.path.join(output_dir, gt_name))
                
                # 实时显示进度
                if i % 20 == 0 and i > 0:
                    avg_psnr = total_psnr / valid_count
                    avg_ssim = total_ssim / valid_count
                    print(f"\n   📊 中间结果 ({i}/{len(img_names)}张): PSNR={avg_psnr:.2f}dB, SSIM={avg_ssim:.4f}")
                    
            except Exception as e:
                print(f"\n❌ 处理{img_name}时出错: {e}")
                continue
    
    # 最终统计
    test_time = time.time() - start_time
    
    if valid_count > 0:
        avg_psnr = total_psnr / valid_count
        avg_ssim = total_ssim / valid_count
        
        print(f"\n🎉 公平测试完成!")
        print("="*60)
        print(f"📊 最终结果 (512×512对512×512):")
        print(f"   测试图像: {valid_count}/{len(img_names)}张")
        print(f"   平均PSNR: {avg_psnr:.2f}dB")
        print(f"   平均SSIM: {avg_ssim:.4f}")
        print(f"   测试耗时: {test_time:.1f}秒")
        print(f"   平均每张: {test_time/valid_count:.2f}秒")
        
        # 性能评价
        print(f"\n💡 性能评价:")
        if avg_psnr >= 32:
            print(f"   PSNR {avg_psnr:.2f}dB - 🌟 优秀!")
        elif avg_psnr >= 30:
            print(f"   PSNR {avg_psnr:.2f}dB - ✅ 良好!")
        elif avg_psnr >= 27:
            print(f"   PSNR {avg_psnr:.2f}dB - ⚠️  一般")
        else:
            print(f"   PSNR {avg_psnr:.2f}dB - ❌ 需要改进")
            
        if avg_ssim >= 0.95:
            print(f"   SSIM {avg_ssim:.4f} - 🌟 优秀!")
        elif avg_ssim >= 0.90:
            print(f"   SSIM {avg_ssim:.4f} - ✅ 良好!")
        else:
            print(f"   SSIM {avg_ssim:.4f} - ⚠️  需要改进")
        
        # 保存测试报告
        report = {
            'model_path': model_path,
            'test_images': valid_count,
            'avg_psnr': avg_psnr,
            'avg_ssim': avg_ssim,
            'test_time': test_time,
            'resolution': '512x512_fair_test'
        }
        
        import json
        with open(os.path.join(output_dir, 'test_report_512_fair.json'), 'w') as f:
            json.dump(report, f, indent=4)
        
        return avg_psnr, avg_ssim
    else:
        print("\n❌ 没有成功处理任何图像!")
        return 0, 0


def main():
    parser = argparse.ArgumentParser(description='公平测试 - 512×512对512×512')
    parser.add_argument('--model_path', type=str, 
                       default='checkpoints/mambairv2_gppnn_latest/models/best_model.pth',
                       help='模型checkpoint路径')
    parser.add_argument('--test_dir', type=str, 
                       default='photo/testdateset',
                       help='测试数据目录')
    parser.add_argument('--output_dir', type=str,
                       default='test_results_512_fair',
                       help='输出目录')
    parser.add_argument('--device', type=str, default='auto',
                       help='设备 (auto/cuda/cpu)')
    
    args = parser.parse_args()
    
    # 设备设置
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # 开始测试
    test_fair_512(args.model_path, args.test_dir, args.output_dir, device)


if __name__ == '__main__':
    main()
