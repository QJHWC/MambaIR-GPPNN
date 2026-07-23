# -*- coding: utf-8 -*-
"""
公平测试脚本 - 256对256
验证MambaIRv2-GPPNN架构在同分辨率下的真实性能
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
import json

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


def load_and_resize_image(img_path, target_size=(256, 256), is_grayscale=False):
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


def test_fair_256(model_path, test_dir, output_dir, device):
    """公平的256×256测试 - 增强版（包含详细指标）"""

    print(f"🧪 开始公平测试 - 256×256对256×256 (增强版)")
    print("="*70)
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

    # 🔥 新增：加载损失函数计算器
    from train import IRDN_Loss
    criterion = IRDN_Loss(alpha=1.0, beta=0.3, gamma=0.2, edge_aware=True, freq_loss=True).to(device)

    # 获取测试图像列表
    gt_dir = os.path.join(test_dir, 'GT')
    ms_dir = os.path.join(test_dir, 'MS')
    pan_dir = os.path.join(test_dir, 'PAN')

    if not all(os.path.exists(d) for d in [gt_dir, ms_dir, pan_dir]):
        raise FileNotFoundError(f"测试目录不完整: {test_dir}")

    img_names = [f for f in os.listdir(gt_dir) if f.lower().endswith(('.jpg', '.png'))]
    img_names = sorted(img_names)
    print(f"\n📊 找到测试图像: {len(img_names)}张")

    # 🔥 增强的测试统计
    total_psnr = 0.0
    total_ssim = 0.0
    total_loss = 0.0
    total_l1_loss = 0.0
    total_grad_loss = 0.0
    total_ssim_loss = 0.0
    total_edge_loss = 0.0
    total_freq_loss = 0.0
    total_mse = 0.0

    psnr_list = []
    ssim_list = []
    loss_list = []

    valid_count = 0

    print(f"\n🔬 开始逐张测试...")
    start_time = time.time()

    with torch.no_grad():
        for i, img_name in enumerate(tqdm(img_names, desc="测试进度")):
            try:
                # 加载原始图像并resize到256×256
                gt_path = os.path.join(gt_dir, img_name)
                ms_path = os.path.join(ms_dir, img_name)
                pan_path = os.path.join(pan_dir, img_name)

                # 读取并resize为256×256 (关键步骤!)
                gt_256 = load_and_resize_image(gt_path, (256, 256), False)
                ms_256 = load_and_resize_image(ms_path, (256, 256), False)
                pan_256 = load_and_resize_image(pan_path, (256, 256), True)

                # 添加batch维度并移到GPU
                gt_256 = gt_256.unsqueeze(0).to(device)    # [1, 3, 256, 256]
                ms_256 = ms_256.unsqueeze(0).to(device)    # [1, 3, 256, 256]
                pan_256 = pan_256.unsqueeze(0).to(device)  # [1, 1, 256, 256]

                # 模型推理
                outputs = model(ms_256, pan_256)
                _, _, output_full = outputs

                # 🔥 计算详细指标 (256×256 vs 256×256 - 完全公平!)
                psnr = calculate_psnr(output_full, gt_256)
                ssim = calculate_ssim_simple(output_full, gt_256)

                # 🔥 新增：计算各项损失
                loss_dict = criterion(outputs, gt_256)
                total_loss_val = loss_dict['total_loss'].item()
                l1_loss_val = loss_dict['l1_loss'].item()
                grad_loss_val = loss_dict['grad_loss'].item()
                ssim_loss_val = loss_dict['ssim_loss'].item()
                edge_loss_val = loss_dict['edge_loss'].item()
                freq_loss_val = loss_dict['freq_loss'].item()

                # 🔥 新增：计算MSE
                mse = torch.mean((output_full - gt_256) ** 2).item()

                # 累加统计
                total_psnr += psnr.item()
                total_ssim += ssim
                total_loss += total_loss_val
                total_l1_loss += l1_loss_val
                total_grad_loss += grad_loss_val
                total_ssim_loss += ssim_loss_val
                total_edge_loss += edge_loss_val
                total_freq_loss += freq_loss_val
                total_mse += mse

                # 保存单张数据（用于分析方差）
                psnr_list.append(psnr.item())
                ssim_list.append(ssim)
                loss_list.append(total_loss_val)

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
                    gt_np = gt_256[0].cpu().numpy().transpose(1, 2, 0)
                    gt_np = np.clip(gt_np * 255, 0, 255).astype(np.uint8)
                    gt_pil = Image.fromarray(gt_np)
                    gt_name = f"gt_{i:03d}_{img_name}"
                    gt_pil.save(os.path.join(output_dir, gt_name))

                # 🔥 增强的实时显示进度
                if i % 20 == 0 and i > 0:
                    avg_psnr = total_psnr / valid_count
                    avg_ssim = total_ssim / valid_count
                    avg_loss = total_loss / valid_count
                    avg_mse = total_mse / valid_count
                    print(f"\n   📊 中间结果 ({i}/{len(img_names)}张):")
                    print(f"      PSNR={avg_psnr:.2f}dB, SSIM={avg_ssim:.4f}, Loss={avg_loss:.6f}, MSE={avg_mse:.6f}")

            except Exception as e:
                print(f"\n❌ 处理{img_name}时出错: {e}")
                continue

    # 最终统计
    test_time = time.time() - start_time

    if valid_count > 0:
        # 计算平均值
        avg_psnr = total_psnr / valid_count
        avg_ssim = total_ssim / valid_count
        avg_loss = total_loss / valid_count
        avg_l1_loss = total_l1_loss / valid_count
        avg_grad_loss = total_grad_loss / valid_count
        avg_ssim_loss = total_ssim_loss / valid_count
        avg_edge_loss = total_edge_loss / valid_count
        avg_freq_loss = total_freq_loss / valid_count
        avg_mse = total_mse / valid_count

        # 🔥 计算标准差（衡量稳定性）
        psnr_std = np.std(psnr_list)
        ssim_std = np.std(ssim_list)
        loss_std = np.std(loss_list)
        psnr_min = np.min(psnr_list)
        psnr_max = np.max(psnr_list)
        ssim_min = np.min(ssim_list)
        ssim_max = np.max(ssim_list)

        print(f"\n🎉 公平测试完成!")
        print("="*70)
        print(f"📊 最终结果 (256×256对256×256) - 增强版:")
        print(f"   测试图像: {valid_count}/{len(img_names)}张")
        print(f"\n📈 主要指标:")
        print(f"   平均PSNR: {avg_psnr:.4f}dB  (范围: {psnr_min:.2f} ~ {psnr_max:.2f}dB, 标准差: {psnr_std:.3f})")
        print(f"   平均SSIM: {avg_ssim:.4f}    (范围: {ssim_min:.4f} ~ {ssim_max:.4f}, 标准差: {ssim_std:.4f})")
        print(f"   平均MSE:  {avg_mse:.6f}")
        print(f"\n🔬 损失函数详细分解:")
        print(f"   总损失 (Total Loss):     {avg_loss:.6f}  (标准差: {loss_std:.6f})")
        print(f"   ├─ L1损失:              {avg_l1_loss:.6f}")
        print(f"   ├─ 梯度损失 (×0.3):     {avg_grad_loss:.6f}")
        print(f"   ├─ SSIM损失 (×0.2):     {avg_ssim_loss:.6f}")
        print(f"   ├─ 边缘损失 (×0.15):    {avg_edge_loss:.6f}")
        print(f"   └─ 频域损失 (×0.1):     {avg_freq_loss:.6f}")
        print(f"\n⏱️  性能统计:")
        print(f"   测试耗时: {test_time:.1f}秒")
        print(f"   平均每张: {test_time/valid_count:.2f}秒")

        # 性能评价
        print(f"\n💡 性能评价:")
        if avg_psnr >= 30:
            print(f"   PSNR {avg_psnr:.2f}dB - 🌟 优秀!")
        elif avg_psnr >= 28:
            print(f"   PSNR {avg_psnr:.2f}dB - ✅ 良好!")
        elif avg_psnr >= 25:
            print(f"   PSNR {avg_psnr:.2f}dB - ⚠️  一般")
        else:
            print(f"   PSNR {avg_psnr:.2f}dB - ❌ 需要改进")

        if avg_ssim >= 0.95:
            print(f"   SSIM {avg_ssim:.4f} - 🌟 优秀!")
        elif avg_ssim >= 0.90:
            print(f"   SSIM {avg_ssim:.4f} - ✅ 良好!")
        else:
            print(f"   SSIM {avg_ssim:.4f} - ⚠️  需要改进")

        # 🔥 保存增强的测试报告
        report = {
            'model_path': model_path,
            'test_images': valid_count,
            'resolution': '256x256_fair_test',

            # 主要指标
            'metrics': {
                'psnr': {
                    'mean': float(avg_psnr),
                    'std': float(psnr_std),
                    'min': float(psnr_min),
                    'max': float(psnr_max)
                },
                'ssim': {
                    'mean': float(avg_ssim),
                    'std': float(ssim_std),
                    'min': float(ssim_min),
                    'max': float(ssim_max)
                },
                'mse': float(avg_mse)
            },

            # 损失函数分解
            'loss_breakdown': {
                'total_loss': {
                    'mean': float(avg_loss),
                    'std': float(loss_std)
                },
                'l1_loss': float(avg_l1_loss),
                'grad_loss': float(avg_grad_loss),
                'ssim_loss': float(avg_ssim_loss),
                'edge_loss': float(avg_edge_loss),
                'freq_loss': float(avg_freq_loss)
            },

            # 性能统计
            'performance': {
                'test_time_sec': float(test_time),
                'time_per_image_sec': float(test_time/valid_count)
            }
        }

        with open(os.path.join(output_dir, 'test_report_256_fair.json'), 'w') as f:
            json.dump(report, f, indent=4)

        return avg_psnr, avg_ssim
    else:
        print("\n❌ 没有成功处理任何图像!")
        return 0, 0


def main():
    parser = argparse.ArgumentParser(description='公平测试 - 256×256对256×256')
    parser.add_argument('--model_path', type=str,
                       default='checkpoints/mambairv2_gppnn_0925_1327/models/best_model.pth',
                       help='模型checkpoint路径')
    parser.add_argument('--test_dir', type=str,
                       default='photo/testdateset',
                       help='测试数据目录')
    parser.add_argument('--output_dir', type=str,
                       default='test_results_256_fair',
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
    test_fair_256(args.model_path, args.test_dir, args.output_dir, device)


if __name__ == '__main__':
    main()
