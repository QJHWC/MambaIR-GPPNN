# -*- coding: utf-8 -*-
"""
世界模型增强模块快速测试
验证所有5大模块的基本功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from models.world_model import (
    WorldStateMemory,
    DeformableCrossAttention,
    SensorConsistencyLoss,
    WACXLoss,
    PatchPriorRefiner
)

def main():
    print("="*70)
    print("世界模型增强模块 - 快速功能测试")
    print("="*70)
    print()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 小显存使用CPU
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_mem < 6:
            print(f"GPU显存仅{gpu_mem:.1f}GB，使用CPU测试\n")
            device = 'cpu'
        else:
            print(f"设备: {device} ({gpu_mem:.1f}GB)\n")
    else:
        print(f"设备: {device}\n")
    
    # 测试参数
    B, C, H, W = 2, 96, 64, 64
    
    try:
        # 测试1: WSM
        print("[1/5] 测试WSM (World State Memory)...")
        feat = torch.randn(B, C, H, W).to(device)
        wsm = WorldStateMemory(C, 128).to(device)
        out, h, g, b = wsm(feat)
        assert out.shape == feat.shape
        print(f"      [OK] 输出形状: {out.shape}, 隐状态: {h.shape}\n")
        
        # 测试2: DCA-FIM
        print("[2/5] 测试DCA-FIM (Deformable Cross-Attention)...")
        dca = DeformableCrossAttention(C, 4).to(device)
        aligned = dca(feat, feat)
        assert aligned.shape == feat.shape
        print(f"      [OK] 输出形状: {aligned.shape}\n")
        
        # 测试3: DSC
        print("[3/5] 测试DSC (Sensor Consistency Loss)...")
        hrms = torch.randn(B, 3, H, W).to(device)
        pan = torch.randn(B, 1, H, W).to(device)
        dsc = SensorConsistencyLoss().to(device)
        loss_dict = dsc(hrms, pan)
        assert 'dsc_total' in loss_dict
        print(f"      [OK] DSC损失: {loss_dict['dsc_total'].item():.6f}\n")
        
        # 测试4: WAC-X
        print("[4/5] 测试WAC-X (Cross-band Consistency)...")
        wacx = WACXLoss().to(device)
        loss_dict = wacx(hrms, pan)
        assert 'wacx_total' in loss_dict
        print(f"      [OK] WAC-X损失: {loss_dict['wacx_total'].item():.6f}\n")
        
        # 测试5: Patch Prior
        print("[5/5] 测试Patch Prior Refiner...")
        refiner = PatchPriorRefiner(patch_size=32)
        refined = refiner.refine(hrms)
        assert refined.shape == hrms.shape
        print(f"      [OK] 输出形状: {refined.shape}\n")
        
        print("="*70)
        print("[SUCCESS] All World Model modules working properly!")
        print("="*70)
        print()
        print("[OK] Ready for training! Usage examples:")
        print()
        print("  # Core modules (recommended)")
        print("  python train.py --model_size base --img_size 256 \\")
        print("    --enable_world_model --use_wsm --use_dsc")
        print()
        print("  # Full modules (best performance)")
        print("  python train.py --model_size base --img_size 256 \\")
        print("    --enable_world_model --use_wsm --use_dca_fim --use_dsc --use_wacx")
        print()
        
        return 0
        
    except Exception as e:
        print()
        print("="*70)
        print(f"[FAIL] 测试失败: {e}")
        print("="*70)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

