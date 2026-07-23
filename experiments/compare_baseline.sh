#!/bin/bash
# ============================================================================
# 世界模型增强 vs Baseline 对比实验脚本
# ============================================================================
#
# 使用方法:
#   chmod +x experiments/compare_baseline.sh
#   ./experiments/compare_baseline.sh
#
# 实验配置:
#   - 模型: Base
#   - 分辨率: 256×256
#   - 训练轮数: 50 (快速验证)
#   - 配置: Baseline, WSM, DSC, WSM+DSC, Full
# ============================================================================

echo "======================================================================"
echo "世界模型增强 - 消融实验"
echo "======================================================================"
echo ""

# 实验参数
MODEL_SIZE="base"
IMG_SIZE="256"
EPOCHS="50"
BATCH_SIZE="8"

# 创建实验目录
mkdir -p experiments/results

# ========== 实验1: Baseline ==========
echo "[实验1/5] Baseline (无世界模型)"
echo "----------------------------------------------------------------------"
python train.py --model_size $MODEL_SIZE --img_size $IMG_SIZE \
  --epochs $EPOCHS --batch_size $BATCH_SIZE \
  --save_dir experiments/checkpoints/exp1_baseline \
  --log_dir experiments/logs/exp1_baseline

# ========== 实验2: WSM Only ==========
echo ""
echo "[实验2/5] WSM Only (仅世界状态记忆)"
echo "----------------------------------------------------------------------"
python train.py --model_size $MODEL_SIZE --img_size $IMG_SIZE \
  --epochs $EPOCHS --batch_size $BATCH_SIZE \
  --enable_world_model --use_wsm \
  --save_dir experiments/checkpoints/exp2_wsm \
  --log_dir experiments/logs/exp2_wsm

# ========== 实验3: DSC Only ==========
echo ""
echo "[实验3/5] DSC Only (仅物理一致性)"
echo "----------------------------------------------------------------------"
python train.py --model_size $MODEL_SIZE --img_size $IMG_SIZE \
  --epochs $EPOCHS --batch_size $BATCH_SIZE \
  --enable_world_model --use_dsc \
  --save_dir experiments/checkpoints/exp3_dsc \
  --log_dir experiments/logs/exp3_dsc

# ========== 实验4: WSM+DSC ==========
echo ""
echo "[实验4/5] WSM+DSC (核心功能)"
echo "----------------------------------------------------------------------"
python train.py --model_size $MODEL_SIZE --img_size $IMG_SIZE \
  --epochs $EPOCHS --batch_size $BATCH_SIZE \
  --enable_world_model --use_wsm --use_dsc \
  --save_dir experiments/checkpoints/exp4_wsm_dsc \
  --log_dir experiments/logs/exp4_wsm_dsc

# ========== 实验5: Full ==========
echo ""
echo "[实验5/5] Full (完整世界模型)"
echo "----------------------------------------------------------------------"
python train.py --model_size $MODEL_SIZE --img_size $IMG_SIZE \
  --epochs $EPOCHS --batch_size $BATCH_SIZE \
  --enable_world_model --use_wsm --use_dca_fim --use_dsc --use_wacx \
  --save_dir experiments/checkpoints/exp5_full \
  --log_dir experiments/logs/exp5_full

echo ""
echo "======================================================================"
echo "所有实验完成！"
echo "======================================================================"
echo ""
echo "结果位置:"
echo "  checkpoints: experiments/checkpoints/"
echo "  logs: experiments/logs/"
echo ""
echo "查看TensorBoard:"
echo "  tensorboard --logdir experiments/logs"
echo ""

