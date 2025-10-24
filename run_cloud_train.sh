#!/bin/bash
# ============================================================================
# 🚀 MambaIRv2-GPPNN 云端训练快速启动脚本
# ============================================================================
#
# 使用方法:
#   ./run_cloud_train.sh --model base --size 256
#   ./run_cloud_train.sh --model base --size 512
#   ./run_cloud_train.sh --model large --size 256
#   ./run_cloud_train.sh --model large --size 512
#
# 可选参数:
#   --epochs N          训练轮数 (默认自动)
#   --batch_size N      批次大小 (默认自动)
#   --lr FLOAT          学习率 (默认自动)
#   --resume PATH       断点续训路径
#   --auto_resume       自动断点续训
#   --auto_batch_size   🔥 自动查找最大batch_size (推荐!)
# ============================================================================

# 默认参数
MODEL_SIZE="base"
IMG_SIZE="256"
EPOCHS=""
BATCH_SIZE=""
LR=""
RESUME=""
AUTO_RESUME=""
AUTO_BATCH_SIZE=""
EXTRA_ARGS=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --size)
            IMG_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="--epochs $2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="--batch_size $2"
            shift 2
            ;;
        --lr)
            LR="--lr $2"
            shift 2
            ;;
        --resume)
            RESUME="--resume $2"
            shift 2
            ;;
        --auto_resume)
            AUTO_RESUME="--auto_resume"
            shift
            ;;
        --auto_batch_size)
            AUTO_BATCH_SIZE="--auto_batch_size"
            shift
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# 验证参数
if [[ ! "$MODEL_SIZE" =~ ^(base|large)$ ]]; then
    echo "❌ 错误: --model 必须是 base 或 large"
    exit 1
fi

if [[ ! "$IMG_SIZE" =~ ^(256|512)$ ]]; then
    echo "❌ 错误: --size 必须是 256 或 512"
    exit 1
fi

# 打印配置
echo "============================================================================"
echo "🚀 MambaIRv2-GPPNN 云端训练启动"
echo "============================================================================"
echo "📋 训练配置:"
echo "   模型大小: $MODEL_SIZE"
echo "   图像尺寸: ${IMG_SIZE}×${IMG_SIZE}"
[ -n "$EPOCHS" ] && echo "   训练轮数: $(echo $EPOCHS | cut -d' ' -f2)"
[ -n "$BATCH_SIZE" ] && echo "   批次大小: $(echo $BATCH_SIZE | cut -d' ' -f2)"
[ -n "$LR" ] && echo "   学习率: $(echo $LR | cut -d' ' -f2)"
[ -n "$RESUME" ] && echo "   断点续训: $(echo $RESUME | cut -d' ' -f2)"
[ -n "$AUTO_RESUME" ] && echo "   自动续训: 启用"
echo "============================================================================"

# 检查CUDA可用性
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "🎮 GPU信息:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo "============================================================================"
fi

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ 错误: 未找到Python环境"
    exit 1
fi

# 检查必要的文件
if [ ! -f "train_unified.py" ]; then
    echo "❌ 错误: 未找到 train_unified.py"
    exit 1
fi

if [ ! -d "photo" ]; then
    echo "⚠️  警告: 未找到 photo 目录，请确保数据集已准备好"
fi

# 构建训练命令
CMD="python train_unified.py \
    --model_size $MODEL_SIZE \
    --img_size $IMG_SIZE \
    $EPOCHS \
    $BATCH_SIZE \
    $LR \
    $RESUME \
    $AUTO_RESUME \
    $AUTO_BATCH_SIZE \
    $EXTRA_ARGS"

echo ""
echo "🚀 启动训练命令:"
echo "   $CMD"
echo "============================================================================"
echo ""

# 创建日志目录
mkdir -p logs
mkdir -p checkpoints

# 执行训练（带日志记录）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/train_${MODEL_SIZE}_${IMG_SIZE}_${TIMESTAMP}.log"

echo "📝 训练日志将保存到: $LOG_FILE"
echo ""
echo "⏳ 3秒后开始训练... (Ctrl+C 取消)"
sleep 3

# 使用tee同时输出到终端和日志文件
eval $CMD 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "============================================================================"
    echo "✅ 训练成功完成!"
    echo "============================================================================"
else
    echo ""
    echo "============================================================================"
    echo "❌ 训练异常退出 (退出码: $EXIT_CODE)"
    echo "============================================================================"
fi

exit $EXIT_CODE
