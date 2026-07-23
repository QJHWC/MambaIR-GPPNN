#!/bin/bash
# ============================================================================
# 🧪 MambaIRv2-GPPNN 云端测试快速启动脚本
# ============================================================================
#
# 使用方法:
#   ./run_cloud_test.sh --model base --size 256 --checkpoint checkpoints/.../best_model.pth
#   ./run_cloud_test.sh --model base --size 512 --checkpoint checkpoints/.../best_model.pth
#   ./run_cloud_test.sh --model large --size 256 --checkpoint checkpoints/.../best_model.pth
#   ./run_cloud_test.sh --model large --size 512 --checkpoint checkpoints/.../best_model.pth
#
# 可选参数:
#   --test_dir PATH     测试数据目录 (默认: photo/testdateset)
#   --output_dir PATH   输出目录 (默认: 自动生成)
#   --device cuda/cpu   指定设备 (默认: auto)
# ============================================================================

# 默认参数
MODEL_SIZE="base"
IMG_SIZE="256"
CHECKPOINT=""
TEST_DIR="photo/testdateset"
OUTPUT_DIR=""
DEVICE="auto"
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
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --test_dir)
            TEST_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
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

if [ -z "$CHECKPOINT" ]; then
    echo "❌ 错误: 必须指定 --checkpoint 参数"
    echo ""
    echo "💡 提示: 使用以下命令查找可用的checkpoint:"
    echo "   find checkpoints -name 'best_model.pth'"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ 错误: checkpoint文件不存在: $CHECKPOINT"
    exit 1
fi

# 自动生成输出目录
if [ -z "$OUTPUT_DIR" ]; then
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    OUTPUT_DIR="test_results_${MODEL_SIZE}_${IMG_SIZE}_${TIMESTAMP}"
fi

# 选择正确的测试脚本
if [ "$IMG_SIZE" == "256" ]; then
    TEST_SCRIPT="test_256_fair.py"
else
    TEST_SCRIPT="test_512_fair.py"
fi

# 打印配置
echo "============================================================================"
echo "🧪 MambaIRv2-GPPNN 云端测试启动"
echo "============================================================================"
echo "📋 测试配置:"
echo "   模型大小: $MODEL_SIZE"
echo "   图像尺寸: ${IMG_SIZE}×${IMG_SIZE}"
echo "   测试脚本: $TEST_SCRIPT"
echo "   Checkpoint: $CHECKPOINT"
echo "   测试目录: $TEST_DIR"
echo "   输出目录: $OUTPUT_DIR"
echo "   设备: $DEVICE"
echo "============================================================================"

# 检查CUDA可用性
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "🎮 GPU信息:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo "============================================================================"
fi

# 检查必要的文件
if [ ! -f "$TEST_SCRIPT" ]; then
    echo "❌ 错误: 未找到测试脚本 $TEST_SCRIPT"
    exit 1
fi

if [ ! -d "$TEST_DIR" ]; then
    echo "❌ 错误: 测试目录不存在: $TEST_DIR"
    exit 1
fi

# 检查测试目录结构
if [ ! -d "$TEST_DIR/GT" ] || [ ! -d "$TEST_DIR/MS" ] || [ ! -d "$TEST_DIR/PAN" ]; then
    echo "❌ 错误: 测试目录结构不完整，需要包含 GT/, MS/, PAN/ 子目录"
    exit 1
fi

# 统计测试图像数量
GT_COUNT=$(ls -1 "$TEST_DIR/GT" 2>/dev/null | wc -l)
echo ""
echo "📊 测试数据统计:"
echo "   GT图像数: $GT_COUNT"
echo "============================================================================"

# 构建测试命令
CMD="python $TEST_SCRIPT \
    --model_path $CHECKPOINT \
    --test_dir $TEST_DIR \
    --output_dir $OUTPUT_DIR \
    --device $DEVICE \
    $EXTRA_ARGS"

echo ""
echo "🚀 启动测试命令:"
echo "   $CMD"
echo "============================================================================"
echo ""
echo "⏳ 3秒后开始测试... (Ctrl+C 取消)"
sleep 3

# 执行测试（带日志记录）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/test_${MODEL_SIZE}_${IMG_SIZE}_${TIMESTAMP}.log"
mkdir -p logs

echo "📝 测试日志将保存到: $LOG_FILE"
echo ""

# 使用tee同时输出到终端和日志文件
eval $CMD 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "============================================================================"
    echo "✅ 测试成功完成!"
    echo "============================================================================"
    echo "📊 结果保存在: $OUTPUT_DIR"
    echo "📝 测试日志: $LOG_FILE"

    # 显示测试报告
    if [ -f "$OUTPUT_DIR/test_report_${IMG_SIZE}_fair.json" ]; then
        echo ""
        echo "📈 测试结果概览:"
        cat "$OUTPUT_DIR/test_report_${IMG_SIZE}_fair.json" | python -m json.tool | grep -E '(avg_psnr|avg_ssim|test_images)'
    fi

    echo "============================================================================"
else
    echo ""
    echo "============================================================================"
    echo "❌ 测试异常退出 (退出码: $EXIT_CODE)"
    echo "============================================================================"
fi

exit $EXIT_CODE
