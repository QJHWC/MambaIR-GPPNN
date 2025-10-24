#!/usr/bin/env python3
"""准备上传GitHub - 清理临时文件"""
import os
import shutil
import glob

print("="*70)
print("准备上传GitHub - 清理临时文件")
print("="*70)
print()

# 需要删除的临时测试文件
temp_files = [
    'diagnose_error.py',
    'test_fp16_effect.py',
    'test_fp16_real.py',
    'test_memory_killer.py',
    'test_512_training.py',
    'test_training_step_by_step.py',
    'verify_fixes.py',
    'final_test_before_training.py',
    'compare_memory_usage.py',
    'check_actual_image_size.py',
    'check_env.py',
    'test_wsm_integration.py',
    'test_config_phase1.py',
    'TRAIN_COMPARISON.md',
]

deleted_count = 0
for file in temp_files:
    if os.path.exists(file):
        os.remove(file)
        print(f"✅ 删除: {file}")
        deleted_count += 1

print(f"\n删除临时文件: {deleted_count}个")
print()

# 删除Python缓存
print("清理Python缓存...")
cache_dirs = glob.glob('**/__pycache__', recursive=True)
for cache_dir in cache_dirs:
    shutil.rmtree(cache_dir)
    print(f"✅ 删除: {cache_dir}")

pyc_files = glob.glob('**/*.pyc', recursive=True)
for pyc_file in pyc_files:
    os.remove(pyc_file)

print(f"清理缓存目录: {len(cache_dirs)}个")
print()

# 创建.gitignore
print("创建.gitignore...")
gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# 训练产物
checkpoints/
checkpoints_safe/
checkpoints_optimized/
logs/
results/
*.pth
*.pt

# 数据集（大文件不上传）
photo/dataset/
photo/testdateset/

# 实验临时文件
experiments/checkpoints/
experiments/logs/
experiments/results/

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Jupyter
.ipynb_checkpoints/

# TensorBoard
events.out.tfevents.*

# 临时文件
*.tmp
*.log
*.bak
"""

with open('.gitignore', 'w', encoding='utf-8') as f:
    f.write(gitignore_content)

print("✅ .gitignore已创建")
print()

print("="*70)
print("清理完成！现在可以上传GitHub")
print("="*70)

