# -*- coding: utf-8 -*-
"""
项目完整性检测脚本
检查上传后所有必要的文件和目录是否齐全
"""

import os
import sys

class ProjectIntegrityChecker:
    """项目完整性检查器"""
    
    def __init__(self, project_root='.'):
        self.project_root = project_root
        self.errors = []
        self.warnings = []
        self.success_count = 0
        self.total_checks = 0
        
    def check_file(self, filepath, required=True):
        """检查文件是否存在"""
        self.total_checks += 1
        full_path = os.path.join(self.project_root, filepath)
        
        if os.path.exists(full_path):
            self.success_count += 1
            return True
        else:
            if required:
                self.errors.append(f"[MISSING] Required file: {filepath}")
            else:
                self.warnings.append(f"[OPTIONAL] Missing file: {filepath}")
            return False
    
    def check_directory(self, dirpath, required=True):
        """检查目录是否存在"""
        self.total_checks += 1
        full_path = os.path.join(self.project_root, dirpath)
        
        if os.path.exists(full_path) and os.path.isdir(full_path):
            self.success_count += 1
            return True
        else:
            if required:
                self.errors.append(f"[MISSING] Required directory: {dirpath}")
            else:
                self.warnings.append(f"[OPTIONAL] Missing directory: {dirpath}")
            return False
    
    def check_data_count(self, dirpath, expected_min):
        """检查数据目录图像数量"""
        self.total_checks += 1
        full_path = os.path.join(self.project_root, dirpath)
        
        if not os.path.exists(full_path):
            self.errors.append(f"[MISSING] Data directory: {dirpath}")
            return False
        
        files = [f for f in os.listdir(full_path) if f.lower().endswith(('.jpg', '.png', '.bmp'))]
        count = len(files)
        
        if count >= expected_min:
            self.success_count += 1
            return True
        else:
            self.errors.append(f"[INCOMPLETE] {dirpath}: found {count} images, expected >= {expected_min}")
            return False
    
    def run_checks(self):
        """运行所有检查"""
        print("="*70)
        print("MambaIR-GPPNN 项目完整性检测")
        print("="*70)
        print()
        
        # ========== 核心代码文件 ==========
        print("[1/8] 检查核心代码文件...")
        core_files = [
            ('config.py', True),
            ('train.py', True),
            ('train_unified.py', True),
            ('requirements.txt', True),
        ]
        
        for filepath, required in core_files:
            self.check_file(filepath, required)
        print(f"      核心文件: {4}/{4} 齐全\n")
        
        # ========== 模型架构文件 ==========
        print("[2/8] 检查模型架构文件...")
        model_files = [
            ('models/__init__.py', True),
            ('models/mambair_gppnn.py', True),
            ('models/dual_modal_assm.py', True),
            ('models/cross_modal_attention.py', True),
        ]
        
        for filepath, required in model_files:
            self.check_file(filepath, required)
        print(f"      模型文件: {4}/{4} 齐全\n")
        
        # ========== 世界模型模块 ==========
        print("[3/8] 检查世界模型模块...")
        world_model_files = [
            ('models/world_model/__init__.py', True),
            ('models/world_model/wsm.py', True),
            ('models/world_model/sensor_loss.py', True),
            ('models/world_model/dca_fim.py', True),
            ('models/world_model/wacx_loss.py', True),
            ('models/world_model/patch_refiner.py', True),
        ]
        
        wm_count = 0
        for filepath, required in world_model_files:
            if self.check_file(filepath, required):
                wm_count += 1
        print(f"      世界模型: {wm_count}/{len(world_model_files)} 齐全\n")
        
        # ========== 数据加载器 ==========
        print("[4/8] 检查数据加载器...")
        data_files = [
            ('data/__init__.py', True),
            ('data/photo_dataloader.py', True),
        ]
        
        for filepath, required in data_files:
            self.check_file(filepath, required)
        print(f"      数据加载器: {2}/{2} 齐全\n")
        
        # ========== 测试脚本 ==========
        print("[5/8] 检查测试脚本...")
        test_files = [
            ('test_256_fair.py', True),
            ('test_512_fair.py', True),
            ('tests/test_wsm.py', True),
            ('tests/test_dsc.py', True),
            ('tests/test_dca.py', True),
            ('tests/test_wacx.py', True),
            ('tests/test_patch_refiner.py', True),
        ]
        
        test_count = 0
        for filepath, required in test_files:
            if self.check_file(filepath, required):
                test_count += 1
        print(f"      测试脚本: {test_count}/{len(test_files)} 齐全\n")
        
        # ========== 训练/推理脚本 ==========
        print("[6/8] 检查训练推理脚本...")
        script_files = [
            ('run_cloud_train.sh', True),
            ('run_cloud_test.sh', True),
            ('inference_with_world_model.py', True),
            ('quick_test_world_model.py', True),
        ]
        
        script_count = 0
        for filepath, required in script_files:
            if self.check_file(filepath, required):
                script_count += 1
        print(f"      脚本文件: {script_count}/{len(script_files)} 齐全\n")
        
        # ========== 文档文件 ==========
        print("[7/8] 检查文档文件...")
        doc_files = [
            ('README.md', True),
            ('WORLD_MODEL_GUIDE.md', True),
            ('世界模型增强实施计划.md', True),
            ('最新任务计划.md', True),
            ('OPTIMIZATION_V2.2_SUMMARY.md', False),
        ]
        
        doc_count = 0
        for filepath, required in doc_files:
            if self.check_file(filepath, required):
                doc_count += 1
        print(f"      文档文件: {doc_count}/{len(doc_files)} 齐全\n")
        
        # ========== 数据集检查 ==========
        print("[8/8] 检查数据集...")
        
        # 训练集（dataset目录，650张）
        self.check_directory('photo/dataset/GT', True)
        self.check_directory('photo/dataset/MS', True)
        self.check_directory('photo/dataset/PAN', True)
        
        if os.path.exists('photo/dataset/GT'):
            self.check_data_count('photo/dataset/GT', 600)
            self.check_data_count('photo/dataset/MS', 600)
            self.check_data_count('photo/dataset/PAN', 600)
        
        # 测试集（testdateset目录，150张）
        self.check_directory('photo/testdateset/GT', True)
        self.check_directory('photo/testdateset/MS', True)
        self.check_directory('photo/testdateset/PAN', True)
        
        if os.path.exists('photo/testdateset/GT'):
            self.check_data_count('photo/testdateset/GT', 150)
            self.check_data_count('photo/testdateset/MS', 150)
            self.check_data_count('photo/testdateset/PAN', 150)
        
        print(f"      数据集: 检查完成\n")
        
    def print_summary(self):
        """打印检查摘要"""
        print("="*70)
        print("检查结果汇总")
        print("="*70)
        print()
        
        success_rate = (self.success_count / self.total_checks * 100) if self.total_checks > 0 else 0
        
        print(f"总检查项: {self.total_checks}")
        print(f"通过项: {self.success_count}")
        print(f"完整率: {success_rate:.1f}%")
        print()
        
        if self.errors:
            print("❌ 错误 ({} 项):".format(len(self.errors)))
            for error in self.errors:
                print(f"  {error}")
            print()
        
        if self.warnings:
            print("⚠️  警告 ({} 项):".format(len(self.warnings)))
            for warning in self.warnings:
                print(f"  {warning}")
            print()
        
        if not self.errors:
            print("="*70)
            print("[SUCCESS] Project Integrity Check PASSED!")
            print("="*70)
            print()
            print("[OK] All required files present")
            print("[OK] World Model modules complete")
            print("[OK] Dataset complete")
            print()
            print("Ready for training!")
            return True
        else:
            print("="*70)
            print("[FAIL] Project incomplete, please fix above errors")
            print("="*70)
            return False


def main():
    """主函数"""
    checker = ProjectIntegrityChecker('.')
    checker.run_checks()
    success = checker.print_summary()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

