#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查公开仓库结构，并可选检查本地私有数据集。

默认模式面向全新 GitHub clone：只检查版本控制中应当存在的源码、
文档和测试，不要求未随仓库发布的 ``photo/`` 数据。需要验证本地
训练数据时，显式传入 ``--with-data``。
"""

import argparse
import os
import sys


class ProjectIntegrityChecker:
    """项目完整性检查器。"""

    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    def __init__(self, project_root="."):
        self.project_root = os.path.abspath(project_root)
        self.errors = []
        self.warnings = []
        self.success_count = 0
        self.total_checks = 0
        self.data_checked = False

    def check_file(self, filepath, required=True):
        """检查文件是否存在。"""
        self.total_checks += 1
        full_path = os.path.join(self.project_root, filepath)

        if os.path.isfile(full_path):
            self.success_count += 1
            return True

        if required:
            self.errors.append(f"[MISSING] Required file: {filepath}")
        else:
            self.warnings.append(f"[OPTIONAL] Missing file: {filepath}")
        return False

    def check_directory(self, dirpath, required=True):
        """检查目录是否存在。"""
        self.total_checks += 1
        full_path = os.path.join(self.project_root, dirpath)

        if os.path.isdir(full_path):
            self.success_count += 1
            return True

        if required:
            self.errors.append(f"[MISSING] Required directory: {dirpath}")
        else:
            self.warnings.append(f"[OPTIONAL] Missing directory: {dirpath}")
        return False

    def check_data_count(self, dirpath, expected_min):
        """检查数据目录中的图像数量。"""
        self.total_checks += 1
        full_path = os.path.join(self.project_root, dirpath)

        if not os.path.isdir(full_path):
            self.errors.append(f"[MISSING] Data directory: {dirpath}")
            return False

        count = sum(
            1
            for filename in os.listdir(full_path)
            if filename.lower().endswith(self.IMAGE_EXTENSIONS)
        )

        if count >= expected_min:
            self.success_count += 1
            return True

        self.errors.append(
            f"[INCOMPLETE] {dirpath}: found {count} images, "
            f"expected >= {expected_min}"
        )
        return False

    def _check_file_group(self, position, total_groups, title, files):
        """检查并显示一组仓库文件。"""
        print(f"[{position}/{total_groups}] {title}...")
        present = 0
        for filepath, required in files:
            if self.check_file(filepath, required):
                present += 1
        print(f"      文件: {present}/{len(files)} 齐全\n")

    def _check_dataset(self, position, total_groups):
        """检查不随 Git 仓库发布的本地训练和测试数据。"""
        print(f"[{position}/{total_groups}] 检查本地数据集...")
        datasets = [
            ("photo/dataset/GT", 600),
            ("photo/dataset/MS", 600),
            ("photo/dataset/PAN", 600),
            ("photo/testdateset/GT", 150),
            ("photo/testdateset/MS", 150),
            ("photo/testdateset/PAN", 150),
        ]

        for dirpath, expected_min in datasets:
            if self.check_directory(dirpath, True):
                self.check_data_count(dirpath, expected_min)

        print("      本地数据集: 检查完成\n")

    def run_checks(self, with_data=False):
        """运行仓库检查；仅在 ``with_data=True`` 时验证 ``photo/``。"""
        self.data_checked = with_data

        print("=" * 70)
        print("MambaIR-inspired GPPNN 项目完整性检查")
        print("=" * 70)
        print()

        groups = [
            (
                "检查项目元数据与许可",
                [
                    ("README.md", True),
                    ("LICENSE", True),
                    ("LICENSES/Apache-2.0.txt", True),
                    ("THIRD_PARTY_NOTICES.md", True),
                    ("CITATION.cff", True),
                    ("requirements.txt", True),
                ],
            ),
            (
                "检查核心训练代码",
                [
                    ("config.py", True),
                    ("train_unified.py", True),
                    ("train.py", True),
                ],
            ),
            (
                "检查模型架构",
                [
                    ("models/__init__.py", True),
                    ("models/mambair_gppnn.py", True),
                    ("models/dual_modal_assm.py", True),
                    ("models/cross_modal_attention.py", True),
                ],
            ),
            (
                "检查可选世界模型模块",
                [
                    ("models/world_model/__init__.py", True),
                    ("models/world_model/wsm.py", True),
                    ("models/world_model/sensor_loss.py", True),
                    ("models/world_model/dca_fim.py", True),
                    ("models/world_model/wacx_loss.py", True),
                    ("models/world_model/patch_refiner.py", True),
                ],
            ),
            (
                "检查数据加载器",
                [
                    ("data/__init__.py", True),
                    ("data/photo_dataloader.py", True),
                ],
            ),
            (
                "检查测试脚本",
                [
                    ("test_256_fair.py", True),
                    ("test_512_fair.py", True),
                    ("tests/test_wsm.py", True),
                    ("tests/test_dsc.py", True),
                    ("tests/test_dca.py", True),
                    ("tests/test_wacx.py", True),
                    ("tests/test_patch_refiner.py", True),
                ],
            ),
            (
                "检查运行脚本",
                [
                    ("run_cloud_train.sh", True),
                    ("run_cloud_test.sh", True),
                    ("inference_with_world_model.py", True),
                    ("quick_test_world_model.py", True),
                ],
            ),
        ]

        total_groups = len(groups) + 1 + int(with_data)
        for position, (title, files) in enumerate(groups, start=1):
            self._check_file_group(position, total_groups, title, files)

        docs_position = len(groups) + 1
        print(f"[{docs_position}/{total_groups}] 检查文档目录...")
        self.check_directory("docs", True)
        print("      文档目录: 检查完成\n")

        if with_data:
            self._check_dataset(docs_position + 1, total_groups)
        else:
            print("      本地数据集: 已跳过（使用 --with-data 可显式检查）\n")

    def print_summary(self):
        """打印检查摘要。"""
        print("=" * 70)
        print("检查结果汇总")
        print("=" * 70)
        print()

        success_rate = (
            self.success_count / self.total_checks * 100
            if self.total_checks > 0
            else 0
        )

        print(f"总检查项: {self.total_checks}")
        print(f"通过项: {self.success_count}")
        print(f"完整率: {success_rate:.1f}%")
        print()

        if self.errors:
            print(f"错误 ({len(self.errors)} 项):")
            for error in self.errors:
                print(f"  {error}")
            print()

        if self.warnings:
            print(f"警告 ({len(self.warnings)} 项):")
            for warning in self.warnings:
                print(f"  {warning}")
            print()

        if not self.errors:
            print("=" * 70)
            print("[SUCCESS] Project integrity check PASSED!")
            print("=" * 70)
            print("[OK] Public repository files are present")
            if self.data_checked:
                print("[OK] Local dataset layout and image counts are valid")
                print("Ready for local training checks.")
            else:
                print("[OK] Private dataset check intentionally skipped")
                print("Clean-clone repository check complete.")
            return True

        print("=" * 70)
        print("[FAIL] Project incomplete, please fix the errors above")
        print("=" * 70)
        return False


def main():
    """命令行入口。"""
    parser = argparse.ArgumentParser(
        description=(
            "检查公开仓库结构；默认不要求未发布的 photo 数据集。"
        )
    )
    parser.add_argument(
        "--with-data",
        action="store_true",
        help="额外检查本地 photo/ 数据目录和最小图像数量",
    )
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    checker = ProjectIntegrityChecker(project_root)
    checker.run_checks(with_data=args.with_data)
    success = checker.print_summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
