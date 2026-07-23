# -*- coding: utf-8 -*-
"""
Photo目录数据加载器
适配用户的photo目录结构：
- photo/dataset/ (650张训练图像)
- photo/testdateset/ (150张测试图像)

参考官方Testing规范: https://github.com/csguoh/MambaIR?tab=readme-ov-file#installation
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from PIL import Image


class PhotoDataset(Dataset):
    """
    Photo目录数据集类
    支持MS+PAN+GT三模态数据加载
    """
    def __init__(self,
                 photo_root,
                 mode='train',        # 'train', 'val', 'test'
                 data_split='dataset', # 'dataset' or 'testdateset'
                 transform=None,
                 img_size=None):
        """
        Args:
            photo_root: photo目录路径 (如: ./photo)
            mode: 模式 ('train', 'val', 'test')
            data_split: 数据分割 ('dataset'=训练集, 'testdateset'=测试集)
            transform: 数据变换
            img_size: 图像尺寸，None表示保持原尺寸
        """
        self.photo_root = photo_root
        self.mode = mode
        self.data_split = data_split
        self.transform = transform
        self.img_size = img_size

        # 构建数据路径
        self.data_dir = os.path.join(photo_root, data_split)
        self.gt_dir = os.path.join(self.data_dir, 'GT')
        self.ms_dir = os.path.join(self.data_dir, 'MS')
        self.pan_dir = os.path.join(self.data_dir, 'PAN')

        # 检查目录存在性
        for dir_path in [self.gt_dir, self.ms_dir, self.pan_dir]:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Directory not found: {dir_path}")

        # 获取图像文件列表
        self.img_names = self._get_image_list()

        # 数据分割 (仅对dataset目录)
        if data_split == 'dataset':
            self.img_names = self._split_dataset()

        print(f"[Data] {self.__class__.__name__} initialized:")
        print(f"   Mode: {mode}")
        print(f"   Split: {data_split}")
        print(f"   Images: {len(self.img_names)}")
        print(f"   Directory: {self.data_dir}")

    def _get_image_list(self):
        """获取图像文件列表"""
        gt_files = set(os.listdir(self.gt_dir))
        ms_files = set(os.listdir(self.ms_dir))
        pan_files = set(os.listdir(self.pan_dir))

        # 找到三个目录的交集
        common_files = gt_files & ms_files & pan_files

        # 排序确保一致性
        img_names = sorted([f for f in common_files if f.lower().endswith(('.jpg', '.png', '.bmp'))])

        if len(img_names) == 0:
            raise ValueError(f"No common images found in {self.data_dir}")

        return img_names

    def _split_dataset(self):
        """分割dataset目录为训练集和验证集"""
        # 参考原GPPNN分割策略：600训练 + 50验证
        total_imgs = len(self.img_names)

        if self.mode == 'train':
            # 前600张作为训练集
            return self.img_names[:600] if total_imgs >= 600 else self.img_names[:int(0.9*total_imgs)]
        elif self.mode == 'val':
            # 后50张作为验证集
            return self.img_names[600:650] if total_imgs >= 650 else self.img_names[int(0.9*total_imgs):]
        else:
            # 测试模式返回全部
            return self.img_names

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        """获取单个样本"""
        img_name = self.img_names[idx]

        # 加载图像
        gt_path = os.path.join(self.gt_dir, img_name)
        ms_path = os.path.join(self.ms_dir, img_name)
        pan_path = os.path.join(self.pan_dir, img_name)

        try:
            # 使用OpenCV加载图像 (BGR格式)
            gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR)
            ms_img = cv2.imread(ms_path, cv2.IMREAD_COLOR)
            pan_img = cv2.imread(pan_path, cv2.IMREAD_GRAYSCALE)

            if gt_img is None or ms_img is None or pan_img is None:
                raise ValueError(f"Failed to load images for {img_name}")

            # 转换为RGB格式
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
            ms_img = cv2.cvtColor(ms_img, cv2.COLOR_BGR2RGB)

            # 确保尺寸一致
            if self.img_size is not None:
                h, w = self.img_size
                gt_img = cv2.resize(gt_img, (w, h))
                ms_img = cv2.resize(ms_img, (w, h))
                pan_img = cv2.resize(pan_img, (w, h))

            # 归一化到[0,1] (v2.2: 添加显式clip确保一致性)
            gt_img = np.clip(gt_img.astype(np.float32) / 255.0, 0.0, 1.0)
            ms_img = np.clip(ms_img.astype(np.float32) / 255.0, 0.0, 1.0)
            pan_img = np.clip(pan_img.astype(np.float32) / 255.0, 0.0, 1.0)

            # 转换为torch tensor
            gt_tensor = torch.from_numpy(gt_img.transpose(2, 0, 1))  # [3, H, W]
            ms_tensor = torch.from_numpy(ms_img.transpose(2, 0, 1))  # [3, H, W]
            pan_tensor = torch.from_numpy(pan_img).unsqueeze(0)      # [1, H, W]

            # 🔥 v2.2: 应用配对几何变换 (翻转+旋转)
            if self.mode == 'train':
                # 几何变换: 翻转和旋转 (训练时应用)
                if random.random() < 0.5:  # 水平翻转
                    gt_tensor = torch.flip(gt_tensor, dims=[2])
                    ms_tensor = torch.flip(ms_tensor, dims=[2])
                    pan_tensor = torch.flip(pan_tensor, dims=[2])

                if random.random() < 0.5:  # 垂直翻转
                    gt_tensor = torch.flip(gt_tensor, dims=[1])
                    ms_tensor = torch.flip(ms_tensor, dims=[1])
                    pan_tensor = torch.flip(pan_tensor, dims=[1])

                if random.random() < 0.5:  # 随机旋转90°倍数
                    k = random.choice([1, 2, 3])
                    gt_tensor = torch.rot90(gt_tensor, k, dims=[1, 2])
                    ms_tensor = torch.rot90(ms_tensor, k, dims=[1, 2])
                    pan_tensor = torch.rot90(pan_tensor, k, dims=[1, 2])

            # 应用颜色变换 (仅GT和MS，不对PAN)
            if self.transform is not None and self.mode == 'train':
                gt_tensor = self.transform(gt_tensor)
                ms_tensor = self.transform(ms_tensor)

            return ms_tensor, pan_tensor, gt_tensor

        except Exception as e:
            print(f"❌ Error loading {img_name}: {e}")
            # 返回随机数据作为fallback
            h, w = (256, 256) if self.img_size is None else self.img_size
            return (torch.randn(3, h, w),
                    torch.randn(1, h, w),
                    torch.randn(3, h, w))


class PairedRandomTransform:
    """
    🔥 v2.2新增: 配对数据增强 - 对MS/PAN/GT同步应用相同的几何变换
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, ms, pan, gt):
        # RandomHorizontalFlip
        if random.random() < self.p:
            ms = torch.flip(ms, dims=[2])
            pan = torch.flip(pan, dims=[2])
            gt = torch.flip(gt, dims=[2])

        # RandomVerticalFlip
        if random.random() < self.p:
            ms = torch.flip(ms, dims=[1])
            pan = torch.flip(pan, dims=[1])
            gt = torch.flip(gt, dims=[1])

        # Random90°Rotation
        if random.random() < self.p:
            k = random.choice([1, 2, 3])  # 90°, 180°, 270°
            ms = torch.rot90(ms, k, dims=[1, 2])
            pan = torch.rot90(pan, k, dims=[1, 2])
            gt = torch.rot90(gt, k, dims=[1, 2])

        return ms, pan, gt


def create_photo_dataloaders(photo_root='./photo',
                           batch_size=4,
                           num_workers=2,
                           img_size=None,
                           train_transform=None,
                           test_transform=None,
                           use_augmentation=True):
    """
    创建photo目录的数据加载器 (v2.2优化版)

    Args:
        photo_root: photo目录路径
        batch_size: 批次大小
        num_workers: 数据加载线程数
        img_size: 图像尺寸 (H, W) 或 None
        train_transform: 训练数据变换
        test_transform: 测试数据变换
        use_augmentation: 是否使用数据增强 (v2.2)

    Returns:
        train_loader, val_loader, test_loader
    """

    # 🔥 v2.2: 增强的数据变换
    if train_transform is None and use_augmentation:
        train_transform = transforms.Compose([
            # 轻微颜色抖动 (适用于全色锐化任务)
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05)
        ])
    elif train_transform is None:
        train_transform = transforms.Compose([])

    if test_transform is None:
        test_transform = transforms.Compose([])

    # 创建数据集
    train_dataset = PhotoDataset(
        photo_root=photo_root,
        mode='train',
        data_split='dataset',
        transform=train_transform,
        img_size=img_size
    )

    val_dataset = PhotoDataset(
        photo_root=photo_root,
        mode='val',
        data_split='dataset',
        transform=test_transform,
        img_size=img_size
    )

    test_dataset = PhotoDataset(
        photo_root=photo_root,
        mode='test',
        data_split='testdateset',
        transform=test_transform,
        img_size=img_size
    )

    # 创建数据加载器
    # 🔧 安全的DataLoader配置 - 避免多进程问题
    safe_pin_memory = True if num_workers == 0 else False  # 多进程时禁用pin_memory

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=safe_pin_memory,
        drop_last=True,
        persistent_workers=False,  # 避免进程泄漏
        prefetch_factor=2 if num_workers > 0 else None  # 控制预取
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=safe_pin_memory,
        drop_last=False,
        persistent_workers=False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=safe_pin_memory,
        drop_last=False,
        persistent_workers=False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    print(f"\n[Data] DataLoaders created:")
    print(f"   Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"   Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"   Test: {len(test_dataset)} samples, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


def test_photo_dataloader():
    """测试photo数据加载器"""
    print("🧪 测试Photo数据加载器...")

    try:
        # 创建数据加载器
        train_loader, val_loader, test_loader = create_photo_dataloaders(
            photo_root='../photo',  # 相对于MambaIR-GPPNN目录
            batch_size=2,
            num_workers=0,  # 测试时使用0避免多进程问题
            img_size=(256, 256)
        )

        # 测试训练数据加载
        print("\n📦 测试训练数据加载:")
        for i, (ms, pan, gt) in enumerate(train_loader):
            print(f"   Batch {i}: MS{ms.shape}, PAN{pan.shape}, GT{gt.shape}")
            print(f"   数据范围: MS[{ms.min():.3f}, {ms.max():.3f}], PAN[{pan.min():.3f}, {pan.max():.3f}]")
            if i >= 2:  # 只测试前几个batch
                break

        # 测试测试数据加载
        print("\n📦 测试测试数据加载:")
        for i, (ms, pan, gt) in enumerate(test_loader):
            print(f"   Batch {i}: MS{ms.shape}, PAN{pan.shape}, GT{gt.shape}")
            if i >= 1:
                break

        print("\n✅ Photo数据加载器测试通过!")

    except Exception as e:
        print(f"\n❌ 数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_photo_dataloader()
