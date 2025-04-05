# 弱监督图像分割项目

## 项目概述

本项目实现了基于Class Activation Map (CAM)的弱监督语义分割方法，以及与全监督方法的对比。使用Oxford-IIIT Pet数据集进行实验，通过CAM生成伪标签，再训练分割模型，最终评估模型性能。

## 环境配置

### 创建并激活Conda环境

```bash
conda create -n comp0197-cw1-pt python=3.12 pip
conda activate comp0197-cw1-pt
```

### 安装PyTorch

```bash
pip install torch==2.5.0 torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 安装其他依赖

```bash
pip install -r requirements.txt
```

requirements.txt包含以下依赖:

- scikit-learn==1.3.0
- pydensecrf (GitHub版本)

## 项目结构

```
DL
│
├── data/                  ← Oxford-IIIT Pet 原始数据
│
├── models/
│   ├── classifier/        ← CAM 分类模型
│   └── segmentor/         ← 分割模型（弱监督 + 全监督）
│       └── fully_supervised/ ← 分割模型（全监督）
│   
├── outputs/
│   ├── cams/              ← 第一步分类模型生成的 CAM 热力图 （cams.zip）
│   ├── base_pseudo/       ← CAM 二值化伪标签 (base_pseudo.zip)
│   ├── pseudo_mask/       ← CRF 后处理伪标签 (pseudo_mask.zip)
│   └── result/
│       ├── weak/          ← 弱监督模型输出 (mIoU）
│       └── full/          ← 全监督模型输出结果（mIoU）
│
└── src/
    ├── cam_extractor.py        ← CAM 分类器训练与提取     (Stage 1)
    ├── m2_base_masks.py        ← 热力图直接生成伪标签      (Stage 2)
    ├── m2_cam_crf.py           ← 热力图CRF处理生成伪标签   (Stage 2)
    ├── m3_train_segmentation.py ← 弱监督模型训练          (Stage 3)
    ├── m3_train_fully_supervised.py ← 完全监督训练        (Stage 4)
    ├── config.py               ← 相关参数
    └── utils.py                ← 相关函数
```

## 数据下载

数据下载：[点击这里](https://drive.google.com/file/d/1vZ1n-iBSJiimCQ11--Df8sWCCpY7bbey/view?usp=sharing)

## 工作流程

1. **分类模型训练与CAM提取**：使用 `cam_extractor.py`训练分类模型并提取CAM热力图。
2. **生成伪标签**：
   - 使用 `m2_base_masks.py`直接从热力图生成二值化伪标签
   - 使用 `m2_cam_crf.py`通过CRF后处理生成更精确的伪标签
3. **分割模型训练**：
   - 使用 `m3_train_segmentation.py`基于伪标签训练弱监督分割模型
   - 使用 `m3_train_fully_supervised.py`训练全监督分割模型作为对比
4. **评估**：对比弱监督和全监督模型的分割性能（mIoU）

## 使用方法

1. 准备数据集，放置在 `data/`目录下
2. 执行CAM提取和伪标签生成
3. 训练分割模型
4. 评估模型性能
