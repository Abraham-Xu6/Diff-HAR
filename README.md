# Diff-HAR: Generative Feature Augmentation for Generalized Zero-Shot Human Activity Recognition

> 本项目提出了一种先进的 **生成式特征增强 (Generative Feature Augmentation)** 框架，专为解决多模态传感器（IMU, mmWave）在人类活动识别 (HAR) 领域的**广义零样本学习 (GZSL)** 问题而设计。
>
> *Targeted for submission in IEEE Internet of Things Journal (IoTJ).*

------

## 🌟 核心创新点 (Key Contributions)

1. **多维统计特征算子:** 针对稀疏毫米波 (mmWave) 数据，创新性地设计了包含重心、标准差与极差的多维联合特征算子，大幅缓解了动作空间信息丢失问题。
2. **CFG 语义引导扩散生成:** 引入 **Classifier-Free Guidance (CFG)** 与自适应噪声注入策略，能够仅凭借文本语义描述，生成高质量的 Unseen 类别虚拟特征。
3. **物理惯性约束与非线性决策校准:** 采用 SVD 能量恢复策略与非线性 MLP 分类器，配合自动化的 Gamma 阈值搜索算法，有效消除了 GZSL 中普遍存在的“已知类偏见 (Seen-Class Bias)”。

------

## 🏆 SOTA 性能表现 (Performance Benchmarks)

经过在三个极具挑战性的公开数据集上的严格测试，本系统在 GZSL 的核心指标 **H-mean (调和平均准确率)** 上全面超越了最新的基准方法。

| **数据集 (Dataset)** | **模态 (Modality)** | **H-mean (Zero-Shot)** | **Unseen Accuracy** | **状态 (Status)**   |
| -------------------- | ------------------- | ---------------------- | ------------------- | ------------------- |
| **PAMAP2**           | IMU                 | **72.86%**             | **95.37%**          | **SOTA 🚀 (+10.7%)** |
| **USC-HAD**          | IMU                 | **57.65%**             | **57.49%**          | **SOTA 🏆**          |
| **MMFi (mmWave)**    | Radar               | **53.88%**             | **46.49%**          | **SOTA 🏆 (+2.1%)**  |

------

## 🛠️ 安装与环境配置 (Installation)

推荐使用 Python 3.8 及以上版本。硬件测试环境配置为 AMD Ryzen 5 5600 搭配 CUDA 兼容 GPU。

```
# 1. 克隆仓库
git clone https://github.com/Abraham Xu/Diff-HAR.git
cd Diff-HAR

# 2. 创建并激活虚拟环境 (推荐使用 conda 或 venv)
conda create -n diff_har python=3.9
conda activate diff_har

# 3. 安装核心依赖
pip install -r requirements.txt
```

------

## 📂 数据集准备 (Data Preparation)

1. 请自行下载对应的原始数据集：PAMAP2, USC-HAD, MMFi。
2. 建议在项目根目录创建 `dataset/` 文件夹并解压数据。
3. 打开 `configs.py`，根据你的实际路径修改 `raw_data_dir`。

------

## 🚀 快速开始 (Quick Start)

我们提供了一个一键式的总控脚本 `main.py`，高度自动化了从数据预处理到最终模型评估的全流水线。

### 运行完整 Zero-Shot 训练与评估流程：

```
# 示例：运行 PAMAP2 的全流程 (步骤 1 到 7)
python main.py --config_choose PAMAP2 --steps 1-7 --mlp_hidden_dim 128 --guidance_scale 3.0
```

**流水线详细解析 (Pipeline Steps):**

- **Step 1:** 数据预处理与滑动窗口切片，执行严格的 Seen/Unseen ZSL 划分。
- **Step 2-3:** 训练骨干特征提取网络，提取并保存真实 Seen 类别的特征。
- **Step 4-5:** 训练文本引导的特征扩散模型，并针对 Unseen 类别合成带有物理惯性约束的虚拟特征。
- **Step 6:** 联合真实特征与虚拟特征，训练非线性 MLP 最终分类器。
- **Step 7:** 执行 GZSL 评估，自动进行 Gamma 阈值扫描以定位最佳 H-mean，并输出校准曲线数据。

------

## 📊 质量评估与实验分析 (Evaluation & Analysis)

除了核心流水线，项目还提供了独立的脚本用于深入评估特征质量和模型鲁棒性。

### 1. 特征质量评估 (Feature Quality)

计算真实 Unseen 特征与生成 Unseen 特征之间的最大均值差异 (MMD) 和 KL 散度，并绘制特征分布直方图。

```
python eval_feature_quality.py --preprocessed_data_dir ./data_preprocessed/PAMAP2
```

*产出：`exp_feature_quality_dist.png` 与 `exp_feature_quality_metrics.csv`*

### 2. 传感器噪声鲁棒性实验 (Robustness Plotting)

绘制符合 IEEE IoTJ 期刊标准的高质量鲁棒性折线图，展示模型在不同 $\sigma$ 漂移噪声下的稳定性。

```
python exp_robustness_noise.py
```

*产出：`exp_robustness_noise_iotj.pdf` 与 `.png` 预览图*

------

## 📁 目录结构 (Directory Structure)

```
Diff-HAR/
├── configs.py                    # 数据集路径与文本 Prompt 映射配置
├── main.py                       # 核心流程控制器
├── data_utils.py                 # 多模态信号预处理引擎 (含 mmWave 特征工程)
├── classifier_model.py           # 骨干网络 (CNN/Transformer)
├── final_classifier_model.py     # 非线性决策网络架构
├── train_feature_extractor.py    # 基础特征提取器训练逻辑
├── extract_real_features.py      # 真实特征提取脚本
├── train_feature_diffusion.py    # 扩散模型训练逻辑
├── generate_virtual_features.py  # 虚拟特征合成与能量约束
├── train_final_classifier.py     # MLP 分类器训练
├── evaluate_gzsl.py              # 全局评估与自动 Gamma 校准搜索
├── eval_feature_quality.py       # MMD/KL 散度计算与分布绘图
├── exp_robustness_noise.py       # 论文格式抗噪鲁棒性绘图脚本
└── requirements.txt              # 环境依赖文件
```
