# Ni-GAC Models — 偕双核镍(I)接力催化体系的反应产率预测

*[English](README.md) | 简体中文*

本仓库提供用于预测与分析三组分偶联反应产率的机器学习代码，重点关注过渡金属
（尤其是镍）催化的转化反应。

本项目对应论文：

> **Transforming Three-Component Coupling through Geminal Nickel(I) Relay Catalysts**
> （《通过偕双核镍(I)接力催化剂改造三组分偶联反应》）

---

## 1. 项目简介

本项目提供了用于催化三组分偶联反应产率预测的数据处理与建模代码。给定一个反应
（以 reaction SMILES 编码）及其过渡金属催化剂和其他试剂，模型预测反应的
**产率（yield）**，既可以作为连续值（回归），也可以作为高/低产率标签（分类）。

仓库包含以下几个互补的组成部分：

- **图神经网络模型（`YieldPredictor` / `YieldClassifier`）**：构建于内置的
  [Chemprop](https://github.com/chemprop/chemprop)（v2.2.0）之上。它将反应编码
  为 *缩合反应图（Condensed Graph of Reaction, CGR）*，将催化剂和其他试剂编码为
  分子图，并通过多层双向 **交叉图注意力（cross-graph attention）** 机制进行融合。
- **基线模型**：基于 Morgan 指纹特征的随机森林（Random Forest）和前馈多层感知机
  （MLP），以及一个可选的基于 SMILES 的 BERT 基线。
- **数据处理工具**：用于反应数据集的清洗、过滤、校验与切分。

> **关于适用范围的说明。** 大多数训练脚本是在 HPC 集群上开发并运行的，其中包含
> **硬编码的绝对数据路径**。在运行任何训练脚本前，你需要先修改这些路径（或设置
> 文档中说明的环境变量）以指向你自己的数据。参见
> [注意事项与可复现性](#11-注意事项与可复现性)。

---

## 2. 项目结构

下面仅列出最相关的文件。内置的 Chemprop 副本位于 `models/chemprop/` 下，作为库
使用；此处不再逐文件描述。

```
Ni_GAC_models-main/
├── README.md                      # 英文说明
├── README.zh-CN.md                # 本文件（中文说明）
├── .gitignore
│
├── split_train_val_test.py        # 分层切分 train/val/test（按是否含催化剂分层）
├── split_reagent.py               # 试剂列处理
├── train_yield_distribution.py    # 查看/绘制产率(y)分布 → train_y_hist.png
├── create_run_test_data.py        # 抽取小规模随机样本用于快速冒烟测试
├── dataset.py                     # （占位 / 近乎空文件）
│
├── check_*.py                     # 数据校验脚本（催化剂、试剂、反应物、Na 等）
├── clean_*.py                     # 数据清洗脚本（试剂 / 离子型试剂规范化）
├── filter_max_reagent.py          # 按试剂数量过滤行
├── print_long_reagent.py          # 诊断性打印过长的试剂字符串
│
└── models/
    ├── Ni_GAC_model.py            # ★ YieldPredictor：CGR↔催化剂 交叉注意力回归模型
    ├── Ni_GAC_model_noattn.py     #   消融对照：去掉交叉注意力的同款模型
    ├── Ni_GAC_classifier.py       # ★ YieldClassifier：高/低产率分类器（复用编码器）
    ├── Ni_GAC_classifier_noattn.py#   去掉交叉注意力的消融分类器
    │
    ├── combined_featurizers.py    # CGR + 分子特征化器（Chemprop）及 SMILES→MolGraph 辅助函数
    ├── new_dataset.py             # RxnGraphDataset：一个反应 → {cgr, cat, other, y}
    ├── new_collate.py             # collate_graph_batch：将 CGR / 催化剂 / 试剂图打包成 batch
    ├── model_dataloader.py        # build_loader()：从 CSV 构建 PyTorch DataLoader
    ├── collate.py / dataset.py    # 早期的 collate/dataset 变体
    │
    ├── train_yield.py             # ★ 训练回归模型（Lightning，多 GPU DDP）
    ├── train_yield_noattn.py      #   训练去注意力的回归消融模型
    ├── train_yield_classifier.py  # ★ 训练分类模型
    ├── train_yield_classifier_noattn.py  # 训练去注意力的消融分类器
    ├── test_yield_classifier.py   #   评估已训练的分类器检查点（注意事项见 §6）
    ├── test_data_loader.py        #   数据管线冒烟测试
    ├── test_reaction_featurization.py
    │
    ├── RF_feature.py              # 预计算 Morgan 指纹特征 .npy 文件
    ├── RF_model.py                # 随机森林产率回归
    ├── RF_model_classifier.py     # 随机森林产率分类
    ├── MLP_model.py               # 基于预计算特征的 MLP 回归/分类
    ├── MLP_regression_meta.json   # 已保存的 MLP 元信息（超参数 + 标准化器）
    ├── MLP_classification_meta.json
    │
    ├── train_pl_bert.py           # 可选的 SMILES BERT 基线（HuggingFace Transformers）
    ├── investigate_model.py       # 检查检查点（例如 EarlyStopping 状态）
    ├── find_ni.py                 # 查找催化剂中含镍的行
    ├── build_model_blockdiagram.py# 渲染模型框图（python-pptx）
    │
    ├── preprocess/                # 数据过滤管线（可特征化性、坏行映射、统计）
    │   ├── filter_all_data_with_featurization.py
    │   ├── clean_featurizationable_data.py
    │   ├── correct_wrong_data.py
    │   ├── find_all_bad_data.py / find_all_bad_entry_info.py / map_all_bad_data_index.py
    │   ├── filter_data_accord_num.py
    │   ├── change_LIAlH4.py
    │   └── stat_of_data.py        # other_reagent 中分子个数的分布
    │
    ├── featurization/             # 特征化辅助代码
    └── chemprop/                  # 内置的 Chemprop 库（v2.2.0）
```

> 一些预处理脚本在仓库根目录和 `models/preprocess/` 下同时存在
> （例如 `clean_featurizationable_data.py`）。`models/preprocess/` 中的副本是更
> 完善的版本。

---

## 3. 环境配置与安装

### Python 版本

内置的 Chemprop 为 **v2.2.0**，仓库中的编译缓存由 **CPython 3.12** 生成。建议使用
**Python 3.11 或 3.12**。

### 依赖

仓库中**没有 `requirements.txt`、`environment.yml` 或 `pyproject.toml`**。以下依赖
列表是根据代码中的 import 语句推断得到的；**在依赖它之前请自行核对具体版本。**

核心（图模型）：

- `torch`
- `lightning`（PyTorch Lightning ≥ 2.2）
- `torchmetrics`
- `rdkit`
- `numpy`、`pandas`

基线 / 工具：

- `scikit-learn`、`joblib`（随机森林、MLP 的标准化/指标）
- `tqdm`
- `matplotlib`（可选；产率分布绘图）
- `rich`（可选；更美观的进度条）
- `transformers`（仅 `train_pl_bert.py` 需要）
- `python-pptx`（仅 `build_model_blockdiagram.py` 需要）

`chemprop` 本身已**内置**于 `models/chemprop/` 下并被直接导入；你**无需**
`pip install chemprop`。但 Chemprop 自身的运行时依赖（RDKit、PyTorch、Lightning 等）
必须安装。

安装示例（请按需调整）：

```bash
conda create -n ni_gac python=3.12
conda activate ni_gac

# RDKit 推荐通过 conda 安装最为可靠
conda install -c conda-forge rdkit

# PyTorch：请按 https://pytorch.org 选择匹配的 CUDA 版本
pip install torch

pip install lightning torchmetrics numpy pandas scikit-learn joblib tqdm \
            matplotlib rich
# 可选基线：
pip install transformers python-pptx
```

> **导入路径说明。** 训练脚本混用了包式导入
> （`from models.Ni_GAC_model import YieldPredictor`）与扁平导入
> （`from model_dataloader import build_loader`、`from combined_featurizers import ...`）。
> 这实际上意味着 **仓库根目录和 `models/` 目录都必须在 `PYTHONPATH` 中**。
> 一种简单做法是在 `models/` 目录内运行训练脚本，同时把仓库根目录加入路径，例如：
> ```bash
> cd models
> PYTHONPATH="..:.:$PYTHONPATH" python train_yield.py
> ```
> 运行前请在你的环境中核对导入设置是否正确。

---

## 4. 数据说明

### 期望格式

模型读取一个 **CSV** 文件，其中（至少）包含以下列：

| 列名                          | 说明                                                              |
| ----------------------------- | ----------------------------------------------------------------- |
| `reaction_smiles`             | 反应 SMILES（`reactants>agents>products` 或 `reactants>>products`）。将被转换为缩合反应图（CGR）。 |
| `transition_metal_catalyst`   | 催化剂 SMILES；多个物种以 `.` 分隔。可以为空。                     |
| `other_reagent`               | 其他试剂 SMILES；多个物种以 `.` 分隔。可以为空。                   |
| `yield`                       | 反应产率（数值，期望范围 0–100）。                                 |

`transition_metal_catalyst` / `other_reagent` 列为空时会被妥善处理（以一个空分子图
代替）。

### 数据可获得性

底层的反应数据集**未包含**在本仓库中（`*.csv`、`*.npy` 以及切分目录等数据文件已
通过 `.gitignore` 排除）。你必须按上述格式自行准备数据。

原始脚本中引用了诸如 `final_data_featurizable_max.csv` 之类的数据集文件，以及形如
`.../Ni_GAC_models/splits/` 的切分目录。**这些绝对路径是作者环境下的占位符，必须
替换。** *（待补充信息：反应数据集的来源、规模及是否公开——参见
[§12](#12-需要用户补充的信息)。）*

### 生成 train/val/test 切分

`split_train_val_test.py` 执行分层切分，使验证集和测试集包含固定比例（默认 30%）的
**含**过渡金属催化剂的反应，并使用固定随机种子（`42`）以保证可复现。修改文件顶部的
`CSV_PATH` / `OUT_DIR` / 各集合大小，然后运行：

```bash
python split_train_val_test.py
```

这会写出 `splits/train.csv`、`splits/val.csv` 和 `splits/test.csv`。

---

## 5. 使用方法

> 下面所有训练脚本在文件顶部都包含硬编码的 `TRAIN_CSV` / `VAL_CSV` / `TEST_CSV` /
> `BASE_DIR` 常量。**运行前请修改这些常量。** 许多超参数通过环境变量读取（见各脚本）。
> *运行前请核对参数。*

### 5.1 检查数据

```bash
# 产率分布 + 直方图（输出 train_y_hist.png）
python train_yield_distribution.py --csv splits/train.csv --clip01

# other_reagent 中分子个数的分布（在文件中修改 CSV 路径）
python models/preprocess/stat_of_data.py
```

### 5.2 图模型 — 回归（`YieldPredictor`）

修改 `models/train_yield.py` 中的路径，然后运行：

```bash
cd models
# 超参数可通过环境变量覆盖：
BATCH_SIZE=256 LR=3e-4 D_HIDDEN=1024 N_LAYERS=4 GPU_IDS=0,1,2,3 \
  python train_yield.py
```

主要行为：

- 通过 PyTorch Lightning 进行多 GPU **DDP** 训练（`GPU_IDS` 选择设备）。
- 使用 `OneCycleLR` 调度、`AdamW`、MSE 损失；输出通过 `100 * sigmoid(.)` 压缩到
  `[0, 100]`。
- **自动续训**：若 `BASE_DIR/checkpoints` 中已存在检查点，则从最新的一个续训。
- 检查点（按 `val_loss` 取 top-3 + `last` + `final_full.ckpt`）和 CSV 日志写入
  `BASE_DIR` 下。

### 5.3 图模型 — 分类（`YieldClassifier`）

```bash
cd models
BATCH_SIZE=256 LR=3e-4 D_HIDDEN=1024 N_LAYERS=4 \
  YIELD_THRESHOLD=10.0 GPU_IDS=0,1,2,3 \
  python train_yield_classifier.py
```

当 `yield >= YIELD_THRESHOLD`（默认 `10.0`）时，反应被标注为高产率。训练以验证集
**AUROC** 为监控指标，并启用早停，同时记录 准确率 / 平均精度 / AUROC / F1。

### 5.4 消融实验（去掉交叉注意力）

```bash
cd models
python train_yield_noattn.py             # 回归消融
python train_yield_classifier_noattn.py  # 分类消融
```

### 5.5 随机森林基线

```bash
# 第 1 步：预计算 Morgan 指纹特征（逐切分）
python models/RF_feature.py --csv splits/train.csv --split_name train \
    --save-dir RF_feature --bits 128 --radius 2
python models/RF_feature.py --csv splits/val.csv   --split_name val \
    --save-dir RF_feature --bits 128 --radius 2
python models/RF_feature.py --csv splits/test.csv  --split_name test \
    --save-dir RF_feature --bits 128 --radius 2

# 第 2 步：在预计算特征上训练随机森林回归器
python models/RF_model.py --feat_dir RF_feature --outdir RF_run --n-iter-search 50
```

说明：
- 特征为 `[Δ指纹, 催化剂指纹, 试剂指纹, （可选描述符）, 2 个缺失标志]`。
  当 `--bits 128` 且不使用描述符时，维度为 `3*128 + 2 = 386`。
- 若省略 `--feat_dir`，`RF_model.py` 也可从 CSV（`--train/--val/--test`）实时构建
  特征。脚本中存在超参数搜索代码块，但已被注释掉，默认使用一组固定的最佳参数。
  *运行前请核对参数。*
- `RF_model_classifier.py` 提供对应的分类版本。

### 5.6 MLP 基线

在相同的预计算特征 `.npy` 文件上运行：

```bash
python models/MLP_model.py --feat_dir RF_feature --outdir MLP_run \
    --task regression --epochs 80 --layers 256,512,512,256,128
# 分类：
python models/MLP_model.py --feat_dir RF_feature --outdir MLP_cls_run \
    --task classification --threshold 10.0
```

### 5.7 可选：SMILES BERT 基线

`models/train_pl_bert.py` 在反应 SMILES 上微调一个 HuggingFace 序列分类模型。它由
argparse 驱动；具体选项请查看脚本。*运行前请核对参数。*

---

## 6. 评估

- **图回归**：训练过程中记录验证集 `val_loss`（MSE）；最佳检查点按 `val_loss` 选取。
- **图分类**：`test_yield_classifier.py` 用于加载已训练的检查点并报告测试集
  准确率 / AUROC / F1 / AP。

  > **注意事项：** `test_yield_classifier.py` 目前从
  > `Ni_GAC_models.models.train_yield_classifier_v1` 导入，这与当前的文件名
  > （`train_yield_classifier.py`）或包结构不匹配，且默认的检查点文件名是硬编码的。
  > 该脚本很可能需要先根据你的目录结构进行修改才能运行。*运行前请核对。*

- **RF / MLP**：两者都会写出 `val_predictions.csv` 和 `test_predictions.csv`，并
  记录 MAE / RMSE / R²（回归）或 准确率 / F1 / 精确率 / 召回率 / AUC / AP（分类）。

---

## 7. 模型说明

### 核心图模型（`models/Ni_GAC_model.py`）

`YieldPredictor` 是一个 PyTorch Lightning 模块，由以下部分组成：

1. **图编码器（`GraphEncoder`）**：一个有向键消息传递网络（Chemprop 的
   `BondMessagePassing`，即 D-MPNN），后接均值聚合。共使用三个独立的编码器：
   - 一个用于 **CGR**（缩合反应图），
   - 一个用于 **催化剂** 分子图，
   - 一个用于 **其他试剂** 分子图。

   每个编码器同时返回节点级嵌入（保留供注意力使用）和图级嵌入（均值池化）。

2. **其他试剂融合**：其他试剂的图嵌入在反应级别上做均值池化，并拼接到每个催化剂
   节点上，然后再下投影（`cat_other_proj`），使催化剂节点在进入注意力之前携带试剂
   上下文信息。

3. **多层双向交叉图注意力（`MultiCrossGraphAttn` / `_AttnLayer`）**：多头注意力在
   催化剂节点与 CGR 节点之间双向运行，并加掩码使注意力限制在每个反应内部。各层的
   注意力权重可选择性保留（`return_attn=True`）以便通过 `get_last_attention()`
   进行解释。输出按反应做均值池化。

4. **回归头（`mlp_out`）**：拼接池化后的 CGR、催化剂和其他试剂表示（`3 * d_h`），
   映射为单个数值；输出通过 `100 * sigmoid(.)` 缩放到 `[0, 100]`。

训练使用 **MSE 损失**、**AdamW**（权重衰减 `1e-2`）以及 **OneCycleLR** 调度
（余弦退火，10% 预热）。

### 分类器（`models/Ni_GAC_classifier.py`）

`YieldClassifier` 复用相同的 `GraphEncoder` 和 `MultiCrossGraphAttn` 组件，但将输出
头替换为二分类器，使用 `BCEWithLogitsLoss` 训练。标签通过对产率阈值化得到
（`yield >= threshold`，默认 `10.0`）。它通过 `torchmetrics` 报告 准确率、平均精度、
AUROC 和 F1。

### 消融模型

`Ni_GAC_model_noattn.py` 和 `Ni_GAC_classifier_noattn.py` 是对照变体，去掉了交叉
注意力，改为通过直接均值池化来融合催化剂与 CGR 表示，用于与基于注意力的模型作对比。

### 基线

- **随机森林**（`RF_model.py`、`RF_model_classifier.py`）：基于计数型 Morgan 指纹
  及可选的 RDKit 描述符。
- **MLP**（`MLP_model.py`）：基于标准化后的指纹特征。
- **BERT**（`train_pl_bert.py`）：基于原始反应 SMILES（可选）。

---

## 8. 输出结果

根据所运行的组件，输出包括：

- **检查点**（`*.ckpt`），例如按指标取 top-k、`last.ckpt`、`final_full.ckpt` ——
  位于 `BASE_DIR/checkpoints/` 下。
- **日志**：每次运行的 CSV 指标（PyTorch Lightning `CSVLogger`）和文本日志
  （`run_resume.log`、`training.log`），位于 `BASE_DIR/logs/` 下。
- **预测**：`val_predictions.csv`、`test_predictions.csv`（RF / MLP）。
- **训练曲线 / 元信息**：`training_curve.csv`、`meta.json`（MLP）；
  `model_final.joblib`、`search_results.csv`（RF）。
- **特征文件**：`X_{split}.npy`、`y_{split}.npy`（RF 特征提取）。
- **图像**：`train_y_hist.png`（产率分布）。

> 大多数生成的产物（`*.ckpt`、`*.pt`、`*.npy`、`*.joblib`、`splits/`、`run*/`、
> `*_meta.json` 等）已列入 `.gitignore`，**不**在仓库中被跟踪。

---

## 9. 论文引用

如果你使用了本代码，请引用相关论文：

> Transforming Three-Component Coupling through Geminal Nickel(I) Relay
> Catalysts.

BibTeX 条目占位符（**待补充最终发表信息**）：

```bibtex
@article{ni_gac_relay_catalysts,
  title   = {Transforming Three-Component Coupling through Geminal Nickel(I) Relay Catalysts},
  author  = {TODO: 作者列表},
  journal = {TODO: 期刊},
  year    = {TODO: 年份},
  volume  = {TODO},
  pages   = {TODO},
  doi     = {TODO}
}
```

本项目还基于 **Chemprop** 构建；如果你使用了图模型，请按需引用 Chemprop。

---

## 10. 许可证（License）

许可证信息待补充。本仓库当前不包含 `LICENSE` 文件。

> 注意：本仓库在 `models/chemprop/` 下内置了一份 **Chemprop** 副本，它按其自身的
> 许可证分发。请查阅并遵守上游 Chemprop 的许可条款。

---

## 11. 注意事项与可复现性

- **硬编码路径。** 训练/评估脚本中包含作者计算环境下的绝对路径
  （例如 `/HOME/scz0sy0/...`、`/sharefs/...`、`/home/liujie/...`）。运行前请改为你
  自己的位置。
- **环境变量。** 许多超参数（`BATCH_SIZE`、`LR`、`D_HIDDEN`、`N_LAYERS`、`GPU_IDS`、
  `YIELD_THRESHOLD`、`NUM_WORKERS` 等）在图模型训练脚本中通过环境变量读取。
- **硬件 / GPU。** 图模型脚本默认进行多 GPU DDP 训练
  （`GPU_IDS=0,1,2,3,4,5,6,7`）。请将 `GPU_IDS` 设置为与你的机器匹配，或缩减为
  单块 GPU。精度为 `32-true`。
- **随机种子。** 数据切分（`split_train_val_test.py`，种子 `42`）和 RF/MLP 基线
  （`--seed 42`）设置了种子以保证可复现。图模型训练脚本默认不设置全局种子；如需精确
  复现，请自行添加。
- **续训。** `train_yield.py` 会自动从 `BASE_DIR/checkpoints` 中找到的最新检查点续训；
  如需从头开始，请删除或移走旧检查点。
- **导入路径。** 参见 [§3](#3-环境配置与安装) 中的导入路径说明。
- **过时脚本。** `test_yield_classifier.py` 引用了与当前目录结构不匹配的模块/文件名
  （见 [§6](#6-评估)）。

---

## 12. 需要用户补充的信息

以下内容无法从代码中确定，需要补充：

1. **数据集**：来源、规模、许可，以及能否公开共享；下载链接或准备说明。
2. **规范的数据路径**：将硬编码的绝对路径替换为相对路径或配置文件。
3. **依赖版本锁定**：提供一个包含已测试版本的 `requirements.txt` / `environment.yml`
   （PyTorch、Lightning、RDKit、CUDA 等）。
4. **许可证**：选择并添加一个 `LICENSE` 文件（并确认与内置 Chemprop 许可证兼容）。
5. **引用信息**：BibTeX 条目所需的最终作者列表、期刊、年份、卷/页码和 DOI。
6. **报告的结果**：关键指标（例如回归的测试 MAE/R²、分类的 AUROC）以及发表模型所用的
   确切超参数。
7. **预训练检查点**：是否会发布已训练的模型权重，以及发布位置。
```
