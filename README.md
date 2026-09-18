# Predictor0916

Predictor0916 使用大语言模型在 prefill 阶段产生的隐藏状态预测该请求的输出长度。当前实现从 LLM 最后一层、prompt 最后一个非 padding token 提取表示，并使用轻量级 MLP 将表示映射到长度区间上的概率分布，最终以各区间中心的期望作为预测长度。

项目当前包含两个阶段：

1. 数据预处理：从 [ForeLen](https://huggingface.co/datasets/abinzzz/ForeLen) 读取 prompt，调用指定 LLM 提取隐藏状态并生成完整回答长度，保存为 Parquet。
2. MLP 训练：读取 Parquet，划分训练集和验证集，训练长度预测头并保存指标、预测结果和模型参数。

## 项目结构

```text
Predictor0916/
├── README.md
├── data/                         # 默认数据目录，首次运行时自动创建
│   └── llama3.2-1b-rl-generated.parquet
├── outputs/                      # 默认实验输出目录，首次运行时自动创建
│   └── test_training[_npu]/
├── scripts/
│   ├── run_training_gpu.sh       # NVIDIA GPU 启动脚本
│   ├── run_training_npu.sh       # Ascend NPU 启动脚本
│   ├── test_training.py          # 完整的数据处理、训练和验证入口
│   └── test_training_colab.ipynb # Google Colab GPU Notebook
├── src/
│   ├── __init__.py
│   ├── dataset.py                # ForeLen 预处理、Parquet 读写及数据划分
│   ├── model.py                  # LLM 加载、隐藏状态提取和回答长度生成
│   ├── mlp.py                    # MLP 长度预测头及参数保存/加载
│   ├── trainer.py                # 损失函数、训练、验证和 early stopping
│   └── utils.py                  # 随机种子、设备、指标、日志和 JSON 工具
└── tests/
    └── test_pipeline.py          # 单元测试与端到端流水线测试
```

## 环境要求

基础依赖包括：

```text
Python
PyTorch
Transformers
datasets
huggingface_hub
pandas
NumPy
PyArrow
pytest（仅测试需要）
```

Ascend NPU 环境还需要：

- 与硬件和驱动匹配的 CANN；
- 与 PyTorch、CANN 版本匹配的 `torch_npu`；
- 正确配置的 Ascend 环境变量。必要时在运行脚本前执行 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`，并按实际安装位置修改路径。

如果使用 Hugging Face 上需要授权的模型，例如默认的 Llama 模型，需要提前申请模型访问权限并完成 Hugging Face 登录，或者将 `--model-id-or-path` 指向已经下载的本地模型目录。

## 运行流程

`scripts/test_training.py` 按以下顺序运行：

1. 解析命令行参数；
2. 当设备为 `npu:*` 时导入 `torch_npu`；
3. 设置随机种子；
4. 根据 `--model-id-or-path` 初始化并加载 LLM；
5. 使用同一个 LLM 实例初始化 `Dataset`；
6. 检查 `--data-path`：
   - 文件存在：直接加载；
   - 文件不存在：先调用 `Dataset.process()` 生成，再加载；
   - 路径存在但不是文件：抛出 `IsADirectoryError`；
7. 随机划分训练集和验证集；
8. 根据训练目标的指定分位数计算 MLP 长度范围；
9. 初始化 MLP，并按需加载已有参数；
10. 初始化 `Trainer`，训练并根据验证集 MAE 选择最佳状态；
11. 在验证集上计算 MAE、RMSE 和 R²；
12. 保存 checkpoint、配置、训练历史、验证集预测和实验记录。

### 数据文件不存在时

默认从 ForeLen 的 `llama3.2-1b-rl` subset、`train` split 读取 `user_prompt_content`。对每条 prompt：

- `Model.extract()` 提取最后一层最后一个有效 prompt token 的隐藏状态；
- `Model.generate()` 生成回答并统计生成 token 数；
- 终止 EOS 和尾部 padding 不计入回答长度；
- 默认最大生成长度为 8192 个新 token。

该过程需要逐条执行 LLM prefill 和生成，可能花费较长时间。中途失败时临时文件会被删除，只有全部处理完成后才会产生最终 Parquet。

### 已处理数据格式

`--data-path` 指向的 Parquet 至少需要以下两列：

| 列名 | 类型 | 含义 |
| --- | --- | --- |
| `hidden_state` | 一维浮点数组 | 一条 prompt 对应的 LLM 隐藏状态；所有样本维度必须相同 |
| `response_length` | 数值 | 对应回答的生成 token 数；由 `Dataset.process()` 生成时为非负整数 |

文件至少需要两条样本，且隐藏状态和回答长度不能包含 NaN 或无穷值。

## 启动方式

所有命令均建议从仓库根目录运行。

### Ascend NPU

直接编辑 `scripts/run_training_npu.sh` 中每一行参数，然后运行：

```bash
bash Predictor0916/scripts/run_training_npu.sh
```

脚本的核心命令为：

```bash
python Predictor0916/scripts/test_training.py \
  --model-id-or-path meta-llama/Llama-3.2-1B-Instruct \
  --llm-device npu:0 \
  --device npu:0 \
  --torch-dtype float16 \
  --llm-batch-size 1 \
  --data-path Predictor0916/data/llama3.2-1b-rl-generated.parquet \
  --output-dir Predictor0916/outputs/test_training_npu \
  --epochs 10 \
  --batch-size 256
```

`ASCEND_RT_VISIBLE_DEVICES=0` 后，程序内部使用的第一张可见卡为逻辑设备 `npu:0`。

### CPU 或 CUDA GPU

NVIDIA GPU 可以直接运行预置脚本，并按需编辑其中的参数：

```bash
bash Predictor0916/scripts/run_training_gpu.sh
```

默认使用 `bfloat16`；如果 GPU 不支持 BF16，请在脚本中将 `--torch-dtype` 改为 `float16`。

等价的核心命令为：

```bash
python Predictor0916/scripts/test_training.py \
  --model-id-or-path meta-llama/Llama-3.2-1B-Instruct \
  --llm-device cuda \
  --device cuda \
  --torch-dtype bfloat16 \
  --llm-batch-size 4 \
  --data-path Predictor0916/data/llama3.2-1b-rl-generated.parquet \
  --output-dir Predictor0916/outputs/test_training_cuda
```

使用 CPU 时将两个设备参数改为 `cpu`，并建议使用 `--torch-dtype float32`。

### Google Colab

在 Colab 中打开 `scripts/test_training_colab.ipynb`，选择 GPU 运行时后按顺序执行各单元格。Notebook 不调用封装后的 `training_main()`，而是将数据准备、Qwen 模型下载、ForeLen 数据处理、Trainer 初始化、训练和结果保存拆成独立步骤。Notebook 默认使用 `Qwen/Qwen2.5-0.5B-Instruct` 和 ForeLen 的 `qwen2.5-0.5b-longseq` subset；Colab T4 默认使用 `float16`。

### 从已有 MLP 参数继续训练

```bash
python Predictor0916/scripts/test_training.py \
  --model-id-or-path meta-llama/Llama-3.2-1B-Instruct \
  --data-path Predictor0916/data/llama3.2-1b-rl-generated.parquet \
  --load-checkpoint Predictor0916/outputs/previous/checkpoints/best_layers.pt
```

`MLP.load()` 使用严格参数匹配。checkpoint 对应的 LLM 隐藏维度和 `--num-bins` 必须与本次运行一致。checkpoint 只包含 `self.layers`，长度区间中心会根据本次训练数据的分位数重新计算。

## 输入参数

### LLM 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model-id-or-path` | `meta-llama/Llama-3.2-1B-Instruct` | Hugging Face 模型 ID 或本地模型目录 |
| `--llm-batch-size` | `1` | LLM 隐藏状态提取和生成时的 batch size |
| `--llm-device` | `auto` | LLM 设备，例如 `cpu`、`cuda`、`cuda:0`、`npu:0` |
| `--torch-dtype` | `bfloat16` | LLM 权重类型，可选 `bfloat16`、`float16`、`float32` |
| `--max-prompt-length` | 不限制 | 可选的 prompt tokenizer 截断长度，必须为正整数 |
| `--trust-remote-code` | 开启 | 允许加载 Hugging Face 仓库中的自定义代码 |
| `--no-trust-remote-code` | — | 禁用自定义远程代码 |

### 数据和输出参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--data-path` | `Predictor0916/data/llama3.2-1b-rl-generated.parquet` | 已处理 Parquet 的读取/生成位置 |
| `--dataset-subset` | `llama3.2-1b-rl` | Parquet 不存在时使用的 ForeLen config；切换到 Qwen2.5-0.5B 时可设为 `qwen2.5-0.5b-rl` |
| `--output-dir` | `Predictor0916/outputs/test_training` | checkpoint、指标和预测结果的输出目录 |
| `--validation-ratio` | `0.2` | 验证集比例，必须严格位于 `(0, 1)` |
| `--seed` | `42` | 数据划分、参数初始化和训练随机种子 |

### MLP 和训练参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--num-bins` | `20` | 输出长度离散区间数量，至少为 2 |
| `--target-quantiles` `LOW HIGH` | `0.01 0.99` | 使用训练长度的 LOW–HIGH 分位数确定预测范围，满足 `0 ≤ LOW < HIGH ≤ 1` |
| `--loss-type` | `soft_label` | 损失类型，可选 `soft_label` 或 `mae` |
| `--lambda-val` | `0.95` | soft-label 损失中 KL 项的权重，范围 `[0, 1]` |
| `--epochs` | `10` | 最大训练轮数 |
| `--batch-size` | `256` | MLP 训练 batch size |
| `--learning-rate` | `2e-5` | AdamW 学习率 |
| `--weight-decay` | `0.0` | AdamW weight decay |
| `--patience` | `3` | Early stopping 容忍轮数；设为 `0` 时关闭 |
| `--device` | `auto` | MLP 训练设备，与 `--llm-device` 独立 |
| `--load-checkpoint` | 无 | 使用 `MLP.save()` 生成的 `self.layers` 参数文件 |

soft-label 模式使用以下组合损失：

```text
Loss = lambda_val * KL(log_softmax(logits), soft_labels)
     + (1 - lambda_val) * MSE(predicted_length, target_length)
```

### DataLoader 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--num-workers` | `0` | `Dataset.load()` 使用的 worker 数量 |
| `--pin-memory` | 关闭 | 为 `Dataset.load()` 创建的 DataLoader 启用 pinned memory |
| `--persistent-workers` | 关闭 | worker 在 epoch 间保持；仅在 `num_workers > 0` 时使用 |
| `--prefetch-factor` | PyTorch 默认值 | 每个 worker 预取的 batch 数；必须为正数且仅在 `num_workers > 0` 时使用 |
| `--drop-last` | 关闭 | 训练 DataLoader 是否丢弃最后一个不完整 batch |

当前训练入口从 `Dataset.load()` 返回的 `TensorDataset` 中取出完整 tensors，再由 `Trainer` 创建实际训练 DataLoader。因此 `num_workers`、`pin_memory`、`persistent_workers`、`prefetch_factor` 和 `drop_last` 目前不会改变 `Trainer` 内部 DataLoader 的行为；这些参数保留用于后续将 Trainer 改为直接消费 DataLoader。

查看实时参数说明：

```bash
python Predictor0916/scripts/test_training.py --help
```

## 输出结构

假设使用：

```text
--output-dir Predictor0916/outputs/test_training_npu
```

运行后得到：

```text
Predictor0916/outputs/test_training_npu/
├── checkpoints/
│   └── best_layers.pt
├── manifest.json
├── metrics.json
├── results.jsonl
└── validation_predictions.csv
```

### `checkpoints/best_layers.pt`

验证集 MAE 最优轮次对应的 MLP `self.layers.state_dict()`。包含两个 Linear 层和 LayerNorm 的参数，不包含 LLM 权重，也不包含 `bin_centers`。

### `manifest.json`

记录本次实验配置以及从数据推导出的参数，包括：

- LLM 路径、LLM batch size、设备、dtype 和 prompt 长度限制；
- 数据路径、验证集比例和随机种子；
- 隐藏状态维度、bin 数量、目标分位数和实际目标范围；
- 损失、学习率、epoch、batch size、weight decay 和 early stopping；
- 可选的输入 checkpoint 路径。

### `metrics.json`

包含两部分：

- `metrics`：验证集 `mae`、`rmse` 和 `r2`；
- `history`：逐 epoch 的 `train_loss`、`train_mae`、`validation_loss`、`validation_mae`，以及 `best_epoch`、`best_score`、`epochs_trained`、`stopped_early`。

### `validation_predictions.csv`

每条验证样本一行：

| 列名 | 含义 |
| --- | --- |
| `sample_index` | 当前验证集内的顺序索引 |
| `predicted_length` | MLP 预测的回答 token 数 |
| `target_length` | 预处理阶段生成的真实回答 token 数 |
| `absolute_error` | 两者的绝对误差 |

### `results.jsonl`

每次运行追加一行 JSON，不会覆盖历史记录。每条记录包含：

- `config`：完整实验配置；
- `split_sizes`：训练集和验证集样本数；
- `metrics`：验证集回归指标；
- `history`：训练历史；
- `artifacts`：checkpoint 和预测 CSV 的路径。

## 测试

从仓库根目录执行：

```bash
python -m pytest Predictor0916/tests -q
```

测试使用伪造的小模型和小规模数据，不会下载真实 LLM 或 ForeLen 数据集。

## 注意事项

- `--data-path` 已存在时不会重新生成数据。用户需要确保该文件确实由 `--model-id-or-path` 指定的同一个 LLM 生成；当前 Parquet 本身不保存模型身份元数据。
- 更换 LLM 时应同步设置 `--dataset-subset`。例如 Qwen2.5-0.5B 的 RL 数据对应 `qwen2.5-0.5b-rl`。
- 主训练脚本目前无论 Parquet 是否已经存在，都会先加载指定 LLM。请预留相应设备内存。
- 数据预处理默认使用确定性生成 `do_sample=False`。
- 更换 LLM 后隐藏维度可能变化，旧 MLP checkpoint 通常无法直接加载。
- `results.jsonl` 使用追加模式；重复运行不会清空旧记录。
