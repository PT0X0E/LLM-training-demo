# LLM training demo

本项目为从零训练GPT风格大模型的极简实用代码，包括预训练、SFT、RLHF/DPO等流程。

---

## 🚀 快速开始

### 1. 安装依赖

```bash
conda create -n llm python=3.10 -y
conda activate llm
pip install -r requirements.txt
```

### 2. 下载数据

```bash
python src/data/download_base.py --dataset wikitext2
python src/data/download_sft.py --dataset alpaca
python src/data/download_preference.py --dataset ultrafeedback
```

### 3. 训练分词器

```bash
python src/data/tokenizer.py --action train
```

### 4. 预训练GPT模型

```bash
python src/training/pretrain.py --model_size small --batch_size 8 --num_epochs 10
```

### 5. SFT指令微调

```bash
python src/training/sft.py --pretrained_path pretrain_run --prompt_template alpaca --batch_size 4 --num_epochs 3
```

### 6. DPO偏好训练

```bash
python src/training/dpo.py --sft_model_path sft_run --batch_size 4 --num_epochs 1
```

### 7. RLHF训练（可选）

```bash
python src/training/rlhf.py reward --sft_model_path sft_run
python src/training/rlhf.py ppo --sft_model_path sft_run --reward_model_path reward_model_run
```

---
