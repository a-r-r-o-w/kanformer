# kanformer

Kolomogorov-Arnold networks were recently proposed as a promising alternative to MLPs. This repository naively replaces linear layers of the [original transformer](https://arxiv.org/abs/1706.03762) implementation with KANLinear layers.

This repository is a toy playground for experimentation as I work my way through understanding more about KANs. I have a few ideas based on my current understanding on improving the architecture but so far everything performs worse in comparison to the original transformer with similar-ish parameter count. My next steps are to convert my dirty implementation of HF transformers GPT-KAN that looked somewhat promising and make available soon hopefully. Feel free to contribute if you find this interesting.

First run after making the KAN linear replacement:

![initial.png](./assets/initial.png)

### Installation

```bash
# Clone repository
git clone --recurse-submodules https://github.com/a-r-r-o-w/kanformer
cd kanformer

# Install python package
pip install -r requirements.txt
pip install .  # or "pip install -e ." for editable mode
python3 setup.py develop
```

### Usage

#### Using a GPT Config

```py
from kanformer import TransformerTextGeneration
from kanformer.config import ModelConfig, ModelType

# `ModelConfig` can be one of the predefined configurations:
#   - GPT2, GPT2_MEDIUM, GPT2_LARGE, GPT2_XL, GPT2_MICRO, GPT2_MINI, GPT2_NANO
# or, you can create your own configuration by editing the dictionary
#
# `ModelType` can be one of "MLP", "KAN_ORIGINAL", "KAN_EFFICIENT", "KAN_CHEBYSHEV", "KAN_FAST"
# Note: Using any KAN variant adds many more trainable parameters so be careful when comparing
#       with MLP.  Make sure to use the same number of parameters for a fair comparison. To get
#       similar number of parameters, you can reduce the embedding_dim, query_key_dim, value_dim,
#       ffn_hidden_dim, and other parameters for KAN variants.
config = ModelConfig.GPT2()
config["max_length"] = 1024
config["model_type"] = ModelType.MLP

model = TransformerTextGeneration.from_config(config)

print(model.config)
print(sum(p.numel() for p in model.parameters() if p.requires_grad)) # Parameters: 123,674,449

# If we initialized with ModelType.KAN_CHEBYSHEV, the number of parameters would be 656,294,400
# even if the configuration is the same. This is because KAN adds many more trainable parameters.
```

#### Initializing transformer directly

```py
import torch
from kanformer import TransformerSeq2Seq

# mlp
model = TransformerSeq2Seq(
    num_encoder_layers=3,
    num_decoder_layers=3,
    vocab_src_size=5000,
    vocab_tgt_size=5000,
    pad_src_idx=1,
    pad_tgt_idx=1,
    embedding_dim=512,
    query_key_dim=512,
    value_dim=512,
    num_heads=8,
    ffn_hidden_dim=768,
    ffn_activation="swiglu",
    dropout_rate=0.1,
    max_length=2048,
    model_type="mlp"
).to("cuda")
print(sum(p.numel() for p in model.parameters() if p.requires_grad)) # 21662720

# KAN efficient (https://github.com/Blealtan/efficient-kan/)
model = TransformerSeq2Seq(
    num_encoder_layers=3,
    num_decoder_layers=3,
    vocab_src_size=5000,
    vocab_tgt_size=5000,
    pad_src_idx=1,
    pad_tgt_idx=1,
    embedding_dim=128,
    query_key_dim=128,
    value_dim=128,
    num_heads=8,
    ffn_hidden_dim=512,
    ffn_activation="swiglu",
    dropout_rate=0.1,
    max_length=2048,
    use_kan_bias=True,
    model_type="kan_chebyshev"
).to("cuda")
print(sum(p.numel() for p in model.parameters() if p.requires_grad)) # 13331200

batch_size = 32
seq_length = 512
en_tensors = torch.randint(0, 5000, (batch_size, seq_length)).to("cuda")
de_tensors = torch.randint(0, 5000, (batch_size, seq_length)).to("cuda")

output = model(en_tensors, de_tensors)
print(output.shape)  # (batch_size, seq_length, vocab_tgt_size)
```

<details>
<summary> Training </summary>

Currently, there are various limitations with the codebase that will be improved soon. For experimentation, Multi30k has been hardcoded.

**model_type:** Can be one of "mlp", "kan_original", "kan_efficient", "kan_chebyshev" or "kan_fast".

```bash
# MLP
python3 main.py train \
  --num_encoder_layers=3 \
  --num_decoder_layers=3 \
  --vocab_src_size=5000 \
  --vocab_tgt_size=5000 \
  --pad_src_idx=-1 \
  --pad_tgt_idx=-1 \
  --embedding_dim=512 \
  --query_key_dim=512 \
  --value_dim=512 \
  --num_heads=8 \
  --ffn_hidden_dim=1024 \
  --ffn_activation="swiglu" \
  --use_pffn_bias \
  --use_final_linear_bias \
  --dropout_rate=0.1 \
  --max_length=32 \
  --weight_initialization_method="kaiming_uniform" \
  --learning_rate=1e-4 \
  --weight_decay=0.0001 \
  --batch_size=32 \
  --dataset_name="multi30k" \
  --epochs=20 \
  --seed=42 \
  --validation_epochs=1 \
  --checkpoint_path="checkpoints" \
  --experiment_name="en_de_translation_mlp" \
  --checkpoint_steps=5000 \
  --gradient_accumulation_steps=1 \
  --device="cuda:0" \
  --model_type="mlp" \
  --track_wandb

# Efficient KAN
python3 main.py train \
  --num_encoder_layers=3 \
  --num_decoder_layers=3 \
  --vocab_src_size=5000 \
  --vocab_tgt_size=5000 \
  --pad_src_idx=-1 \
  --pad_tgt_idx=-1 \
  --embedding_dim=128 \
  --query_key_dim=128 \
  --value_dim=128 \
  --num_heads=4 \
  --ffn_hidden_dim=256 \
  --ffn_activation="swiglu" \
  --use_kan_bias \
  --use_pffn_bias \
  --use_final_linear_bias \
  --dropout_rate=0.1 \
  --max_length=32 \
  --weight_initialization_method="kaiming_uniform" \
  --learning_rate=1e-4 \
  --weight_decay=0.0001 \
  --batch_size=32 \
  --dataset_name="multi30k" \
  --epochs=20 \
  --seed=42 \
  --validation_epochs=1 \
  --checkpoint_path="checkpoints" \
  --experiment_name="en_de_translation_kan_efficient" \
  --checkpoint_steps=5000 \
  --gradient_accumulation_steps=1 \
  --device="cuda:0" \
  --model_type="kan_efficient" \
  --track_wandb
```
</details>

<details>
<summary> Inference </summary>

```bash
python3 main.py inference \
  --checkpoint_path="checkpoints" \
  --experiment_name="en_de_translation_mlp_relu" \
  --input="A man in shorts and a Hawaiian shirt leans over the rail of a pilot boat, with fog and mountains in the background." \
  --top_p=0.7 \
  --temperature=1 \
  --sample \
  --max_length=100

# Output:
Input: A man in shorts and a Hawaiian shirt leans over the rail of a pilot boat, with fog and mountains in the background.
Output: <sos> ein mann in shorts und mit sonnenbrille lehnt sich über ein geländer des pp des grill s und einem motor blick über das blaues see . <eos>
Generated token indices: [0, 73, 93, 71, 731, 87, 90, 735, 995, 147, 207, 73, 1120, 326, 171, 326, 1387, 49, 87, 83, 557, 413, 207, 226, 1564, 1010, 14, 1]
```
</details>

### TODO

- [x] Implement transformer base
- [x] Use original KAN (https://github.com/KindXiaoming/pykan/)
- [x] Use Efficient KAN (https://github.com/Blealtan/efficient-kan/)
- [x] Use Chebyshev KAN (https://github.com/SynodicMonth/ChebyKAN)
- [x] Use Fast KAN (https://github.com/ZiyaoLi/fast-kan)
- [ ] Dataset agnostic training
- [x] wandb logging
- [ ] Implement RoPE
- [ ] Implement MoE
- [x] Implement GLU activation variants
- [x] Checkpointing
- [x] Gradient accumulation
- [ ] MultiGPU support using HF ecosystem
- [x] Common GPT configs
- [ ] Common BERT configs
- [x] Improve docs
- [ ] Tests

```
@misc{liu2024kan,
      title={KAN: Kolmogorov-Arnold Networks}, 
      author={Ziming Liu and Yixuan Wang and Sachin Vaidya and Fabian Ruehle and James Halverson and Marin Soljačić and Thomas Y. Hou and Max Tegmark},
      year={2024},
      eprint={2404.19756},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
