#!/bin/bash

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
  --experiment_name="en_de_translation_kan_chebyshev" \
  --checkpoint_steps=5000 \
  --gradient_accumulation_steps=1 \
  --device="cuda:3" \
  --model_type="kan_chebyshev" \
  --chebykan_degree=4 \
  # --track_wandb \
