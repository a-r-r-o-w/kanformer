import os
import json
from typing import List, Optional, Union

import dotenv
import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import DataLoader
from tqdm import tqdm

from kanformer import (
    PositionalEncoding,
    EncoderDecoderTransformer,
    LRScheduler,
    ModelType,
)
from utils import get_summary, initialize_weights, collate_fn


dotenv.load_dotenv()

WANDB_API_KEY = os.getenv("WANDB_API_KEY")


def _print_with_line(content: str, line_length: int = 80):
    print(content)
    print("-" * line_length)


class CLI:
    r"""Command-line interface to interact with the transformer implementation for
    training or inference.
    """

    def __init__(self) -> None:
        pass

    def train(
        self,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        vocab_src_size: int = 25000,
        vocab_tgt_size: int = 25000,
        pad_src_idx: int = 24999,
        pad_tgt_idx: int = 24999,
        embedding_dim: int = 512,  # `d_model` in paper
        query_key_dim: int = 512,  # `d_k` in paper
        value_dim: int = 512,  # `d_v` in paper
        num_heads: int = 8,  # `h` in paper
        ffn_hidden_dim: int = 2048,
        ffn_activation: str = "relu",
        use_kan_bias: bool = True,
        use_pffn_bias: bool = True,
        use_final_linear_bias: bool = False,
        dropout_rate: float = 0.1,
        max_length: int = 10000,
        weight_initialization_method: str = "kaiming_uniform",
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        dataset_name: str = "multi30k",
        epochs: int = 10,
        seed: int = 42,
        validation_epochs: int = 1,
        checkpoint_path: str = "checkpoints",
        experiment_name: str = "transformer",
        checkpoint_steps: int = 500,
        gradient_accumulation_steps: int = 1,
        model_type: str = "mlp",
        track_wandb: bool = False,
    ) -> None:
        r"""Train the transformer model. You can configure various hyperparameters.

        Args:
            num_layers:
                Number of encoder/decoder layers to be used in the transformer.
        """

        config = {k: v for k, v in locals().items() if k != "self"}

        if track_wandb and not WANDB_API_KEY:
            raise ValueError("WANDB_API_KEY not set in environment variables")

        if track_wandb:
            wandb.login(key=WANDB_API_KEY)
            run = wandb.init(project="kanformer", name=experiment_name, config=config)
            print("Logged in to wandb")

        def wandb_log(log_dict: dict) -> None:
            if track_wandb:
                wandb.log(log_dict)

        if model_type not in [
            ModelType.MLP,
            ModelType.KAN_ORIGINAL,
            ModelType.KAN_EFFICIENT,
        ]:
            raise ValueError(f"Model type {model_type} not supported")

        torch.manual_seed(seed)
        np.random.seed(seed)

        sos_token = "<sos>"
        eos_token = "<eos>"
        unk_token = "<unk>"
        pad_token = "<pad>"

        experiment_dir = os.path.join(checkpoint_path, experiment_name)
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir, exist_ok=True)

        match dataset_name:
            case "multi30k":
                path = "dataset/multi30k"
                files = {
                    "train": "train.jsonl",
                    "test": "test.jsonl",
                    "val": "val.jsonl",
                }
                data = {}
                for split, filename in files.items():
                    if split not in ["train", "val", "test"]:
                        raise ValueError(f"Split '{split}' is not supported")

                    data[split] = []

                    with open(os.path.join(path, filename), "r") as f:
                        for line in f:
                            item = json.loads(line)
                            item["en"] = item["en"].lower()
                            item["de"] = item["de"].lower()
                            data[split].append(item)

                    # data[split] = data[split][:10000]

                sentences_en = [
                    item["en"] for split in data.keys() for item in data[split]
                ]
                sentences_de = [
                    item["de"] for split in data.keys() for item in data[split]
                ]

                tokenizer_en = Tokenizer(BPE(unk_token=unk_token))
                tokenizer_de = Tokenizer(BPE(unk_token=unk_token))
                tokenizer_en.pre_tokenizer = Whitespace()
                tokenizer_de.pre_tokenizer = Whitespace()

                trainer_en = BpeTrainer(
                    special_tokens=[sos_token, eos_token, unk_token, pad_token],
                    vocab_size=vocab_src_size,
                    min_frequency=2,
                )
                trainer_de = BpeTrainer(
                    special_tokens=[sos_token, eos_token, unk_token, pad_token],
                    vocab_size=vocab_tgt_size,
                    min_frequency=2,
                )

                tokenizer_en.train_from_iterator(sentences_en, trainer_en)
                tokenizer_de.train_from_iterator(sentences_de, trainer_de)
            case _:
                raise ValueError(f"Dataset {dataset_name} not supported")

        sos_token_idx = tokenizer_en.token_to_id(sos_token)
        eos_token_idx = tokenizer_en.token_to_id(eos_token)
        for split in data.keys():
            data_tensors = []
            for item in data[split]:
                item["en"] = (
                    [sos_token_idx]
                    + tokenizer_en.encode(item["en"]).ids
                    + [eos_token_idx]
                )
                item["de"] = (
                    [sos_token_idx]
                    + tokenizer_de.encode(item["de"]).ids
                    + [eos_token_idx]
                )
                item["en"] = torch.tensor(item["en"][:max_length], dtype=torch.long)
                item["de"] = torch.tensor(item["de"][:max_length], dtype=torch.long)
                data_tensors.append(item)
            data[split] = data_tensors

        if pad_src_idx == -1:
            pad_src_idx = tokenizer_en.token_to_id(pad_token)
        if pad_tgt_idx == -1:
            pad_tgt_idx = tokenizer_de.token_to_id(pad_token)

        def collate_helper(batch):
            return collate_fn(
                batch,
                en_pad_token_id=pad_src_idx,
                de_pad_token_id=pad_tgt_idx,
                max_length=max_length,
            )

        train_dataloader = DataLoader(
            [(item["en"], item["de"]) for item in data["train"]],
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_helper,
        )
        val_dataloader = DataLoader(
            [(item["en"], item["de"]) for item in data["val"]],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_helper,
        )
        test_dataloader = DataLoader(
            [(item["en"], item["de"]) for item in data["test"]],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_helper,
        )

        transformer = EncoderDecoderTransformer(
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            vocab_src_size=vocab_src_size,
            vocab_tgt_size=vocab_tgt_size,
            pad_src_idx=pad_src_idx,
            pad_tgt_idx=pad_tgt_idx,
            embedding_dim=embedding_dim,
            query_key_dim=query_key_dim,
            value_dim=value_dim,
            num_heads=num_heads,
            ffn_hidden_dim=ffn_hidden_dim,
            ffn_activation=ffn_activation,
            use_kan_bias=use_kan_bias,
            use_pffn_bias=use_pffn_bias,
            use_final_linear_bias=use_final_linear_bias,
            dropout_rate=dropout_rate,
            max_length=max_length,
            model_type=model_type,
        ).to(device="cuda")

        initialize_weights(transformer, weight_initialization_method)

        _print_with_line(transformer)
        _print_with_line(f"Summary:\n{get_summary(transformer)}")

        optimizer = optim.Adam(
            params=transformer.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            # based on section 5.3 of paper
            betas=(0.9, 0.98),
            eps=1e-9,
        )
        # lr_scheduler = LRScheduler(
        #     optimizer, embedding_dim=embedding_dim, warmup_steps=4000
        # )

        criterion = nn.CrossEntropyLoss(ignore_index=pad_tgt_idx)

        if track_wandb:
            wandb.watch(transformer, log="all", log_freq=1000)

        train_losses = []
        val_losses = []
        test_losses = []
        learning_rates = []
        step = 0
        total_steps = len(train_dataloader) * epochs

        with tqdm(total=total_steps, desc="Training") as train_bar:
            for epoch in range(1, epochs + 1):
                total_loss = 0.0

                transformer.train()
                for i, (en_tensors, de_tensors) in enumerate(train_dataloader):
                    en_tensors = en_tensors.to(device="cuda")
                    de_tensors = de_tensors.to(device="cuda")
                    src_de = de_tensors[:, :-1]
                    tgt_de = de_tensors[:, 1:].contiguous().view(-1)

                    optimizer.zero_grad()
                    output = transformer(en_tensors, src_de)
                    loss = criterion(
                        output.contiguous().view(-1, vocab_tgt_size), tgt_de
                    )

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1)

                    if (
                        step + 1 == total_steps
                        or step % gradient_accumulation_steps == 0
                    ):
                        for param in transformer.parameters():
                            if param.grad is not None:
                                param.grad /= gradient_accumulation_steps
                        optimizer.step()
                        optimizer.zero_grad()

                    total_loss += loss.item()
                    # learning_rates.append(lr_scheduler.get_lr())
                    # lr_scheduler.step()

                    step += 1
                    train_bar.update()

                    if step % checkpoint_steps == 0:
                        torch.save(
                            transformer.state_dict(),
                            os.path.join(
                                experiment_dir, f"{experiment_name}_{step}.pth"
                            ),
                        )

                train_losses.append(total_loss / len(train_dataloader))
                wandb_log(
                    {
                        "train_loss": train_losses[-1],
                        "train_perplexity": np.exp(train_losses[-1]),
                    }
                )
                print()
                print(f"Epoch: {epoch}")
                print(f"Train Loss: [{total_loss=:.3f}] {train_losses[-1]:.3f}")
                print(f"Perplexity: {np.exp(train_losses[-1]):.3f}")
                print()

                # val set
                if (epoch - 1) % validation_epochs == 0:
                    total_loss = 0.0
                    transformer.eval()
                    with torch.no_grad():
                        with tqdm(
                            total=len(val_dataloader), desc="Validation"
                        ) as valbar:
                            for i, (en_tensors, de_tensors) in enumerate(
                                val_dataloader
                            ):
                                en_tensors = en_tensors.to(device="cuda")
                                de_tensors = de_tensors.to(device="cuda")
                                src_de = de_tensors[:, :-1]
                                tgt_de = de_tensors[:, 1:].contiguous().view(-1)

                                output = transformer(en_tensors, src_de)
                                loss = criterion(
                                    output.contiguous().view(-1, vocab_tgt_size), tgt_de
                                )
                                total_loss += loss.item()
                                valbar.update()

                    val_losses.append(total_loss / len(val_dataloader))
                    wandb_log(
                        {
                            "val_loss": val_losses[-1],
                            "val_perplexity": np.exp(val_losses[-1]),
                        }
                    )
                    print()
                    print(f"Validation Loss: [{total_loss=:.3f}] {val_losses[-1]:.3f}")
                    print(f"Perplexity: {np.exp(val_losses[-1]):.3f}")
                    print()

                    print("Running inference on validation set")
                    tgt_tokens = tokenizer_de.decode_batch(
                        de_tensors[:5, 1:].cpu().numpy(), skip_special_tokens=False
                    )
                    output_tokens = tokenizer_de.decode_batch(
                        output[:5].argmax(dim=-1).cpu().numpy(),
                        skip_special_tokens=False,
                    )

                    for tgt, out in zip(tgt_tokens, output_tokens):
                        print(f"   target: {tgt}")
                        print(f"generated: {out}")
                        print()

                total_loss = 0.0
                transformer.eval()
                with torch.no_grad():
                    with tqdm(total=len(test_dataloader), desc="Testing") as testbar:
                        for i, (en_tensors, de_tensors) in enumerate(test_dataloader):
                            en_tensors = en_tensors.to(device="cuda")
                            de_tensors = de_tensors.to(device="cuda")
                            src_de = de_tensors[:, :-1]
                            tgt_de = de_tensors[:, 1:].contiguous().view(-1)

                            output = transformer(en_tensors, src_de)
                            loss = criterion(
                                output.contiguous().view(-1, vocab_tgt_size), tgt_de
                            )
                            total_loss += loss.item()
                            testbar.update()

                test_losses.append(total_loss / len(test_dataloader))
                wandb_log(
                    {
                        "test_loss": test_losses[-1],
                        "test_perplexity": np.exp(test_losses[-1]),
                    }
                )
                print()
                print(f"Test Loss: [{total_loss=:.3f}] {test_losses[-1]:.3f}")
                print(f"Perplexity: {np.exp(test_losses[-1]):.3f}")

        with open(os.path.join(experiment_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        with open(os.path.join(experiment_dir, f"train.json"), "w") as f:
            json.dump(
                {
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "test_losses": test_losses,
                    "learning_rates": learning_rates,
                },
                f,
                indent=4,
            )

        torch.save(
            transformer.state_dict(),
            os.path.join(experiment_dir, f"{experiment_name}_final.pth"),
        )

        tokenizer_en.save(os.path.join(experiment_dir, "tokenizer_en.json"))
        tokenizer_de.save(os.path.join(experiment_dir, "tokenizer_de.json"))

    def inference(
        self,
        checkpoint_path: str,
        experiment_name: str,
        input: Union[str, List[str]],
        top_k: int = -1,
        top_p: float = -1.0,
        temperature: float = 1.0,
        sample: bool = False,
        max_length: int = 100,
    ) -> None:
        if isinstance(input, str):
            input = [input]

        experiment_dir = os.path.join(checkpoint_path, experiment_name)
        with open(os.path.join(experiment_dir, "config.json"), "r") as f:
            config = json.load(f)

        # read model
        transformer = EncoderDecoderTransformer(
            num_encoder_layers=config["num_encoder_layers"],
            num_decoder_layers=config["num_decoder_layers"],
            vocab_src_size=config["vocab_src_size"],
            vocab_tgt_size=config["vocab_tgt_size"],
            pad_src_idx=config["pad_src_idx"],
            pad_tgt_idx=config["pad_tgt_idx"],
            embedding_dim=config["embedding_dim"],
            query_key_dim=config["query_key_dim"],
            value_dim=config["value_dim"],
            num_heads=config["num_heads"],
            ffn_hidden_dim=config["ffn_hidden_dim"],
            ffn_activation=config["ffn_activation"],
            use_kan_bias=config["use_kan_bias"],
            use_pffn_bias=config["use_pffn_bias"],
            use_final_linear_bias=config["use_final_linear_bias"],
            dropout_rate=config["dropout_rate"],
            max_length=max_length,
        ).to(device="cuda")

        transformer.load_state_dict(
            torch.load(os.path.join(experiment_dir, f"{experiment_name}_final.pth")),
            strict=False,
        )

        tokenizer_en = Tokenizer.from_file(
            os.path.join(experiment_dir, "tokenizer_en.json")
        )
        tokenizer_de = Tokenizer.from_file(
            os.path.join(experiment_dir, "tokenizer_de.json")
        )

        sos_token_idx = tokenizer_en.token_to_id("<sos>")
        eos_token_idx = tokenizer_en.token_to_id("<eos>")

        transformer.eval()
        with torch.no_grad():
            for i, sentence in enumerate(input):
                en_tokens = (
                    [sos_token_idx]
                    + tokenizer_en.encode(sentence.lower()).ids
                    + [eos_token_idx]
                )
                en_tensors = torch.tensor([en_tokens], dtype=torch.long).to(
                    device="cuda"
                )

                de_tokens = [sos_token_idx]
                memory = transformer.encode(en_tensors)

                for _ in range(max_length - 1):
                    de_tensors = torch.tensor([de_tokens], dtype=torch.long).to(
                        device="cuda"
                    )
                    logits = transformer.decode(en_tensors, de_tensors, memory)
                    logits /= temperature

                    if top_k > 0:
                        v, _ = torch.topk(logits[:, -1, :], top_k, dim=-1)
                        logits[logits < v[:, -1].unsqueeze(0)] = -float("inf")

                    if top_p > 0.0:
                        sorted_logits, sorted_indices = torch.sort(
                            logits[:, -1, :], descending=True
                        )
                        cumulative_probs = torch.cumsum(
                            torch.softmax(sorted_logits, dim=-1), dim=-1
                        )
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                            ..., :-1
                        ].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        logits[:, -1, indices_to_remove] = -float("inf")

                    probs = torch.softmax(logits[:, -1, :], dim=-1)

                    if sample:
                        pred_token = torch.multinomial(probs, num_samples=1)
                    else:
                        _, pred_token = torch.topk(probs, k=1, dim=-1)

                    de_tokens.append(pred_token.item())

                    if pred_token.item() == eos_token_idx:
                        break

                output = tokenizer_de.decode(de_tokens, skip_special_tokens=False)
                print(f"Input: {sentence}")
                print(f"Output: {output}")
                print(f"Generated token indices: {de_tokens}")
                print()

    def visualize_positional_encoding(
        self,
        embedding_dim: int = 64,
        max_length: int = 64,
        *,
        save: bool = False,
        output_path: str = "pe.png",
    ) -> None:
        r"""Visualize positional encoding used in the paper.

        Args:
            embedding_dim:
                The dimensionality of vector space embeddings (`d_model` in the paper)
            max_length:
                Maximum sequence length of tokens
            save:
                Whether or not to save the plot
            output_path:
                Path to file where plot is to be saved
        """

        position_encoder = PositionalEncoding(embedding_dim, max_length)
        pe: np.ndarray = position_encoder.pe.detach().numpy()

        figsize = (
            min(embedding_dim // 8, 20),
            min(max_length // 8, 20),
        )
        plt.figure(figsize=figsize)
        plt.imshow(pe, cmap="magma")
        plt.xlabel("Embedding size (d_model)", fontsize=20)
        plt.ylabel("Sequence length", fontsize=20)
        plt.title("Positional Encoding", fontsize=20)

        if save:
            plt.savefig(output_path)

        plt.show()


if __name__ == "__main__":
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire(CLI())
