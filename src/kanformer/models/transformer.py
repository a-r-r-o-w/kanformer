import functools
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..config import ModelType, ModelConfig
from ..config_utils import register_to_config
from ..model_utils import get_linear_cls
from .attention import MultiHeadAttention


T = torch.FloatTensor


def get_activation(name: str, **kwargs) -> nn.Module:
    if name == "relu" or name == "reglu":
        return nn.ReLU(**kwargs)
    elif name == "gelu" or name == "geglu":
        return nn.GELU(**kwargs)
    elif name == "silu" or name == "swish" or name == "swiglu":
        return nn.SiLU(**kwargs)
    elif name == "sigmoid" or name == "glu":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh(**kwargs)
    elif name == "elu":
        return nn.ELU(**kwargs)
    elif name == "leaky_relu":
        return nn.LeakyReLU(**kwargs)
    raise ValueError(f"{name} is not a supported activation")


def _is_glu_activation(activation: str) -> bool:
    return activation in ["glu", "reglu", "geglu", "swiglu"]


class PositionwiseFeedForward(nn.Module):
    r"""Position-wise Feed-forward Network (section 3.3 in paper).

    Args:
        in_out_dim (`int`):
            The dimension of the input and output vectors.
        hidden_dim (`int`):
            The dimension of the hidden layer.
        activation (`str`, optional):
            The activation function to use. Defaults to `"relu"`.
        dropout_rate (`float`, optional):
            The dropout rate to use. Defaults to `0.1`.
        use_bias (`bool`, optional):
            Whether to use bias in the linear layers. Defaults to `True`.
        model_type (`ModelType`, optional):
            The type of model to use. Defaults to `ModelType.MLP`.
    """

    @register_to_config
    def __init__(
        self,
        in_out_dim: int,
        hidden_dim: int,
        activation: str = "relu",
        dropout_rate: float = 0.1,
        use_bias: bool = True,
        model_type: ModelType = ModelType.MLP,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__()

        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.use_bias = use_bias

        cls = get_linear_cls(model_type, use_bias, **kwargs)
        self.in_proj = cls(in_out_dim, hidden_dim, bias=use_bias)
        self.out_proj = cls(hidden_dim, in_out_dim, bias=use_bias)
        self.gates = (
            cls(in_out_dim, hidden_dim, bias=use_bias)
            if _is_glu_activation(activation)
            else None
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.act = get_activation(activation)

    def forward(self, x: T) -> T:
        if self.gates is None:
            in_proj = self.in_proj(x)
            x = self.act(in_proj)
        else:
            in_proj = self.in_proj(x)
            gate = self.gates(x)
            x = self.act(gate) * in_proj

        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class PositionalEncoding(nn.Module):
    r"""Positional Encoding (Section 3.5 of paper).

    Args:
        embedding_dim (`int`):
            The dimension of the embedding space (d_model in paper).
        max_length (`int`, optional):
            The maximum length of the sequence. Defaults to `10000`.
    """

    @register_to_config
    def __init__(
        self,
        embedding_dim: int,  # `d_model` in paper
        max_length: int = 10000,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.max_length = max_length

        two_i = torch.arange(
            0, embedding_dim, 2, dtype=torch.float32, requires_grad=False
        )
        numerator = torch.arange(
            0, max_length, dtype=torch.float32, requires_grad=False
        ).unsqueeze(1)
        denominator = 10000.0 ** (two_i / embedding_dim)

        self.pe = torch.zeros(max_length, embedding_dim, requires_grad=False)
        self.pe[:, 0::2] = torch.sin(numerator / denominator)
        self.pe[:, 1::2] = torch.cos(numerator / denominator)

    def forward(self, x: T) -> T:
        seq_length = x.size(1)
        self.pe = self.pe.to(x.device)
        return x + self.pe[:seq_length, :]


class EncoderBlock(nn.Module):
    r"""A single encoder block as shown in Figure 1 of the paper.

    Args:
        embedding_dim (`int`):
            The dimension of the embedding space (d_model in paper).
        query_key_dim (`int`):
            The dimension of the query and key vectors (d_k in paper).
        value_dim (`int`):
            The dimension of the value vectors (d_v in paper).
        num_heads (`int`):
            The number of attention heads (h in paper).
        ffn_hidden_dim (`int`):
            The dimension of the hidden layer in the feed-forward network.
        ffn_activation (`str`, optional):
            The activation function to use in the feed-forward network. Defaults to `"relu"`.
        use_kan_bias (`bool`, optional):
            Whether to use KAN bias. Defaults to `True`.
        use_final_linear_mha_bias (`bool`, optional):
            Whether to use bias in the final linear layer of multi-head attention. Defaults to `False`.
        use_pffn_bias (`bool`, optional):
            Whether to use bias in the feed-forward network. Defaults to `True`.
        dropout_rate (`float`, optional):
            The dropout rate to use. Defaults to `0.1`.
        model_type (`ModelType`, optional):
            The type of model to use. Defaults to `ModelType.MLP`.
    """

    @register_to_config
    def __init__(
        self,
        embedding_dim: int,  # `d_model` in paper
        query_key_dim: int,  # `d_k` in paper
        value_dim: int,  # `d_v` in paper
        num_heads: int,  # `h` in paper
        ffn_hidden_dim: int,
        ffn_activation: str = "relu",
        use_kan_bias: bool = True,
        use_final_linear_mha_bias: bool = False,
        use_pffn_bias: bool = True,
        dropout_rate: float = 0.1,
        model_type: ModelType = ModelType.MLP,
    ) -> None:
        super().__init__()

        self.mha = MultiHeadAttention(
            embedding_dim=embedding_dim,
            query_key_dim=query_key_dim,
            value_dim=value_dim,
            num_heads=num_heads,
            use_final_linear_mha_bias=use_final_linear_mha_bias,
            use_kan_bias=use_kan_bias,
            model_type=model_type,
        )

        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.pffn = PositionwiseFeedForward(
            in_out_dim=embedding_dim,
            hidden_dim=ffn_hidden_dim,
            activation=ffn_activation,
            dropout_rate=dropout_rate,
            use_bias=use_pffn_bias,
            model_type=model_type,
        )
        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x: T, mask: Optional[T] = None) -> T:
        residual = x
        x = self.mha(x, x, x, mask)
        x = self.norm1(residual + self.dropout1(x))

        residual = x
        x = self.pffn(x)
        x = self.norm2(residual + self.dropout2(x))
        return x


class DecoderBlock(nn.Module):
    r"""A single decoder block as shown in Figure 1 of the paper.

    Args:
        embedding_dim (`int`):
            The dimension of the embedding space (d_model in paper).
        query_key_dim (`int`):
            The dimension of the query and key vectors (d_k in paper).
        value_dim (`int`):
            The dimension of the value vectors (d_v in paper).
        num_heads (`int`):
            The number of attention heads (h in paper).
        ffn_hidden_dim (`int`):
            The dimension of the hidden layer in the feed-forward network.
        ffn_activation (`str`, optional):
            The activation function to use in the feed-forward network. Defaults to `"relu"`.
        use_kan_bias (`bool`, optional):
            Whether to use KAN bias. Defaults to `True`.
        use_final_linear_mha_bias (`bool`, optional):
            Whether to use bias in the final linear layer of multi-head attention. Defaults to `False`.
        use_pffn_bias (`bool`, optional):
            Whether to use bias in the feed-forward network. Defaults to `True`.
        dropout_rate (`float`, optional):
            The dropout rate to use. Defaults to `0.1`.
        model_type (`ModelType`, optional):
            The type of model to use. Defaults to `ModelType.MLP`.
        use_encoder_attn (`bool`, optional):
            Whether to use encoder hidden states and perform attention with them. Defaults to `True`.
    """

    @register_to_config
    def __init__(
        self,
        embedding_dim: int,  # `d_model` in paper
        query_key_dim: int,  # `d_k` in paper
        value_dim: int,  # `d_v` in paper
        num_heads: int,  # `h` in paper
        ffn_hidden_dim: int,
        ffn_activation: str = "relu",
        use_kan_bias: bool = True,
        use_final_linear_mha_bias: bool = False,
        use_pffn_bias: bool = True,
        dropout_rate: float = 0.1,
        model_type: ModelType = ModelType.MLP,
        use_encoder_attn: bool = True,
    ) -> None:
        super().__init__()

        self.use_encoder_attn = use_encoder_attn

        self.mha1 = MultiHeadAttention(
            embedding_dim=embedding_dim,
            query_key_dim=query_key_dim,
            value_dim=value_dim,
            num_heads=num_heads,
            use_final_linear_mha_bias=use_final_linear_mha_bias,
            use_kan_bias=use_kan_bias,
            model_type=model_type,
        )
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm(embedding_dim)

        if use_encoder_attn:
            self.mha2 = MultiHeadAttention(
                embedding_dim=embedding_dim,
                query_key_dim=query_key_dim,
                value_dim=value_dim,
                num_heads=num_heads,
                use_final_linear_mha_bias=use_final_linear_mha_bias,
                use_kan_bias=use_kan_bias,
                model_type=model_type,
            )
            self.dropout2 = nn.Dropout(dropout_rate)
            self.norm2 = nn.LayerNorm(embedding_dim)
        else:
            self.mha2 = None
            self.dropout2 = None
            self.norm2 = None

        self.pffn = PositionwiseFeedForward(
            in_out_dim=embedding_dim,
            hidden_dim=ffn_hidden_dim,
            activation=ffn_activation,
            dropout_rate=dropout_rate,
            use_bias=use_pffn_bias,
            model_type=model_type,
        )
        self.dropout3 = nn.Dropout(dropout_rate)
        self.norm3 = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        x: T,
        enc_x: T = None,
        mask: Optional[T] = None,
        dec_enc_mask: Optional[T] = None,
    ) -> T:
        residual = x
        x = self.mha1(x, x, x, mask)
        x = self.norm1(residual + self.dropout1(x))

        if self.use_encoder_attn:
            residual = x
            x = self.mha2(x, enc_x, enc_x, dec_enc_mask)
            x = self.norm2(residual + self.dropout2(x))

        residual = x
        x = self.pffn(x)
        x = self.norm3(residual + self.dropout3(x))
        return x


class MaskedBlock(nn.Module):
    r"""A single unassuming masked block that can be used in any model configuration. It
    is essentially same as the EncoderBlock and consists of a multi-head attention layer
    followed by a position-wise feed-forward network, and normalization after each sublayer.

    Args:
        embedding_dim (`int`):
            The dimension of the embedding space (d_model in paper).
        query_key_dim (`int`):
            The dimension of the query and key vectors (d_k in paper).
        value_dim (`int`):
            The dimension of the value vectors (d_v in paper).
        num_heads (`int`):
            The number of attention heads (h in paper).
        ffn_hidden_dim (`int`):
            The dimension of the hidden layer in the feed-forward network.
        ffn_activation (`str`, optional):
            The activation function to use in the feed-forward network. Defaults to `"relu"`.
        use_kan_bias (`bool`, optional):
            Whether to use KAN bias. Defaults to `True`.
        use_final_linear_mha_bias (`bool`, optional):
            Whether to use bias in the final linear layer of multi-head attention. Defaults to `False`.
        use_pffn_bias (`bool`, optional):
            Whether to use bias in the feed-forward network. Defaults to `True`.
        dropout_rate (`float`, optional):
            The dropout rate to use. Defaults to `0.1`.
        model_type (`ModelType`, optional):
            The type of model to use. Defaults to `ModelType.MLP`.
    """

    @register_to_config
    def __init__(
        self,
        embedding_dim: int,  # `d_model` in paper
        query_key_dim: int,  # `d_k` in paper
        value_dim: int,  # `d_v` in paper
        num_heads: int,  # `h` in paper
        ffn_hidden_dim: int,
        ffn_activation: str = "relu",
        use_kan_bias: bool = True,
        use_final_linear_mha_bias: bool = False,
        use_pffn_bias: bool = True,
        dropout_rate: float = 0.1,
        model_type: ModelType = ModelType.MLP,
    ) -> None:
        super().__init__()

        self.mha = MultiHeadAttention(
            embedding_dim=embedding_dim,
            query_key_dim=query_key_dim,
            value_dim=value_dim,
            num_heads=num_heads,
            use_final_linear_mha_bias=use_final_linear_mha_bias,
            use_kan_bias=use_kan_bias,
            model_type=model_type,
        )
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.pffn = PositionwiseFeedForward(
            in_out_dim=embedding_dim,
            hidden_dim=ffn_hidden_dim,
            activation=ffn_activation,
            dropout_rate=dropout_rate,
            use_bias=use_pffn_bias,
            model_type=model_type,
        )
        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x: T, mask: Optional[T] = None) -> T:
        residual = x
        x = self.mha(x, x, x, mask)
        x = self.norm1(residual + self.dropout1(x))

        residual = x
        x = self.pffn(x)
        x = self.norm2(residual + self.dropout2(x))
        return x


class TransformerBase(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def from_config(config: Dict[str, Any]) -> "TransformerBase":
        cls_name = config.pop("_cls")
        cls = _cls_name_to_cls_mapping[cls_name]
        return cls(**config)


class TransformerSeq2Seq(TransformerBase):
    r"""Transformer - the model proposed in "Attention Is All You Need".
    Paper: https://arxiv.org/abs/1706.03762
    Original Code: https://github.com/tensorflow/tensor2tensor

    This implementation is an almost close replica of the original transformer model. I've made assumptions
    about some details that are not clear from reading the paper and so the overall number of parameters may
    or may not match completely with the model developed in the original codebase.

    Args:
        num_encoder_layers (`int`):
            The number of encoder blocks to use.
        num_decoder_layers (`int`):
            The number of decoder blocks to use.
        vocab_src_size (`int`):
            The size of the source vocabulary.
        vocab_tgt_size (`int`):
            The size of the target vocabulary.
        pad_src_idx (`int`):
            The index of the padding token in the source vocabulary.
        pad_tgt_idx (`int`):
            The index of the padding token in the target vocabulary.
        embedding_dim (`int`):
            The dimension of the embedding space (d_model in paper).
        query_key_dim (`int`):
            The dimension of the query and key vectors (d_k in paper).
        value_dim (`int`):
            The dimension of the value vectors (d_v in paper).
        num_heads (`int`):
            The number of attention heads (h in paper).
        ffn_hidden_dim (`int`):
            The dimension of the hidden layer in the feed-forward network.
        ffn_activation (`str`, optional):
            The activation function to use in the feed-forward network. Defaults to `"relu"`.
        use_kan_bias (`bool`, optional):
            Whether to use KAN bias. Defaults to `True`.
        use_final_linear_mha_bias (`bool`, optional):
            Whether to use bias in the final linear layer of multi-head attention. Defaults to `False`.
        use_pffn_bias (`bool`, optional):
            Whether to use bias in the feed-forward network. Defaults to `True`.
        dropout_rate (`float`, optional):
            The dropout rate to use. Defaults to `0.1`.
        model_type (`ModelType`, optional):
            The type of model to use. Defaults to `ModelType.MLP`.
    """

    @register_to_config
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        vocab_src_size: int,
        vocab_tgt_size: int,
        pad_src_idx: int,
        pad_tgt_idx: int,
        embedding_dim: int,  # `d_model` in paper
        query_key_dim: int,  # `d_k` in paper
        value_dim: int,  # `d_v` in paper
        num_heads: int,  # `h` in paper
        ffn_hidden_dim: int,
        ffn_activation: str = "relu",
        use_kan_bias: bool = True,
        use_pffn_bias: bool = True,
        use_final_linear_bias: bool = False,
        dropout_rate: float = 0.1,
        max_length: int = 10000,
        model_type: ModelType = ModelType.MLP,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__()

        self.pe = PositionalEncoding(
            embedding_dim=embedding_dim,
            max_length=max_length,
        )

        self.src_emb = nn.Embedding(vocab_src_size, embedding_dim)
        self.tgt_emb = nn.Embedding(vocab_tgt_size, embedding_dim)

        scale = torch.sqrt(torch.tensor(embedding_dim, dtype=torch.float32))
        self.register_buffer("scale", scale)
        self.scale: T

        self.src_dropout = nn.Dropout(dropout_rate)
        self.tgt_dropout = nn.Dropout(dropout_rate)

        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    embedding_dim=embedding_dim,
                    query_key_dim=query_key_dim,
                    value_dim=value_dim,
                    num_heads=num_heads,
                    ffn_hidden_dim=ffn_hidden_dim,
                    ffn_activation=ffn_activation,
                    use_kan_bias=use_kan_bias,
                    use_final_linear_mha_bias=use_final_linear_bias,
                    use_pffn_bias=use_pffn_bias,
                    dropout_rate=dropout_rate,
                    model_type=model_type,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    embedding_dim=embedding_dim,
                    query_key_dim=query_key_dim,
                    value_dim=value_dim,
                    num_heads=num_heads,
                    ffn_hidden_dim=ffn_hidden_dim,
                    ffn_activation=ffn_activation,
                    use_kan_bias=use_kan_bias,
                    use_final_linear_mha_bias=use_final_linear_bias,
                    use_pffn_bias=use_pffn_bias,
                    dropout_rate=dropout_rate,
                    model_type=model_type,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        cls = get_linear_cls(model_type, use_kan_bias, **kwargs)
        self.linear = cls(embedding_dim, vocab_tgt_size, bias=use_final_linear_bias)

        if isinstance(self.linear, nn.Linear):
            self.linear.weight = self.tgt_emb.weight

    def _get_src_mask(self, x: T, pad_idx: int) -> torch.BoolTensor:
        r"""Helper utility to get mask for padded tokens. Padded tokens should not be paid attention."""
        pad_mask = (x != pad_idx).bool().unsqueeze(1).unsqueeze(2)
        return pad_mask

    def _get_tgt_mask(self, x: T, pad_idx: int) -> torch.BoolTensor:
        r"""Helper utility to get mask for decoder. The decoder should not pay attention to future tokens.

        This returns a tensor that looks like:
            [
                [1, 0, 0, ...],
                [1, 1, 0, ...],
                [1, 1, 1, ...],
                ...
            ]
        """
        # batch_size = x.size(0)
        seq_length = x.size(1)
        pad_mask = (x != pad_idx).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.ones((1, seq_length, seq_length), device=x.device)
        causal_mask = torch.tril(causal_mask).bool()
        mask = pad_mask & causal_mask
        return mask

    def forward(self, src_x: T, tgt_x: T) -> T:
        memory = self.encode(src_x)
        tgt_x = self.decode(src_x, tgt_x, memory)
        return tgt_x

    def encode(self, src_x: T) -> T:
        # 1. Prepare masks for encoder
        src_mask = self._get_src_mask(src_x, self.config.pad_src_idx)

        # 2. Convert tokens to embeddings
        src_x = self.src_emb(src_x)

        # 3. Apply positional encoding
        src_x = src_x * self.scale
        src_x = self.pe(src_x)

        # 4. Regularization after embed as described in section 5.4 of the paper
        src_x = self.src_dropout(src_x)

        # 5. Forward pass through encoder
        for block in self.encoder_blocks:
            src_x = block(src_x, src_mask)

        return src_x

    def decode(self, src_x: T, tgt_x: T, memory: T) -> T:
        # 1. Prepare masks for decoder
        src_mask = self._get_src_mask(src_x, self.config.pad_src_idx)
        tgt_mask = self._get_tgt_mask(tgt_x, self.config.pad_tgt_idx)

        # 2. Convert tokens to embeddings
        tgt_x = self.tgt_emb(tgt_x)

        # 3. Apply positional encoding
        tgt_x = tgt_x * self.scale
        tgt_x = self.pe(tgt_x)

        # 4. Regularization after embed as described in section 5.4 of the paper
        tgt_x = self.tgt_dropout(tgt_x)

        # 5. Forward pass through decoder
        for block in self.decoder_blocks:
            tgt_x = block(tgt_x, memory, tgt_mask, src_mask)

        # 6. Linear projection to get probabilities for output tokens using softmax
        tgt_x = self.linear(tgt_x)

        return tgt_x


class TransformerTextGeneration(TransformerBase):
    r"""A simple text generation model using transformer architecture.

    Args:
        num_layers (`int`):
            The number of transformer blocks to use.
        vocab_size (`int`):
            The size of the vocabulary.
        embedding_dim (`int`):
            The dimension of the embedding space (d_model in paper).
        query_key_dim (`int`):
            The dimension of the query and key vectors (d_k in paper).
        value_dim (`int`):
            The dimension of the value vectors (d_v in paper).
        num_heads (`int`):
            The number of attention heads (h in paper).
        ffn_hidden_dim (`int`):
            The dimension of the hidden layer in the feed-forward network.
        ffn_activation (`str`, optional):
            The activation function to use in the feed-forward network. Defaults to `"relu"`.
        use_kan_bias (`bool`, optional):
            Whether to use KAN bias. Defaults to `True`.
        use_final_linear_bias (`bool`, optional):
            Whether to use bias in the final linear layer of multi-head attention. Defaults to `False`.
        use_pffn_bias (`bool`, optional):
            Whether to use bias in the feed-forward network. Defaults to `True`.
        dropout_rate (`float`, optional):
            The dropout rate to use. Defaults to `0.1`.
        max_length (`int`, optional):
            The maximum length of the sequence. Defaults to `10000`.
        model_type (`ModelType`, optional):
            The type of model to use. Defaults to `ModelType.MLP`.
    """

    _tie_weights = [["emb", "linear"]]

    @register_to_config
    def __init__(
        self,
        num_layers: int,
        vocab_size: int,
        embedding_dim: int,  # `d_model` in paper
        query_key_dim: int,  # `d_k` in paper
        value_dim: int,  # `d_v` in paper
        num_heads: int,  # `h` in paper
        ffn_hidden_dim: int,
        ffn_activation: str = "relu",
        use_kan_bias: bool = True,
        use_pffn_bias: bool = True,
        use_final_linear_bias: bool = False,
        dropout_rate: float = 0.1,
        max_length: int = 10000,
        model_type: ModelType = ModelType.MLP,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__()

        self.scale = embedding_dim**0.5
        self.pe = PositionalEncoding(
            embedding_dim=embedding_dim,
            max_length=max_length,
        )
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList(
            [
                MaskedBlock(
                    embedding_dim=embedding_dim,
                    query_key_dim=query_key_dim,
                    value_dim=value_dim,
                    num_heads=num_heads,
                    ffn_hidden_dim=ffn_hidden_dim,
                    ffn_activation=ffn_activation,
                    use_kan_bias=use_kan_bias,
                    use_final_linear_mha_bias=use_final_linear_bias,
                    use_pffn_bias=use_pffn_bias,
                    dropout_rate=dropout_rate,
                    model_type=model_type,
                )
                for _ in range(num_layers)
            ]
        )
        cls = get_linear_cls(model_type, use_kan_bias, **kwargs)
        self.linear = cls(embedding_dim, vocab_size, bias=use_final_linear_bias)

        if isinstance(self.linear, nn.Linear):
            self.linear.weight = self.emb.weight

    def forward(self, x: T, mask: Optional[T] = None) -> T:
        x = self.emb(x) * self.scale
        x = self.pe(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, mask)
        x = self.linear(x)
        return x


_cls_name_to_cls_mapping = {
    TransformerSeq2Seq.__name__: TransformerSeq2Seq,
    TransformerTextGeneration.__name__: TransformerTextGeneration,
}
