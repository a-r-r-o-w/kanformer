from typing import Optional

import torch
import torch.nn as nn

from ..config import ModelType
from ..model_utils import get_model_cls
from .attention import MultiHeadAttention


T = torch.FloatTensor


def get_activation(name: str, **kwargs) -> nn.Module:
    if name == "relu":
        return nn.ReLU(**kwargs)
    elif name == "gelu":
        return nn.GELU(**kwargs)
    elif name == "silu" or name == "swish":
        return nn.SiLU(**kwargs)
    elif name == "leaky_relu":
        return nn.LeakyReLU(**kwargs)
    raise ValueError(f"{name} is not a supported activation")


class PositionwiseFeedForward(nn.Module):
    r"""Position-wise Feed-forward Network (section 3.3 in paper).

    Args:
    """

    def __init__(
        self,
        in_out_dim: int,
        hidden_dim: int,
        activation: str = "relu",
        use_bias_1: bool = True,
        use_bias_2: bool = True,
        use_kan_bias: bool = True,
        model_type: ModelType = ModelType.MLP,
    ) -> None:
        super().__init__()

        cls = get_model_cls(model_type, use_kan_bias)
        self.linear_1 = cls(in_out_dim, hidden_dim, bias=use_bias_1)
        self.linear_2 = cls(hidden_dim, in_out_dim, bias=use_bias_2)
        self.activation = get_activation(activation)

    def forward(self, x: T) -> T:
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        return x


class PositionalEncoding(nn.Module):
    r"""Positional Encoding (Section 3.5 of paper).

    Args:
    """

    def __init__(
        self,
        embedding_dim: int,  # `d_model` in paper
        max_length: int = 10000,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.max_length = max_length

        two_i = torch.arange(0, embedding_dim, 2, dtype=torch.float32)
        numerator = torch.arange(0, max_length, dtype=torch.float32).unsqueeze(1)
        denominator = 10000.0 ** (two_i / embedding_dim)

        self.pe = torch.zeros(max_length, embedding_dim)
        self.pe[:, 0::2] = torch.sin(numerator / denominator)
        self.pe[:, 1::2] = torch.cos(numerator / denominator)

    def forward(self, x: T) -> T:
        seq_length = x.size(1)
        self.pe = self.pe.to(x.device)
        return x + self.pe[:seq_length, :]


class EncoderBlock(nn.Module):
    r"""A single encoder block as shown in Figure 1 of the paper.

    Args:
    """

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
        use_ffn_bias_1: bool = True,
        use_ffn_bias_2: bool = True,
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
            use_bias_1=use_ffn_bias_1,
            use_bias_2=use_ffn_bias_2,
            use_kan_bias=use_kan_bias,
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
    """

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
        use_ffn_bias_1: bool = True,
        use_ffn_bias_2: bool = True,
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
            use_bias_1=use_ffn_bias_1,
            use_bias_2=use_ffn_bias_2,
            use_kan_bias=use_kan_bias,
            model_type=model_type,
        )
        self.dropout3 = nn.Dropout(dropout_rate)
        self.norm3 = nn.LayerNorm(embedding_dim)

    def forward(
        self, x: T, enc_x: T, mask: Optional[T] = None, dec_enc_mask: Optional[T] = None
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


class EncoderDecoderTransformer(nn.Module):
    r"""Transformer - the model proposed in "Attention Is All You Need".
    Paper: https://arxiv.org/abs/1706.03762
    Original Code: https://github.com/tensorflow/tensor2tensor

    This implementation is an almost close replica of the original transformer model. I've made assumptions
    about some details that are not clear from reading the paper and so the overall number of parameters may
    or may not match completely with the model developed in the original codebase.

    Args:
    """

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
        use_ffn_bias_1: bool = True,
        use_ffn_bias_2: bool = True,
        use_final_linear_bias: bool = False,
        dropout_rate: float = 0.1,
        max_length: int = 10000,
        model_type: ModelType = ModelType.MLP,
    ) -> None:
        super().__init__()

        self.pad_src_idx = pad_src_idx
        self.pad_tgt_idx = pad_tgt_idx

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
                    use_ffn_bias_1=use_ffn_bias_1,
                    use_ffn_bias_2=use_ffn_bias_2,
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
                    use_ffn_bias_1=use_ffn_bias_1,
                    use_ffn_bias_2=use_ffn_bias_2,
                    dropout_rate=dropout_rate,
                    model_type=model_type,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        cls = get_model_cls(model_type, use_kan_bias)
        self.linear = cls(embedding_dim, vocab_tgt_size, bias=use_final_linear_bias)

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
        src_mask = self._get_src_mask(src_x, self.pad_src_idx)

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
        src_mask = self._get_src_mask(src_x, self.pad_src_idx)
        tgt_mask = self._get_tgt_mask(tgt_x, self.pad_tgt_idx)

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


class EncoderTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        vocab_src_size: int,
        pad_src_idx: int,
        embedding_dim: int,  # `d_model` in paper
        query_key_dim: int,  # `d_k` in paper
        value_dim: int,  # `d_v` in paper
        num_heads: int,  # `h` in paper
        ffn_hidden_dim: int,
        ffn_activation: str = "relu",
        use_kan_bias: bool = True,
        use_ffn_bias_1: bool = True,
        use_ffn_bias_2: bool = True,
        use_final_linear_bias: bool = False,
        dropout_rate: float = 0.1,
        max_length: int = 10000,
        model_type: ModelType = ModelType.MLP,
    ) -> None:
        super().__init__()

        self.pad_src_idx = pad_src_idx

        self.pe = PositionalEncoding(
            embedding_dim=embedding_dim,
            max_length=max_length,
        )

        self.src_emb = nn.Embedding(vocab_src_size, embedding_dim)
        self.src_dropout = nn.Dropout(dropout_rate)

        scale = torch.sqrt(torch.tensor(embedding_dim, dtype=torch.float32))
        self.register_buffer("scale", scale)
        self.scale: T

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
                    use_ffn_bias_1=use_ffn_bias_1,
                    use_ffn_bias_2=use_ffn_bias_2,
                    dropout_rate=dropout_rate,
                    model_type=model_type,
                )
                for _ in range(num_encoder_layers)
            ]
        )

    def _get_src_mask(self, x: T, pad_idx: int) -> torch.BoolTensor:
        pad_mask = (x != pad_idx).bool().unsqueeze(1).unsqueeze(2)
        return pad_mask

    def forward(self, x: T) -> T:
        mask = self._get_src_mask(x, self.pad_src_idx)
        x = self.src_emb(x)
        x = x * self.scale
        x = self.pe(x)
        x = self.src_dropout(x)
        for block in self.encoder_blocks:
            x = block(x, mask)
        return x


class DecoderTransformer(nn.Module):
    def __init__(
        self,
        num_decoder_layers: int,
        vocab_tgt_size: int,
        pad_tgt_idx: int,
        embedding_dim: int,  # `d_model` in paper
        query_key_dim: int,  # `d_k` in paper
        value_dim: int,  # `d_v` in paper
        num_heads: int,  # `h` in paper
        ffn_hidden_dim: int,
        ffn_activation: str = "relu",
        use_kan_bias: bool = True,
        use_ffn_bias_1: bool = True,
        use_ffn_bias_2: bool = True,
        use_final_linear_bias: bool = False,
        dropout_rate: float = 0.1,
        max_length: int = 10000,
        model_type: ModelType = ModelType.MLP,
    ) -> None:
        super().__init__()

        self.pad_tgt_idx = pad_tgt_idx

        self.pe = PositionalEncoding(
            embedding_dim=embedding_dim,
            max_length=max_length,
        )

        self.tgt_emb = nn.Embedding(vocab_tgt_size, embedding_dim)
        self.tgt_dropout = nn.Dropout(dropout_rate)

        scale = torch.sqrt(torch.tensor(embedding_dim, dtype=torch.float32))
        self.register_buffer("scale", scale)
        self.scale: T

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
                    use_ffn_bias_1=use_ffn_bias_1,
                    use_ffn_bias_2=use_ffn_bias_2,
                    dropout_rate=dropout_rate,
                    model_type=model_type,
                    use_encoder_attn=False,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        cls = get_model_cls(model_type, use_kan_bias)
        self.linear = cls(embedding_dim, vocab_tgt_size, bias=use_final_linear_bias)

    def _get_tgt_mask(self, x: T, pad_idx: int) -> torch.BoolTensor:
        seq_length = x.size(1)
        pad_mask = (x != pad_idx).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.ones((1, seq_length, seq_length), device=x.device)
        causal_mask = torch.tril(causal_mask).bool()
        mask = pad_mask & causal_mask
        return mask

    def forward(self, x: T) -> T:
        mask = self._get_tgt_mask(x, self.pad_tgt_idx)
        x = self.tgt_emb(x)
        x = x * self.scale
        x = self.pe(x)
        x = self.tgt_dropout(x)
        for block in self.decoder_blocks:
            x = block(x, None, mask)
        x = self.linear(x)
        return x
