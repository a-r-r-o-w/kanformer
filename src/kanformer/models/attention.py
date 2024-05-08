import importlib
from typing import Optional

import torch
import torch.nn as nn

from ..config import ModelType
from ..model_utils import get_model_cls


T = torch.FloatTensor


class ScaledDotProductAttention(nn.Module):
    r"""ScaledDotProductAttention (Section 3.2.1 of paper).

    Args:
        embedding_size (`int`):
        temperature (`float`, *optional*):
    """

    def __init__(self, query_key_dim: int) -> None:
        super().__init__()

        # In the original paper, product of query and key_T are normalized by square root of
        # embedding size. Here, we allow for normalizing with a temperature value too. If
        # temperature is not `None`, it will be used. Otherwise, square root of `embedding_size`
        # will be used.
        self.query_key_dim = query_key_dim

        scale = torch.sqrt(torch.FloatTensor([query_key_dim]))
        self.register_buffer("scale", scale)
        self.scale: T

        self.softmax = nn.Softmax(dim=3)

    def forward(self, query: T, key: T, value: T, mask: Optional[T] = None) -> T:
        # 1. MatMul
        #  query: [batch_size, num_heads, seq_length, query_key_dim]
        #  key_T: [batch_size, num_heads, query_key_dim, seq_length]
        # result: [batch_size, num_heads, seq_length, seq_length]
        key_T = key.transpose(2, 3)
        x = torch.matmul(query, key_T)

        # 2. Scale
        x = x / self.scale

        # 3. Mask
        if mask is not None:
            x = x.masked_fill(mask == False, value=-1e9)

        # 4. SoftMax
        x = self.softmax(x)

        # 5. MatMul
        #      x: [batch_size, num_heads, seq_length, seq_length]
        #  value: [batch_size, num_heads, seq_length, value_dim]
        # result: [batch_size, num_heads, seq_length, value_dim]
        x = torch.matmul(x, value)

        return x


class MultiHeadAttention(nn.Module):
    r"""Multi-Head Attention (Section 3.2.2 of paper).

    Args:
    """

    def __init__(
        self,
        embedding_size: int,  # `d_model` in paper
        query_key_dim: int,  # `d_k` in paper
        value_dim: int,  # `d_v` in paper
        num_heads: int,  # `h` in paper
        use_final_linear_mha_bias: bool = False,
        use_kan_bias: bool = False,
        model_type: ModelType = ModelType.MLP,
    ) -> None:
        super().__init__()

        self.embedding_size = embedding_size
        self.query_key_dim = query_key_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.query_key_dim_per_head = query_key_dim // num_heads
        self.value_dim_per_head = value_dim // num_heads

        cls = get_model_cls(model_type, use_kan_bias)
        self.q_proj = cls(self.embedding_size, self.query_key_dim, bias=False)
        self.k_proj = cls(self.embedding_size, self.query_key_dim, bias=False)
        self.v_proj = cls(self.embedding_size, self.value_dim, bias=False)
        self.attn = ScaledDotProductAttention(query_key_dim)
        self.ff_proj = cls(
            self.value_dim, self.embedding_size, bias=use_final_linear_mha_bias
        )

    def forward(self, query: T, key: T, value: T, mask: Optional[T] = None) -> T:
        # 1. Linear
        q_proj: T = self.q_proj(query)
        k_proj: T = self.k_proj(key)
        v_proj: T = self.v_proj(value)

        # 2. Scaled Dot Product Attention
        batch_size, seq_length, _ = q_proj.shape
        q_proj = q_proj.view(
            batch_size, -1, self.num_heads, self.query_key_dim_per_head
        )
        k_proj = k_proj.view(
            batch_size, -1, self.num_heads, self.query_key_dim_per_head
        )
        v_proj = v_proj.view(batch_size, -1, self.num_heads, self.value_dim_per_head)

        q_proj = q_proj.transpose(1, 2)
        k_proj = k_proj.transpose(1, 2)
        v_proj = v_proj.transpose(1, 2)

        x: T = self.attn(q_proj, k_proj, v_proj, mask)

        # 3. Concat
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, seq_length, self.value_dim)

        # 4. Linear
        x = self.ff_proj(x)
        return x
