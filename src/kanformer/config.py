from enum import Enum


class ModelType(str, Enum):
    MLP = "mlp"
    KAN_ORIGINAL = "kan_original"
    KAN_EFFICIENT = "kan_efficient"
    KAN_CHEBYSHEV = "kan_chebyshev"
    KAN_FAST = "kan_fast"


class ModelConfig(dict, Enum):
    # 124M parameters with ModelType.MLP
    GPT2 = {
        "_cls": "TransformerTextGeneration",
        "num_layers": 12,
        "vocab_size": 50257,
        "embedding_dim": 768,
        "query_key_dim": 768,
        "value_dim": 768,
        "num_heads": 12,
        "ffn_hidden_dim": 4 * 768,
        "ffn_activation": "gelu",
        "use_kan_bias": False,
        "use_pffn_bias": True,
        "use_final_linear_bias": True,
        "dropout": 0.1,
        "max_length": None,  # Must be set while initializing the model
        "model_type": None,  # Must be set while initializing the model
    }

    # 353M parameters with ModelType.MLP
    GPT2_MEDIUM = {
        "_cls": "TransformerTextGeneration",
        "num_layers": 24,
        "vocab_size": 50257,
        "embedding_dim": 1024,
        "query_key_dim": 1024,
        "value_dim": 1024,
        "num_heads": 16,
        "ffn_hidden_dim": 4 * 1024,
        "ffn_activation": "gelu",
        "use_kan_bias": False,
        "use_pffn_bias": True,
        "use_final_linear_bias": True,
        "dropout": 0.1,
        "max_length": None,  # Must be set while initializing the model
        "model_type": None,  # Must be set while initializing the model
    }

    # 773M parameters with ModelType.MLP
    GPT2_LARGE = {
        "_cls": "TransformerTextGeneration",
        "num_layers": 36,
        "vocab_size": 50257,
        "embedding_dim": 1280,
        "query_key_dim": 1280,
        "value_dim": 1280,
        "num_heads": 20,
        "ffn_hidden_dim": 4 * 1280,
        "ffn_activation": "gelu",
        "use_kan_bias": False,
        "use_pffn_bias": True,
        "use_final_linear_bias": True,
        "dropout": 0.1,
        "max_length": None,  # Must be set while initializing the model
        "model_type": None,  # Must be set while initializing the model
    }

    # 1555M (1.55B) parameters with ModelType.MLP
    GPT2_XL = {
        "_cls": "TransformerTextGeneration",
        "num_layers": 48,
        "vocab_size": 50257,
        "embedding_dim": 1600,
        "query_key_dim": 1600,
        "value_dim": 1600,
        "num_heads": 25,
        "ffn_hidden_dim": 4 * 1600,
        "ffn_activation": "gelu",
        "use_kan_bias": False,
        "use_pffn_bias": True,
        "use_final_linear_bias": True,
        "dropout": 0.1,
        "max_length": None,  # Must be set while initializing the model
        "model_type": None,  # Must be set while initializing the model
    }

    # GPT_MINI, GPT_MICRO, GPT_NANO are based on https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    # 72.7M parameters with ModelType.MLP
    GPT2_MINI = {
        "_cls": "TransformerTextGeneration",
        "num_layers": 6,
        "vocab_size": 50257,
        "embedding_dim": 192,
        "query_key_dim": 192,
        "value_dim": 192,
        "num_heads": 6,
        "ffn_hidden_dim": 4 * 192,
        "ffn_activation": "gelu",
        "use_kan_bias": False,
        "use_pffn_bias": True,
        "use_final_linear_bias": True,
        "dropout": 0.1,
        "max_length": None,  # Must be set while initializing the model
        "model_type": None,  # Must be set while initializing the model
    }

    # 12.3M parameters with ModelType.MLP
    GPT2_MICRO = {
        "_cls": "TransformerTextGeneration",
        "num_layers": 4,
        "vocab_size": 50257,
        "embedding_dim": 128,
        "query_key_dim": 128,
        "value_dim": 128,
        "num_heads": 4,
        "ffn_hidden_dim": 4 * 128,
        "ffn_activation": "gelu",
        "use_kan_bias": False,
        "use_pffn_bias": True,
        "use_final_linear_bias": True,
        "dropout": 0.1,
        "max_length": None,  # Must be set while initializing the model
        "model_type": None,  # Must be set while initializing the model
    }

    # 2.5M parameters with ModelType.MLP
    GPT2_NANO = {
        "_cls": "TransformerTextGeneration",
        "num_layers": 3,
        "vocab_size": 50257,
        "embedding_dim": 48,
        "query_key_dim": 48,
        "value_dim": 48,
        "num_heads": 3,
        "ffn_hidden_dim": 4 * 48,
        "ffn_activation": "gelu",
        "use_kan_bias": False,
        "use_pffn_bias": True,
        "use_final_linear_bias": True,
        "dropout": 0.1,
        "max_length": None,  # Must be set while initializing the model
        "model_type": None,  # Must be set while initializing the model
    }

    def __call__(self):
        return self.value.copy()
