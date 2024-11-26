from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class SigLipVisionConfig:
    def __init__(
            self,
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_channels=3,
            image_size=224,
            patch_size=16,
            layer_norm_eps=1e-6,
            attention_dropout=0.0,
            num_image_tokens : int = None,
            **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens

class SigLipVisisonEmbeddings(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        self.config = config
        self.emb_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embeddings = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.emb_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid"
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embeddings = nn.Embedding(self.num_positions, self.emb_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        patch_embeddings = self.patch_embeddings(pixel_values)
        embeddings = patch_embeddings.flatten(2)
        embeddings = embeddings.transpose(-1, -2)
        embeddings = embeddings + self.position_embeddings(self.position_ids)
        return embeddings
    
class SigLipAttention(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.emb_dim = config.hidden_size
        self.head_dim = self.emb_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.emb_dim, self.emb_dim)
        self.v_proj = nn.Linear(self.emb_dim, self.emb_dim)
        self.q_proj = nn.Linear(self.emb_dim, self.emb_dim)
        self.out_proj = nn.Linear(self.emb_dim, self.emb_dim)

    def forward(self, hidden_states: torch.Tensor)  -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.size()
        q_state = self.q_proj(hidden_states)
        k_state = self.k_proj(hidden_states)
        v_state = self.v_proj(hidden_states)

        q_state = q_state.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_state = k_state.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_state = v_state.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weight = (torch.matmul(q_state, k_state.transpose(2, 3)) * self.scale)

        attn_weight = F.softmax(attn_weight, dim=-1, dtype=torch.float32).to(q_state.dtype)
        attn_weight = F.dropout(attn_weight, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weight, v_state)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.emb_dim)

        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weight
    
class SigLipMLP(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class SigLipEncoderLayer(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.emd_dim = config.hidden_size
        self.self_attention = SigLipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.emd_dim, eps=config.layer_norm_eps)
        self.mlp = SigLipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.emd_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attention(hidden_states)
        hidden_states= residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
    

class SigLipEncoder(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SigLipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states
    
class SigLipVisionTransformer(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        emb_dim = config.hidden_size

        self.embeddings = SigLipVisisonEmbeddings(config)
        self.encoder = SigLipEncoder(config)
        self.post_layernorm = nn.LayerNorm(emb_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state

class SigLipVisionModel(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SigLipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        return self.vision_model(pixel_values=pixel_values)