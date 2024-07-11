from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.t5.tokenization_t5 import T5Tokenizer

NEG_INF = -1e9


@dataclass
class TransformerConfig:
  vocab_size: int
  d_model: int = 512
  d_ff: int = 2048
  num_layers: int = 6
  num_heads: int = 8
  dropout: float = 0.1


class Embedding(nn.Module):

  def __init__(self, config: TransformerConfig):
    super().__init__()
    self.embed = nn.Embedding(config.vocab_size, config.d_model)
    self.scalar = np.sqrt(config.d_model)

  def forward(self, x):
    return self.embed(x) * self.scalar


class PositionEncoding(nn.Module):

  def forward(self, x, start_index=None):
    if start_index is None:
      start_index = 0
    batch, length, channels = x.size()
    position = torch.arange(0, length).unsqueeze(0).unsqueeze(-1)
    position += start_index
    scalar = torch.exp(-np.log(1e4) * torch.arange(0, channels, 2) / channels)
    scalar = scalar.reshape(1, 1, -1)
    sin = torch.sin(position * scalar)
    cos = torch.cos(position * scalar)
    signal = torch.concat([sin, cos], dim=-1)
    return x + signal


class Projection(nn.Module):

  def __init__(self, config: TransformerConfig, concat: bool = False):
    super().__init__()
    self.concat = concat
    self.num_heads = config.num_heads
    self.d_model = config.d_model
    self.linear = nn.Linear(config.d_model, config.d_model, bias=False)

  def forward(self, x):
    if not self.concat:
      x = self.linear(x)
      x = x.reshape(x.size(0), x.size(1), self.num_heads, -1)
    x = x.transpose(1, 2)
    if self.concat:
      x = x.reshape(x.size(0), -1, self.d_model)
      x = self.linear(x)
    return x


class Attention(nn.Module):

  def __init__(self, config: TransformerConfig):
    if config.d_model % config.num_heads != 0:
      raise ValueError(
          f"expected d_model % num_heads == 0, but received {config.d_model} % "
          f"{config.num_heads} != 0")
    super().__init__()
    self.q_proj = Projection(config)
    self.k_proj = Projection(config)
    self.v_proj = Projection(config)
    self.out_proj = Projection(config, concat=True)
    self.dropout = nn.Dropout(config.dropout)
    self.scalar = np.sqrt(config.d_model // config.num_heads)

  def forward(self, x, memory, mask):
    q = self.q_proj(x)
    k = self.k_proj(memory)
    v = self.v_proj(memory)
    output = torch.matmul(q, k.transpose(2, 3)) / self.scalar + mask
    output = F.softmax(output, dim=-1)
    output = self.dropout(output)
    output = torch.matmul(output, v)
    return self.out_proj(output)


class SelfAttention(Attention):

  def forward(self, x, mask):
    return super().forward(x, memory=x, mask=mask)


class Residual(nn.Module):

  def __init__(self, config: TransformerConfig):
    super().__init__()
    self.dropout = nn.Dropout(config.dropout)

  def forward(self, residual, x):
    return residual + self.dropout(x)


class FFN(nn.Module):

  def __init__(self, config: TransformerConfig):
    super().__init__()
    self.w1 = nn.Linear(config.d_model, config.d_ff)
    self.w2 = nn.Linear(config.d_ff, config.d_model)
    self.dropout = nn.Dropout(config.dropout)

  def forward(self, x):
    x = F.relu(self.w1(x))
    return self.w2(self.dropout(x))


class Encoder(nn.Module):

  def __init__(self, config: TransformerConfig):
    super().__init__()
    self.self_attn_norm = nn.LayerNorm(config.d_model)
    self.self_attn = SelfAttention(config)
    self.self_attn_res = Residual(config)
    self.ffn_norm = nn.LayerNorm(config.d_model)
    self.ffn = FFN(config)
    self.ffn_res = Residual(config)

  def forward(self, x, mask):
    y = self.self_attn(self.self_attn_norm(x), mask)
    x = self.self_attn_res(x, y)
    y = self.ffn(self.ffn_norm(x))
    x = self.ffn_res(x, y)
    return x


class EncoderTower(nn.Module):

  def __init__(self, config: TransformerConfig):
    super().__init__()
    self.layers = nn.ModuleList(
        [Encoder(config) for _ in range(config.num_layers)])
    self.pe = PositionEncoding()
    self.dropout = nn.Dropout(config.dropout)
    self.norm = nn.LayerNorm(config.d_model)

  def forward(self, src, mask, cache=None):
    if cache is not None and "encoder_output" in cache:
      return cache["encoder_output"]
    mask = (mask * NEG_INF).unsqueeze(1).unsqueeze(1)
    x = self.pe(src)
    x = self.dropout(x)
    for layer in self.layers:
      x = layer(x, mask)
    x = self.norm(x)
    if cache is not None:
      cache["encoder_output"] = x
    return x


class Decoder(nn.Module):

  def __init__(self, config: TransformerConfig):
    super().__init__()
    self.self_attn_norm = nn.LayerNorm(config.d_model)
    self.self_attn = SelfAttention(config)
    self.self_attn_res = Residual(config)
    self.ed_attn_norm = nn.LayerNorm(config.d_model)
    self.ed_attn = Attention(config)
    self.ed_attn_res = Residual(config)
    self.ffn_norm = nn.LayerNorm(config.d_model)
    self.ffn = FFN(config)
    self.ffn_res = Residual(config)

  def forward(self, x, mask, memory, memory_mask):
    y = self.self_attn(self.self_attn_norm(x), mask)
    x = self.self_attn_res(x, y)
    y = self.ed_attn(self.ed_attn_norm(x), memory, memory_mask)
    x = self.ed_attn_res(x, y)
    y = self.ffn(self.ffn_norm(x))
    x = self.ffn_res(x, y)
    return x


class DecoderTower(nn.Module):

  def __init__(self, config: TransformerConfig):
    super().__init__()
    self.layers = nn.ModuleList(
        [Decoder(config) for _ in range(config.num_layers)])
    self.pe = PositionEncoding()
    self.dropout = nn.Dropout(config.dropout)
    self.norm = nn.LayerNorm(config.d_model)

  def forward(self, memory, memory_mask, target, target_start_index=None):
    memory_mask = (memory_mask * NEG_INF).unsqueeze(1).unsqueeze(1)
    length = target.size(1)
    mask = torch.triu(torch.ones(length, length), diagonal=1) * NEG_INF
    mask = mask.unsqueeze(0).unsqueeze(0)
    x = F.pad(target, (0, 0, 1, 0, 0, 0))[:, :-1, :]
    x = self.pe(x, target_start_index)
    x = self.dropout(x)
    for layer in self.layers:
      x = layer(x, mask, memory, memory_mask)
    x = self.norm(x)
    return x


class Generator(nn.Module):

  def __init__(self, config: TransformerConfig):
    super().__init__()
    self.linear = nn.Linear(config.d_model, config.vocab_size)

  def forward(self, x):
    return F.softmax(self.linear(x), dim=-1)


class Transformer(nn.Module):

  def __init__(self, config: TransformerConfig):
    super().__init__()
    self.src_embed = Embedding(config)
    self.target_embed = Embedding(config)
    self.encoder = EncoderTower(config)
    self.decoder = DecoderTower(config)
    self.generator = Generator(config)

  def forward(self, target_start_index=None, **kwargs):
    src = kwargs['input_ids']
    mask = kwargs['attention_mask']
    target = src
    src = self.src_embed(src)
    target = self.target_embed(target)
    memory = self.encoder(src, mask)
    output = self.decoder(memory, mask, target, target_start_index)
    output = self.generator(output)
    return output


if __name__ == "__main__":
  torch.random.manual_seed(0)
  texts = [
      "translate from English to German: i love you",
      "give me a recipe for chicken soup",
  ]
  tokenizer = T5Tokenizer.from_pretrained('t5-small', model_max_length=3072)
  inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
  config = TransformerConfig(vocab_size=len(tokenizer))
  transformer = Transformer(config)
  transformer.eval()
  x = transformer(**inputs)
  predictions = x.argmax(dim=-1)
  for p in predictions:
    print(tokenizer.decode(p))
    print()

  # x = torch.from_numpy(np.random.random((2, 128, 32)))
  # plt.plot(np.arange(0, x.size(1)), PositionEncoding()(x)[0, :, 4:8])
  # plt.show()
