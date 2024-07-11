"""
Transformer model as described in 'Attention is All You Need'
https://arxiv.org/pdf/1706.03762.pdf
"""
# pylint: disable=invalid-name
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# N = 6
# d_model = 512 -> dimensionality of sublayer and embedding outputs
# d_ff = 2048 -> num units in feed forward networks
# h = 8 -> num attention heads in multihead attention
# d_k = 64 -> key dimensionality in self attention
# d_v = 64 -> value dimensionality in self attention
# P_drop = 0.1 -> dropout rate
# eps_ls = 0.1 -> label smoothing

_MIN_TIMESCALE = 1.0
_MAX_TIMESCALE = 1.0e4


class Embedding(nn.Module):

  def __init__(self, vocab_size, hidden_size):
    super().__init__()
    self.lookup = nn.Embedding(vocab_size, hidden_size)
    self.scalar = np.sqrt(hidden_size)

  def forward(self, x):
    return self.lookup(x) * self.scalar  # scalar = sqrt(d_model)


class PositionEncoding(nn.Module):

  def forward(self, x, start_index=None):
    _, length, channels = x.size()  # B, L, C
    if start_index is None:
      start_index = 0
    position = torch.arange(length).unsqueeze(0).unsqueeze(-1)  # 1, L, 1
    position += start_index
    inverse_divisor = torch.exp(
        torch.arange(0, channels, 2) * -np.log(1e4) / channels)
    inverse_divisor = inverse_divisor.reshape(1, 1, -1)  # 1, 1, C
    sin = torch.sin(position * inverse_divisor)  # 1, L, C
    cos = torch.sin(position * inverse_divisor)  # 1, L, C
    signal = torch.concat([sin, cos], dim=2)
    return x + signal  # B, L, C


class Projection(nn.Module):

  def __init__(self, hidden_size, num_heads=None):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.linear = nn.Linear(hidden_size, hidden_size, bias=False)

  def forward(self, x):
    # projecting up into attention head space
    if self.num_heads is not None:
      x = self.linear(x)
      x = x.reshape(x.size(0), x.size(1), self.num_heads, -1)
    x = x.transpose(2, 1)
    # projecting down into output space
    if self.num_heads is None:
      x = x.reshape(x.size(0), -1, self.hidden_size)
      x = self.linear(x)
    return x


class Attention(nn.Module):

  def __init__(self, hidden_size, num_heads, dropout):
    super().__init__()
    if hidden_size % num_heads != 0:
      raise ValueError(
          "Number of attention heads must evenly divide hidden size")
    self.q_proj = Projection(hidden_size, num_heads=num_heads)
    self.k_proj = Projection(hidden_size, num_heads=num_heads)
    self.v_proj = Projection(hidden_size, num_heads=num_heads)
    self.out_proj = Projection(hidden_size)
    self.dropout = dropout
    self.scalar = (hidden_size // num_heads)**(-0.5)

  def forward(self, x, memory, mask):
    q = self.q_proj(x) * self.scalar
    k = self.k_proj(memory)
    v = self.v_proj(memory)
    logits = torch.matmul(q, k.transpose(2, 3)) + mask
    output = F.softmax(logits, dim=-1)
    output = F.dropout(output, p=self.dropout)
    output = torch.matmul(output, v)
    return self.out_proj(output)


class SelfAttention(Attention):

  def forward(self, x, mask):
    return super(SelfAttention, self).forward(x, x, mask)


class PositionWiseFeedFoward(nn.Module):

  def __init__(self, hidden_size, ffn_size, dropout):
    super().__init__()
    self.w1 = nn.Linear(hidden_size, ffn_size)
    self.w2 = nn.Linear(ffn_size, hidden_size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    return self.w2(self.dropout(F.relu(self.w1(x))))


class Pre(nn.Module):

  def __init__(self, hidden_size):
    super().__init__()
    self.norm = nn.LayerNorm(hidden_size)

  def forward(self, x):
    return self.norm(x)


class Post(nn.Module):

  def __init__(self, dropout):
    super().__init__()
    self.dropout = dropout

  def forward(self, x_residual, x):
    x = F.dropout(x, self.dropout)
    return x_residual + x


class Encoder(nn.Module):

  def __init__(self, hidden_size, ffn_size, num_heads, attention_dropout,
               relu_dropout, post_dropout):
    super().__init__()
    self.self_attention_pre = Pre(hidden_size)
    self.self_attention = SelfAttention(hidden_size, num_heads,
                                        attention_dropout)
    self.self_attention_post = Post(post_dropout)
    self.ffn_pre = Pre(hidden_size)
    self.ffn = PositionWiseFeedFoward(hidden_size, ffn_size, relu_dropout)
    self.ffn_post = Post(post_dropout)

  def forward(self, x, mask):
    y = self.self_attention(self.self_attention_pre(x), mask)
    x = self.self_attention_post(x, y)
    y = self.ffn(self.ffn_pre(x))
    x = self.ffn_post(x, y)
    return x


class Decoder(nn.Module):

  def __init__(self, hidden_size, ffn_size, num_heads, attention_dropout,
               relu_dropout, post_dropout):
    super().__init__()
    self.self_attention_pre = Pre(hidden_size)
    self.self_attention = SelfAttention(hidden_size, num_heads,
                                        attention_dropout)
    self.self_attention_post = Post(post_dropout)
    self.encdec_attention_pre = Pre(hidden_size)
    self.encdec_attention = Attention(hidden_size, num_heads, attention_dropout)
    self.encdec_attention_post = Post(post_dropout)
    self.ffn_pre = Pre(hidden_size)
    self.ffn = PositionWiseFeedFoward(hidden_size, ffn_size, relu_dropout)
    self.ffn_post = Post(post_dropout)

  def forward(self, x, mask, memory, memory_mask):
    y = self.self_attention(self.self_attention_pre(x), mask)
    x = self.self_attention_post(x, y)
    y = self.encdec_attention(self.encdec_attention_pre(x), memory, memory_mask)
    x = self.encdec_attention_post(x, y)
    y = self.ffn(self.ffn_pre(x))
    x = self.ffn_post(x, y)
    return x


class EncoderStack(nn.Module):

  def __init__(self, num_layers, hidden_size, ffn_size, num_heads,
               attention_dropout, relu_dropout, post_dropout):
    super().__init__()
    self.pre = Pre(hidden_size)
    self.layers = nn.ModuleList([
        Encoder(hidden_size, ffn_size, num_heads, attention_dropout,
                relu_dropout, post_dropout) for _ in range(num_layers)
    ])
    self.post_dropout = post_dropout

  def forward(self, x, padding, cache=None):
    if cache is not None and "encoder_output" in cache:
      return cache["encoder_output"]
    mask = (padding * -1e9).unsqueeze(1).unsqueeze(1)
    x = PositionEncoding()(x)
    x = F.dropout(x, self.post_dropout)
    for layer in self.layers:
      x = layer(x, mask)
    x = self.pre(x)
    if cache is not None:
      cache["encoder_output"] = x
    return x


class DecoderStack(nn.Module):

  def __init__(self, num_layers, hidden_size, ffn_size, num_heads,
               attention_dropout, relu_dropout, post_dropout):
    super().__init__()
    self.pre = Pre(hidden_size)
    self.layers = nn.ModuleList([
        Decoder(hidden_size, ffn_size, num_heads, attention_dropout,
                relu_dropout, post_dropout) for _ in range(num_layers)
    ])
    self.post_dropout = post_dropout

  def forward(self, memory, memory_padding, target, target_start=None):
    memory_mask = (memory_padding * -1e9).unsqueeze(1).unsqueeze(1)
    sequence_len = target.size(1)
    mask = torch.triu(torch.ones(sequence_len, sequence_len), diagonal=1) * -1e9
    mask = mask.unsqueeze(0).unsqueeze(0)
    x = F.pad(target, (0, 0, 1, 0, 0, 0))[:, :-1, :]  # right shift
    x = PositionEncoding()(x, target_start)
    x = F.dropout(x, self.post_dropout)
    for layer in self.layers:
      x = layer(x, mask, memory, memory_mask)
    x = self.pre(x)
    return x


class Generator(nn.Module):

  def __init__(self, hidden_size, vocab_size):
    super().__init__()
    self.linear = nn.Linear(hidden_size, vocab_size)

  def forward(self, x):
    return F.log_softmax(self.linear(x), dim=-1)


class Transformer(nn.Module):

  def __init__(self, N, d_model, d_ff, h, P_drop):
    super().__init__()
    self.encoder = EncoderStack(N, d_model, d_ff, h, P_drop, P_drop, P_drop)
    self.decoder = DecoderStack(N, d_model, d_ff, h, P_drop, P_drop, P_drop)

  def forward(self, x, target, padding, cache=None, target_start=None):
    memory = self.encoder(x, padding, cache=cache)
    output = self.decoder(memory, padding, target, target_start=target_start)
    return output


class ScheduledOptim:

  def __init__(self, hidden_size, warmup_steps, optimizer):
    self._step = 0
    self._rate = 0
    self.hidden_size = hidden_size
    self.warmup_steps = warmup_steps
    self.optimizer = optimizer

  def step(self):
    self._step += 1
    rate = self.rate()
    for p in self.optimizer.param_groups:
      p['lr'] = rate
    self._rate = rate
    self.optimizer.step()

  def rate(self, step=None):
    if step is None:
      step = self._step
    return (self.hidden_size**(-0.5) *
            min(step**(-0.5), step * self.warmup_steps**(-1.5)))


def get_model_and_optimizer(N=6, d_model=512, d_ff=2048, h=8, P_drop=0.1):
  model = Transformer(N, d_model, d_ff, h, P_drop)
  optimizer = ScheduledOptim(hidden_size=d_model,
                             factor=1,
                             warmup_steps=4000,
                             optimizer=torch.optim.Adam(model.parameters(),
                                                        lr=0,
                                                        betas=(0.9, 0.98),
                                                        eps=1e-9))
  return model, optimizer
