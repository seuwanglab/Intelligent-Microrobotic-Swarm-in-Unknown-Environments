import torch
from torch import nn


class Module(nn.Module):
    def __init__(self):
        super().__init__()

    def grad_norm(self):
        grads = []
        for name, parameter in self.named_parameters():
            grads.append(parameter.grad.view(-1))
        return torch.linalg.norm(torch.cat(grads)).item() if len(grads) > 0 else None

    def grad_mean(self):
        grads = []
        for name, parameter in self.named_parameters():
            grads.append(parameter.grad.view(-1))
        return torch.mean(torch.cat(grads)).item() if len(grads) > 0 else None


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return nn.functional.rms_norm(x, (self.dim,), self.weight, self.eps)


class RoPE(nn.Module):
    def __init__(self, head_dim):
        super(RoPE, self).__init__()
        self.head_dim = head_dim
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, position_ids):
        sinusoid_inp = torch.einsum("bs,d->bsd", position_ids, self.inv_freq)
        sin, cos = sinusoid_inp.sin(), sinusoid_inp.cos()

        sin = sin.unsqueeze(1)
        cos = cos.unsqueeze(1)

        x1, x2 = x[..., 0::2], x[..., 1::2]
        x = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = self.embedding_dim // num_heads
        assert (self.head_dim * num_heads == embedding_dim), \
            "Embedding dimension needs to be divisible by the number of heads"
        self.values = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.keys = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.queries = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.rope = RoPE(self.head_dim)
        self.fc = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, query, keys, values, index, memory_indices, mask):
        batch_size = query.shape[0]
        query_len, key_len, value_len = query.shape[1], keys.shape[1], values.shape[1]

        query = self.queries(query)
        keys = self.keys(keys)
        values = self.values(values)

        query = query.view(batch_size, query_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        keys = keys.view(batch_size, key_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        values = values.view(batch_size, value_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        query = self.rope(query, index)
        keys = self.rope(keys, memory_indices)

        energy = torch.matmul(query, keys.permute(0, 1, 3, 2))
        if mask is not None:
            energy = energy.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float("-1e20"))
        attention_weights = torch.softmax(energy / (self.head_dim ** 0.5), dim=-1)

        score = torch.matmul(attention_weights, values)
        score = score.permute(0, 2, 1, 3).contiguous().view(batch_size, query_len, self.num_heads * self.head_dim)
        attention = self.fc(score)

        return attention, attention_weights


class TransformerBlock(Module):
    def __init__(self, embedding_dim, hidden_size, num_heads):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embedding_dim, num_heads)

        self.norm_1q = RMSNorm(embedding_dim)
        self.norm_1kv = RMSNorm(embedding_dim)
        self.norm_2 = RMSNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, embedding_dim)
        )

    def forward(self, query, values, index, memory_indices, mask):
        query_ = self.norm_1q(query)
        values = self.norm_1kv(values)
        keys = values
        attention, attention_weights = self.attention(query_, keys, values, index, memory_indices, mask)
        h = query + attention
        h_ = self.norm_2(h)
        forward = self.ffn(h_)
        out = h + forward
        return out, attention_weights


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_blocks = config['n_blocks']
        self.num_heads = config['n_heads']
        self.embedding_dim = config['embedding_dim']
        self.hidden_size = config['hidden_size']

        self.transformer_blocks = nn.ModuleList([TransformerBlock(self.embedding_dim, self.hidden_size, self.num_heads)
                                                 for _ in range(self.num_blocks)])

    def forward(self, h, memories, memory_indices, mask):
        batch_size = h.shape[0]
        batch_arrange = torch.arange(batch_size, dtype=torch.int32, device=h.device)
        index = memory_indices[batch_arrange, mask.sum(dim=1) - 1].unsqueeze(1)
        _index = index.clone()
        valid_indices_mask = index.squeeze(1) < memory_indices.shape[1]
        valid_batch = batch_arrange[valid_indices_mask]
        valid_positions = index.squeeze(1)[valid_indices_mask]

        out_memories = []
        for i, block in enumerate(self.transformer_blocks):
            out_memories.append(h.detach())
            memory = memories[:, :, i]
            _memory = memory.clone()
            if len(valid_batch) > 0:
                _memory[valid_batch, valid_positions] = h.squeeze(1)[valid_batch]
            invalid_batch = batch_arrange[~valid_indices_mask]
            if len(invalid_batch) > 0:
                _memory[invalid_batch, -1] = h.squeeze(1)[invalid_batch]
            memory = _memory
            h, attention_weights = block(h.unsqueeze(1), memory, _index, memory_indices, mask)

            h = h.squeeze()
            if len(h.shape) == 1:
                h = h.unsqueeze(0)
        return h, torch.stack(out_memories, dim=1)
