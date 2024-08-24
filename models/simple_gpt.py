import torch
import torch.nn as nn
from torch.nn import functional as F


class SimpleGPT(nn.Module):
    """ A simple GPT """

    def __init__(self, gptConfig):
        super().__init__()
        self.gptConfig = gptConfig
        self.encodeMap = None
        self.decodeMap = None
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(gptConfig.vocab_size, gptConfig.n_embd)
        self.position_embedding_table = nn.Embedding(gptConfig.block_size, gptConfig.n_embd)
        self.blocks = nn.Sequential(*[Block(gptConfig) for _ in range(gptConfig.n_layer)])
        self.ln_f = nn.LayerNorm(gptConfig.n_embd) # final layer norm
        self.lm_head = nn.Linear(gptConfig.n_embd, gptConfig.vocab_size)
        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def createMappings(self, tokens):
        # create a mapping from characters to integers
        self.encodeMap = { token:i for i,token in enumerate(tokens) }
        self.decodeMap = { i:token for i,token in enumerate(tokens) }

    def encode(self, s):
        # encoder: take a string, output a list of integers
        return [self.encodeMap[c] for c in s]

    def decode(self, l):
        # decoder: take a list of integers, output a string
        return ''.join([self.decodeMap[i] for i in l])

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.gptConfig.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.gptConfig.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, gptConfig):
        super().__init__()
        self.gptConfig = gptConfig
        head_size = gptConfig.n_embd // gptConfig.n_head
        self.key = nn.Linear(gptConfig.n_embd, head_size, bias=False)
        self.query = nn.Linear(gptConfig.n_embd, head_size, bias=False)
        self.value = nn.Linear(gptConfig.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(gptConfig.block_size, gptConfig.block_size)))

        self.dropout = nn.Dropout(gptConfig.dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, gptConfig):
        super().__init__()
        self.gptConfig = gptConfig
        head_size = gptConfig.n_embd // gptConfig.n_head
        self.heads = nn.ModuleList([Head(gptConfig) for _ in range(gptConfig.n_head)])
        self.proj = nn.Linear(head_size * gptConfig.n_head, gptConfig.n_embd)
        self.dropout = nn.Dropout(gptConfig.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, gptConfig):
        super().__init__()
        self.gptConfig = gptConfig
        self.net = nn.Sequential(
            nn.Linear(gptConfig.n_embd, 4 * gptConfig.n_embd),
            nn.ReLU(),
            nn.Linear(4 * gptConfig.n_embd, gptConfig.n_embd),
            nn.Dropout(gptConfig.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, gptConfig):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.gptConfig = gptConfig
        self.sa = MultiHeadAttention(gptConfig)
        self.ffwd = FeedFoward(gptConfig)
        self.ln1 = nn.LayerNorm(gptConfig.n_embd)
        self.ln2 = nn.LayerNorm(gptConfig.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x