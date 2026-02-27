from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
chunk_size = 128
batch_size = 4
learning_rate = 3e-4
max_iters = 10000
n_head = 8
n_layer = 6
dropout = 0.2
n_embed = 256 #Defines the dimensionality of vector space where token's meaning will be stored
eval_iters = 1000

input_data = Path("data/input.txt").read_text(encoding="utf-8")
chars = sorted(list(set(input_data)))
vocab_size = len(chars)

stoi = {ch : i for i,ch in enumerate(chars)}
itos = {i: ch for i,ch in enumerate(chars)}

encode = lambda e: [stoi[ch] for ch in e]
decode = lambda d: [itos[i] for i in d]

tokenized_data = torch.tensor(encode(input_data), dtype=torch.long)

n = int(0.9 * len(tokenized_data))
train_data = tokenized_data[:n]
val_data = tokenized_data[n:]

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(0, len(data)-chunk_size, (batch_size,))
    x = torch.stack([data[i: i+chunk_size] for i in ix])
    y = torch.stack([data[i+1: i+chunk_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y

class Head(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias = False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(chunk_size, chunk_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * k.size(-1)**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHead(nn.Module):
    
    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) #Concatenate multiple communication channel: each hidden vector to produce final vector
        out = self.dropout(self.proj(out))
        out = self.proj(out) #Linear transformation of outcome from this layer
        return out 

class FeedForward(nn.Module):
    
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    
    def __init__(self, n_embed, num_heads):
        super().__init__()
        head_size = n_embed // num_heads
        self.sa = MultiHead(num_heads=num_heads, head_size=head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) #Add residual connections
        x = x + self.ffwd(self.ln2(x))
        return x
    
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(chunk_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, num_heads=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
       
    def forward(self, idx, targets=None):
        B,T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -chunk_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        
        for k in range(eval_iters):
            X,Y = get_batch(split)
            _, loss = model(X,Y)
            losses[k] = loss.item()
        
        out[split] = losses.mean()
    model.train()
    return out

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for iter in range(max_iters):
    if iter % eval_iters == 0:
        l = estimate_loss()
        print(f"Step: {iter}, Training loss: {l['train']:.4f} and validation loss: {l['val']:.4f}")

    xb, yb = get_batch("train")
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"Loss after training: {loss.item()}")
print(f"Model saved 8000 chars to output.txt")
context = torch.zeros((1,1), dtype=torch.long, device=device)
output = ''.join(decode(m.generate(context, max_new_tokens=8000)[0].tolist()))
op_path = Path("data/output.txt")
op_path.parent.mkdir(parents=True, exist_ok=True)
op_path.write_text(output, encoding="utf-8")
