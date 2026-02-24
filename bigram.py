from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F

#Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
batch_size = 32 #how many independent sequences we'll process in parallel
chunk_size = 8 #max context length
eval_iters = 200

#Reading input data
input_data = Path("data/input.txt").read_text(encoding="utf-8")
chars = sorted(list(set(input_data)))
vocab_size = len(chars)

#Tokenization
stoi = {ch: i for i,ch in enumerate(chars)}
itos = {i: ch for i,ch in enumerate(chars)}

encode = lambda e: [stoi[ch] for ch in e]
decode = lambda d: "".join([itos[i] for i in d])

tokenized_data = torch.tensor(encode(input_data), dtype=torch.long)

#Train-Test Split
n = int(0.9 * len(tokenized_data))
train_data = tokenized_data[:n]
val_data = tokenized_data[n:]

#Data Loader
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - chunk_size, (batch_size,)) #len(data) - chunk_size to avoid out of index error
    x = torch.stack([data[i: i+chunk_size] for i in ix])
    y = torch.stack([data[i+1: i+chunk_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y

#Loss estimate function
@torch.no_grad() #to tell pytorch we are not going to call backward so save intermediate calculations
def estimate_loss():
    out = {}
    model.eval() #When we use dropout layers this disables it
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            X,Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        
        out[split] = losses.mean()
    
    model.train() #bring back training config
    return out


#Bigram Model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets= targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx


model = BigramLanguageModel(vocab_size=vocab_size)
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0:
        l = estimate_loss()
        print(f"Step: {iter}, Training loss: {l["train"]:.4f} and validation loss: {l["val"]:.4f}")

    xb, yb = get_batch("train")
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True) #set_to_none is an efficient way to clear out gradients from last iter
    loss.backward() #Calculate gradients
    optimizer.step() #Apply gradients to parameters

print(f"Loss after training: {loss.item()}")

print("Model printing 3000 chars after training:")
context = torch.zeros((1,1), dtype=torch.long, device= device)
print(decode(m.generate(context, max_new_tokens=3000)[0].tolist())) 