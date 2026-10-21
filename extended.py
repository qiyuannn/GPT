import tiktoken
from model import Transformer
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import math
import os

save_dir = "/workspace/GPT/checkpoints"
os.makedirs(save_dir, exist_ok=True)

# Model parameters
batch_size = 32
max_steps = 40000
warmup_steps = 1000
min_lr = 6e-5
max_lr = 6e-5
eval_iters = 40

## Transformer parameters
num_embeddings = tiktoken.get_encoding("gpt2").n_vocab
block_size = 512
embedding_dim = 512
num_heads = 8
num_layers = 8
dropout = 0.1

if torch.backends.mps.is_available():
    device = 'mps' 
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu' 
print(f"device: {device}")

# load Dataset
train_data = torch.load("train.pt")
val_data = torch.load("val.pt")
test_data = torch.load("test.pt")

# Model & Optimiser
model = Transformer(num_embeddings, block_size, embedding_dim, num_heads, num_layers, dropout)
model.to(device)
optimiser = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(.9, .95), fused=True)

## load past state
checkpoint = torch.load(
    os.path.join(save_dir, "final.pt"),
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint["model"])
optimiser.load_state_dict(checkpoint["optimiser"])
start_step = checkpoint["step"]

for group in optimiser.param_groups:
    group["lr"] = max_lr

# Helper functions
def getBatch(data):
    ix = torch.randint(0, len(data) - block_size, (batch_size, ))
    positions = ix[:, None] + torch.arange(block_size + 1)
    tokens = data[positions].to(device, dtype=torch.long)
    return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split, split_data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xs, ys = getBatch(split_data)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits = model(xs)
                B, T, C = logits.shape
                losses[k] = F.cross_entropy(logits.view(B * T, C), ys.view(B * T))
        out[split] = losses.mean().item()
    model.train()
    return out

best_ppl = float('inf')

for i in range(start_step, max_steps):
    if i % 500 == 0:
        losses = estimate_loss()
        train_ppl = math.exp(losses['train'])
        val_ppl = math.exp(losses["val"])
        print(f"iteration: {i} train ppl: {train_ppl:.4f}, val ppl: {val_ppl:.4f}")

        if i % 5000 == 0:
            torch.save({
                "model": model.state_dict(),
                "optimiser": optimiser.state_dict(),
                "step": i
            }, os.path.join(save_dir, f"checkpoint_{i}.pt"))

        if val_ppl < best_ppl:
            best_ppl = val_ppl
            torch.save({
                "model": model.state_dict(),
                "optimiser": optimiser.state_dict(),
                "step": i
            }, os.path.join(save_dir, f"best.pt"))

    xs, ys = getBatch(train_data)

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits = model(xs)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), ys.view(B * T))

    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimiser.step()

torch.save({
    "model": model.state_dict(),
    "optimiser": optimiser.state_dict(),
    "step": max_steps
}, os.path.join(save_dir, f"final.pt"))


    