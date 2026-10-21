from datasets import load_dataset
import tiktoken
import torch

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
encoding = tiktoken.get_encoding("gpt2")

for split, filename in [("train", "train.pt"), ("test", "test.pt"), ("validation", "val.pt")]:
    text = "\n".join(ds[split]["text"])
    ids = encoding.encode(text)
    torch.save(torch.tensor(ids, dtype=torch.int32),filename)
