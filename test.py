from mingpt.bpe import BPETokenizer
import torch

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from mingpt.model import GPT
from mingpt.utils import set_seed
from mingpt.bpe import BPETokenizer

input = "Michelle Jones was a top-notch student. Michelle"
print("Input:", input)
bpe = BPETokenizer()
# bpe() gets a string and returns a 2D batch tensor 
# of indices with shape (1, input_length)
tokens = bpe(input)[0]
print("Tokenized input:", tokens)
input_length = tokens.shape[-1]
print("Number of input tokens:", input_length)
# bpe.decode gets a 1D tensor (list of indices) and returns a string
print("Detokenized input from indices:", bpe.decode(tokens))  
tokens_str = [bpe.decode(torch.tensor([token])) for token in tokens]
print("Detokenized input as strings: " + '/'.join(tokens_str))


model_type = 'gpt2-xl'
device = 'cpu'
model = GPT.from_pretrained(model_type)
model.to(device)
model.eval()

num_samples = 10

x = bpe(input).to(device)
x = x.expand(num_samples, -1)
y = model.generate(x, max_new_tokens=20, do_sample=True, top_k=40)

for i in range(num_samples):
    out = bpe.decode(y[i].cpu().squeeze())
    print('-'*80)
    print(out)
