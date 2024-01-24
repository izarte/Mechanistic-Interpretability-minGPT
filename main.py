from mingpt.bpe import BPETokenizer
import torch

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from mingpt.model import GPT
from mingpt.utils import set_seed
from mingpt.bpe import BPETokenizer
from mingpt.utils import set_seed


def create_embeddings(input="Michelle Jones was a top-notch student. Michelle", verbose=False):
    # Set seed to get statical results
    set_seed(42)
    # Tokenize input string
    bpe = BPETokenizer()
    tokens = bpe(input)[0]
    input_length = tokens.shape[-1]
    tokens_str = [bpe.decode(torch.tensor([token])) for token in tokens]
    if verbose:
        print("Number of input tokens:", input_length)
        print("Detokenized input from indices:", bpe.decode(tokens))
        print("Detokenized input as strings: " + '/'.join(tokens_str))

    #Create and select model
    model_type = 'gpt2'
    device = 'cpu'
    model = GPT.from_pretrained(model_type)
    model.to(device)
    model.eval()

    num_samples = 1
    # Process tokens
    x = bpe(input).to(device)
    x = x.expand(num_samples, -1)
    # Obtainr results and internally save embeddings
    y, logits = model.generate(x, max_new_tokens=20, do_sample=True, top_k=40, save_name="embeddings.pt")

    values, indexes = torch.topk(logits, k=10)
    if verbose:
        j = bpe(' Jones').item()
        v = logits[0][j]
        print("Jones value: ", v)
        print(values)
        print(indexes)
    out = bpe.decode(indexes.cpu().squeeze())
    return out



if __name__ == '__main__':
    out = create_embeddings(verbose=True)
    print(out)




# set_seed(42)

# input = "Michelle Jones was a top-notch student. Michelle"
# print("Input:", input)
# bpe = BPETokenizer()
# # bpe() gets a string and returns a 2D batch tensor 
# # of indices with shape (1, input_length)
# tokens = bpe(input)[0]
# print("Tokenized input:", tokens)
# input_length = tokens.shape[-1]
# print("Number of input tokens:", input_length)
# # bpe.decode gets a 1D tensor (list of indices) and returns a string
# print("Detokenized input from indices:", bpe.decode(tokens))  
# tokens_str = [bpe.decode(torch.tensor([token])) for token in tokens]
# print("Detokenized input as strings: " + '/'.join(tokens_str))


# model_type = 'gpt2-xl'
# device = 'cpu'
# model = GPT.from_pretrained(model_type)
# model.to(device)
# model.eval()

# num_samples = 1

# x = bpe(input).to(device)
# x = x.expand(num_samples, -1)
# # y, values, indexes = model.generate(x, max_new_tokens=20, do_sample=True, top_k=40)
# y, logits = model.generate(x, max_new_tokens=20, do_sample=False, top_k=40)

# # print('-'*80)
# # print(values)
# # print(logits)
# j = bpe(' Jones').item()
# print(j)
# v = logits[0][j]
# print(v)
# print(bpe.decode(torch.tensor([j])))
# print("============================================")
# values, indexes = torch.topk(logits, k=10)
# print(values)
# print(indexes)
# out = bpe.decode(indexes.cpu().squeeze())
# print(out)

# for i in range(num_samples):
#     out = bpe.decode(y[i].cpu().squeeze())
#     print('-'*80)
#     print(out)
