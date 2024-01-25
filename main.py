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
    # Obtain results and internally save embeddings
    y, logits = model.generate(x, max_new_tokens=20, do_sample=True, top_k=40, save_name="embeddings.pt")

    values, indexes = torch.topk(logits, k=20)
    out = bpe.decode(indexes.cpu().squeeze())
    if verbose:
        j = bpe(' Jones').item()
        v = logits[0][j]
        print("Jones value: ", v)
        print("Values: ", values)
        print(indexes)
        print(out)
    return indexes


def test_model(corrupted_input="Michelle Smith was a top-notch student. Michelle", clean_indexes=None, verbose=False):
    bpe = BPETokenizer()
    tokens = bpe(corrupted_input)[0]
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
    x = bpe(corrupted_input).to(device)
    x = x.expand(num_samples, -1)
    # Obtain results and internally save embeddings
    y, logits = model.generate(x, max_new_tokens=20, do_sample=True, top_k=40, save_name=None, layer = 0, pos = 1)

    values = logits[0, clean_indexes]
    # values, indexes = torch.topk(logits, k=10)
    # out = bpe.decode(indexes.cpu().squeeze())
    if verbose:
        # j = bpe(' Jones').item()
        # v = logits[0][j]
        # print("Jones value: ", v)
        print(" values: ", values)
        # print(indexes)
        # print(out)
    # return out
    return



def main():
    set_seed(3407)
    indexes = create_embeddings(verbose=True)
    # test_model(clean_indexes=indexes, verbose=True)




if __name__ == '__main__':
    main()




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
