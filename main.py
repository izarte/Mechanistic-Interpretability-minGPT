from mingpt.bpe import BPETokenizer
import torch
import matplotlib.pyplot as plt

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
    y, logits = model.generate(x, max_new_tokens=1, do_sample=True, top_k=40, save_name="embeddings.pt")

    values, indexes = torch.topk(logits, k=20)
    out = bpe.decode(indexes.cpu().squeeze())
    if verbose:
        j = bpe(' Jones').item()
        v = logits[0][j]
        print("Jones value: ", v)
        print("Values: ", values)
        print(indexes)
        print(out)
    return indexes, logits


def test_model(corrupted_input="Michelle Smith was a top-notch student. Michelle", clean_indexes=None, reference_tensor=None, verbose=False):
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

    reference_index = bpe(' Jones').item()
    corrupted_index = bpe(' Smith').item()

    num_samples = 1
    # Process tokens
    x = bpe(corrupted_input).to(device)
    x = x.expand(num_samples, -1)
    # Obtain results and internally save embeddings
    difference_matrix = []
    layers_n = range(0, 12)
    for layer in layers_n:
        difference_logit = []
        for p in range(input_length):
            print("Layer number: ", layer, " pos: ", p)
            y, logits = model.generate(x, max_new_tokens=1, do_sample=True, top_k=40, save_name=None, layer = layer, pos = p)
            # values = logits[0, clean_indexes]
            difference_logit.append(logits[0][corrupted_index] - reference_tensor[0][reference_index])

            # values, indexes = torch.topk(logits, k=10)
            # out = bpe.decode(indexes.cpu().squeeze())
            if verbose:
                pass
                # j = bpe(' Jones').item()
                # v = logits[0][j]
                # print("Jones value: ", v)
                # print(" values: ", values)
                # out = bpe.decode(indexes.cpu().squeeze())
                # print(out)
        difference_matrix.append(difference_logit)

    # # Reshape the tensor for plotting (flatten the first dimension)
    # difference_matrix = difference_matrix.view(12, -1)

    # Labels for each column
    words = tokens_str
    column_labels = [word for word in words]
    row_labels = range(0, 12)

    # Plotting the difference matrix using matshow
    plt.matshow(difference_matrix, cmap='bone', aspect='auto')
    plt.colorbar(label='Differences')

    # Set column labels
    plt.xticks(range(len(column_labels)), column_labels, rotation=45, ha='left')
    plt.yticks(range(len(row_labels)), row_labels)

    plt.title('Difference Matrix')
    plt.xlabel('Reference Tensor')
    plt.ylabel('Comparison Tensors')
    plt.show()

    return



def main():
    set_seed(3407)
    indexes, values = create_embeddings(verbose=False)
    test_model(clean_indexes=indexes, reference_tensor = values, verbose=False)




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
