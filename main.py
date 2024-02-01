from mingpt.bpe import BPETokenizer
import torch
import matplotlib.pyplot as plt

import torch
from mingpt.model import GPT
from mingpt.utils import set_seed
from mingpt.bpe import BPETokenizer
from mingpt.utils import set_seed

import sys
from prettytable import PrettyTable


def check_same_size(real, corrupted):
    bpe = BPETokenizer()
    real_tokens = bpe(real)[0]
    real_length = real_tokens.shape[-1]
    real_tokens_str = [bpe.decode(torch.tensor([token])) for token in real_tokens]
    corrupted_tokens = bpe(corrupted)[0]
    corrupted_length = corrupted_tokens.shape[-1]
    corrupted_tokens_str = [bpe.decode(torch.tensor([token])) for token in corrupted_tokens]


    if real_length - corrupted_length != 0:
        print("ERROR NO SAME SIZE")
        print(f"real: {real_length} corrupted: {corrupted_length}")
        print("Real tokens     : " + '/'.join(real_tokens_str))
        print("corrupted tokens: " + '/'.join(corrupted_tokens_str))
        sys.exit()


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
    y, logits, probs = model.generate(x, max_new_tokens=1, do_sample=True, top_k=40, save_name="embeddings.pt")
    values, indexes = torch.topk(probs, k=20)

    print(indexes[0, 1].item())

    table = PrettyTable()
    table.field_names = ["Position", "Token index", "Token", "Probability"]
    for i in range(0, 20):
        table.add_row([f"{i + 1}", f"{indexes[0][i].item()}", f"{bpe.decode(torch.tensor([indexes[0][i].item()]))}", f"{values[0, i].item():.4f}"])
    with open("results/table.txt", 'w') as file:
        file.write(str(table))
    if verbose:
        print(table)

    return 


def test_model(corrupted_input="Michelle Smith was a top-notch student. Michelle", verbose=False, corrupted_token = None, swapped_token = None):
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



    corrupted_index = bpe(corrupted_token).item()
    swapped_index = bpe(swapped_token).item()

    num_samples = 1
    # Process tokens
    x = bpe(corrupted_input).to(device)
    x = x.expand(num_samples, -1)
    y, corrupted_logits, _ = model.generate(x, max_new_tokens=1, do_sample=True, top_k=40)

    # Obtain results and internally save embeddings
    difference_matrix = []
    layers_n = range(0, 12)
    for layer in layers_n:
        difference_logit = []
        for p in range(input_length):
            print("Layer number: ", layer, " pos: ", p)
            y, logits, _ = model.generate(x, max_new_tokens=1, do_sample=True, top_k=40, save_name=None, layer = layer, pos = p)
            difference_logit.append(corrupted_logits[0][corrupted_index] - logits[0][swapped_index])
        difference_matrix.append(difference_logit)

    # Labels for each column
    words = tokens_str
    column_labels = [word for word in words]
    row_labels = range(0, 12)

    # Plotting the difference matrix using matshow
    plt.matshow(difference_matrix, cmap='bone', aspect='auto')
    plt.colorbar(label=f'Differences between {corrupted_token}-{swapped_token}')

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

    # warm-cold
    # real = "In winter temperature is cold, in summer it is"
    # corrupted = "In winter temperature is warm, in summer it is"

    # jackson-jordan
    # real = "Michael Jordan is a player. Michael"
    # corrupted = "Michael Jackson is a player. Michael"

    # green-purple
    # real = "Red and blue turns in purple. Yellow and blue turns in"
    # corrupted = "Red and blue turns in green. Yellow and blue turns in"

    # sink-float World knowledge
    real = "I have a boat made with wood, in water the boat will"
    corrupted = "I have a boat made with iron, in water the boat will"
    
    # Must be blank space at start
    corrupted_token = " sink"
    swapped_token = " float"
    
    check_same_size(real=real, corrupted=corrupted)

    create_embeddings(input=real, verbose=True)
    test_model(corrupted_input=corrupted,verbose=False, corrupted_token=corrupted_token, swapped_token=swapped_token)



if __name__ == '__main__':
    main()
