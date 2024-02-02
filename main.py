import sys
import torch

from mingpt.bpe import BPETokenizer
from mingpt.model import GPT
from mingpt.utils import set_seed
from mingpt.bpe import BPETokenizer

from prettytable import PrettyTable
import matplotlib.pyplot as plt
import argparse

"""
    Function to check if real and corrupted sentences are same length after tokenization
    args:
        - real, string with real sentence
        - corrupted, string with corrupted sentence
"""
def check_same_size(real : str, corrupted : str):
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

"""
    Function to save embeddings of real sentence
    args:
        -input, sentence to save embeddings
        -verbose, bool, True for print tokenized sentence and table 
"""
def create_embeddings(input="Michelle Jones was a top-notch student. Michelle", verbose=False):
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

    # Create a table with 20 most probable tokens
    table = PrettyTable()
    table.field_names = ["Position", "Token index", "Token", "Probability"]
    for i in range(0, 20):
        table.add_row([f"{i + 1}", f"{indexes[0][i].item()}", f"{bpe.decode(torch.tensor([indexes[0][i].item()]))}", f"{values[0, i].item():.4f}"])
    with open("results/table.txt", 'w') as file:
        file.write(str(table))
    if verbose:
        print(table)

    return 

"""
    Function to get corrupted logits after a swap in each layer for each token and obtain heat map with prob difference
    args:
        - corrupted_input, string with corrupted sentence
        - verbose, bool, True for print tokenized sentence
        - corrupted_token, int, index of corrupted token. The one that differs in corrupted sentence from real sentence
        - swapped_token, int, index of real token that is swapped during execution. The one in real sentence
"""
def test_model(
        corrupted_input : str ="Michelle Smith was a top-notch student. Michelle",
        verbose : bool = False,
        corrupted_token : int = None,
        swapped_token : int = None
    ):
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
    parser = argparse.ArgumentParser(description="Create and save a table.")
    parser.add_argument('-c', '--check_tables', action='store_true', help='Print next token logits for each sentence', default=False)
    # Parse command-line arguments
    args = parser.parse_args()

    set_seed(3407)

    # Jersey-York
    real = "Every summer I go to New York. In New"
    corrupted = "Every summer I go to New Jersey. In New"
    # Must be blank space at start
    corrupted_token = " Jersey"
    swapped_token = " York"

    # green-purple
    # real = "Red and blue turns in purple. Yellow and blue turns in"
    # corrupted = "Red and blue turns in green. Yellow and blue turns in"
    # corrupted_token = " green"
    # swapped_token = " purple"

    # throat-stomach
    # real = "The patient is vomiting. The patient has pain in his"
    # corrupted = "The patient is coughing. The patient has pain in his"
    # corrupted_token = " coughing"
    # swapped_token = " vomiting"

    # sink-float World knowledge
    # real = "I have a boat made with wood, in water the boat will"
    # corrupted = "I have a boat made with iron, in water the boat will"
    # corrupted_token = " sink"
    # swapped_token = " float"
    
    if args.check_tables:
        create_embeddings(input=real, verbose=True)
        create_embeddings(input=corrupted, verbose=True)
        return

    check_same_size(real=real, corrupted=corrupted)

    create_embeddings(input=real, verbose=True)
    test_model(corrupted_input=corrupted,verbose=False, corrupted_token=corrupted_token, swapped_token=swapped_token)


if __name__ == '__main__':
    main()
