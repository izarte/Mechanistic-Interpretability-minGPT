import torch

loaded_dict = torch.load('embeddings.pt')

print(loaded_dict[0])