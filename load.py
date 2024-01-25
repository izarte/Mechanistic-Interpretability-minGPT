import torch

loaded_dict = torch.load('embeddings.pt')

print(loaded_dict)
# print(loaded_dict[0].size(1))
print(loaded_dict.keys())