import torch

loaded_dict = torch.load('embeddings.pt')

# a = len(loaded_dict)

# print(a)

# print(loaded_dict)
print(loaded_dict.keys())
print(loaded_dict[0][:, 1, :].size())
# print(loaded_dict[1])
print(loaded_dict[1].size())
# print(len(loaded_dict[0].item()))