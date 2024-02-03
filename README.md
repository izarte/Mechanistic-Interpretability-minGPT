Mechanistic interpretability is an analysis whose purpose is to understand which embedding of each layer is understanding the information of other words in focus mechanisms.

This project is based on the [minGPT] model. To reach the object of this project, it is necessary to modify some code in [model.py].

# Requirements
execute this command to download the necessary packages

```bash
python3 -m pip install -r requirements.txt
```

# How to use

To try your examples you have to modify the [main.py] file. In this file, there is a main function. You have to modify the following variables to create a different experiment. 

variable | use 
| :---: | :---: 
real | string with the real sentence to save embeddings
corrupted | string with the corrupted sentence to make swaps 
corrupted_token | string with a blank space at the beginning with token A to compare (usually token in the corrupted sentence)
swapped_token | string with a blank space at the beginning with token B to compare (usually token in the real sentence to swap)

You can add the flag -c to just process both sentences and get their 20 most probable tokens table.


[minGPT]: https://github.com/karpathy/minGPT
[model.py]: https://github.com/izarte/Mechanistic-Interpretability-minGPT/blob/main/mingpt/model.py
[main.py]: https://github.com/izarte/Mechanistic-Interpretability-minGPT/blob/main/main.py