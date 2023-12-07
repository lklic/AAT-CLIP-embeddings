# AAT Embeddings using CLIP


## Prerequisites



```
pip install torch torchvision
pip install transformers
```



1. Unzip the AAT terms or downlaod newer ones from the getty and extract them here
   1. Run `extract-data.py`
   2. check the output file `aat_terms.csv`
2. Run the Script `generate-embeddings.py` to create a new CSV file with the embeddings.