# AAT Embeddings using CLIP


## Prerequisites

Milvus 2.2.2

pymilvus

```
pip install lxml
pip install torch torchvision
pip install transformers
pip install pymilvus==2.2.2

```


## Extract relevant data from AAT to be used to generate embeddings

1. Unzip the AAT terms or downlaod newer ones from the getty and extract them here
2. Run `extract-data.py`
3. check the output file `aat_terms.csv`


## Generate embeddings for the output terms

-  Run the Script `generate-embeddings.py` to create a new CSV file with the embeddings.

- This will result in a new file `aat_terms_with_embeddings.csv`

## Insert the embeddings into a new Milvus collection

- Run `insert_embeddings.py` to insert the embeddings into a new Milvus collection

### Notes on generating embeddings

Runnig with an NVIDIA A10 GPU is 5-10 times faster than on my M1 mac. You can monitor the state of the GPU with:
```
watch -n 1 nvidia-smi
```

