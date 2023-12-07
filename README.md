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

My output looks like this:

```
ubuntu@idios:~/AAT-CLIP-embeddings$ python3 insert_embeddings.py 
Collection 'aat_CLIP' created.
Total records to process: 56830
Inserted records 1 to 1000.
Inserted records 1001 to 2000.
Inserted records 2001 to 3000.
Inserted records 3001 to 4000.
...
Inserted records 53001 to 54000.
Inserted records 54001 to 55000.
Inserted records 55001 to 56000.
Inserted records 56001 to 56830.
Index created on 'embedding' field.
Data flushed to disk.
Data inserted into collection 'aat_CLIP'.
```
### Notes on generating embeddings

Runnig with an NVIDIA A10 GPU is 5-10 times faster than on my M1 mac. You can monitor the state of the GPU with:
```
watch -n 1 nvidia-smi
```

