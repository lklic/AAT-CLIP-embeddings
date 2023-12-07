from dotenv import load_dotenv
import os
from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, Index, utility
import numpy as np
import pandas as pd
import csv

# Load environment variables from .env file
load_dotenv()

# Read Milvus server credentials from environment variables
milvus_host = os.getenv('MILVUS_HOST', 'localhost')
milvus_port = os.getenv('MILVUS_PORT', '19530')
milvus_username = os.getenv('MILVUS_USERNAME', 'default_username')
milvus_password = os.getenv('MILVUS_PASSWORD', 'default_password')

# Connect to Milvus server with authentication
connections.connect("default", host=milvus_host, port=milvus_port, user=milvus_username, password=milvus_password)

# Define the collection schema
collection_name = "aat_CLIP"
dim = 512  # Dimension of embeddings
id_field = FieldSchema(name="Subject_ID", dtype=DataType.INT64, is_primary=True)
text_field = FieldSchema(name="Combined_Text", dtype=DataType.VARCHAR, max_length=5000)
embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
schema = CollectionSchema(fields=[id_field, text_field, embedding_field], description="AAT terms with CLIP embeddings")

# Check if the collection already exists
if collection_name in utility.list_collections():
    collection = Collection(name=collection_name)
    print(f"Collection '{collection_name}' already exists.")
else:
    # Create the collection
    collection = Collection(name=collection_name, schema=schema)
    print(f"Collection '{collection_name}' created.")

# Prepare the data from CSV using pandas DataFrame
data = {"Subject_ID": [], "Combined_Text": [], "embedding": []}

with open('aat_terms_with_embeddings.csv', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header

    for row in reader:
        subject_id = int(row[0])
        combined_text = row[1]
        embedding = list(map(float, eval(row[2])))
        data["Subject_ID"].append(subject_id)
        data["Combined_Text"].append(combined_text)
        data["embedding"].append(embedding)

# Convert to pandas DataFrame
df = pd.DataFrame(data)

# Batch insert settings
batch_size = 1000  # inserts in batches of 1000
total_records = len(df)

print(f"Total records to process: {total_records}")

for start_idx in range(0, total_records, batch_size):
    end_idx = min(start_idx + batch_size, total_records)
    batch_df = df.iloc[start_idx:end_idx]
    mr = collection.insert(batch_df)
    print(f"Inserted records {start_idx+1} to {end_idx}.")

# Creating an index
index_params = {
    "metric_type": "L2",  # Same as IDIOS_CLIP
    "index_type": "IVF_FLAT",  # Same as IDIOS_CLIP
    "params": {"nlist": 2048}
}
# Retrieve the collection
collection = Collection(name=collection_name)

# Create an index on the 'embedding' field
collection.create_index(field_name="embedding", index_params=index_params)
print("Index created on 'embedding' field.")


# Flush data to disk for persistence
collection.load()
print("Data flushed to disk.")

print(f"Data inserted into collection '{collection_name}'.")
