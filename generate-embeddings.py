import torch
from transformers import CLIPProcessor, CLIPModel
import csv
import numpy as np

def save_csv_with_embeddings(filename, data):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Subject ID", "Combined Text", "CLIP Embedding"])
        for row in data:
            writer.writerow(row)

def generate_embeddings(csv_path, output_csv_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    max_length = model.config.text_config.max_position_embeddings
    data_with_embeddings = []

    with open(csv_path, newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        
        # For progress logging
        total_rows = sum(1 for row in reader)
        file.seek(0)  # Reset file pointer to the beginning after counting
        next(reader)  # Skip the header row again

        print(f"Total terms to process: {total_rows}")
        for index, row in enumerate(reader, start=1):
            text = row[1][:max_length]  # Truncate the text to the maximum length
            inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model.get_text_features(**inputs)
            embedding = outputs.detach().cpu().numpy()[0]
            data_with_embeddings.append([row[0], text, embedding.tolist()])

            if index % 100 == 0 or index == total_rows:  # Log progress every 100 terms or at the end
                print(f"Processed {index}/{total_rows} terms.")

    save_csv_with_embeddings(output_csv_path, data_with_embeddings)

csv_path = 'aat_terms.csv'
output_csv_path = 'aat_terms_with_embeddings.csv'

generate_embeddings(csv_path, output_csv_path)
