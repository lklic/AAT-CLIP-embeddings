from lxml import etree
import csv

def save_to_csv(data, file_name):
    with open(file_name, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Subject ID", "Combined Text"])
        for item in data:
            writer.writerow(item)

def extract_data(xml_path):
    # Parse the XML with explicit namespace definition
    with open(xml_path, 'r', encoding='utf-8') as file:
        xml_content = file.read()
        # Remove the XML declaration if present
        xml_content = xml_content.replace('<?xml version="1.0" encoding="UTF-8" ?>', '')

        # Add the xsi namespace if missing
        xml_content = xml_content.replace('xsi:noNamespaceSchemaLocation', 'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation')

    tree = etree.fromstring(xml_content.encode('utf-8'))  # Encoding the content to bytes
    root = tree

    nsmap = root.nsmap if root.nsmap else {}

    terms_data = []

    for subject in root.findall('.//Subject', namespaces=nsmap):
        subject_id = subject.get('Subject_ID')
        preferred_term = subject.find('Terms/Preferred_Term/Term_Text').text if subject.find('Terms/Preferred_Term/Term_Text') is not None else ""
        hierarchy = subject.find('Parent_Relationships/Preferred_Parent/Parent_String').text if subject.find('Parent_Relationships/Preferred_Parent/Parent_String') is not None else ""
        descriptive_note = ""
        record_type = subject.find('Record_Type').text if subject.find('Record_Type') is not None else ""

        # Extract English descriptive note
        for note in subject.findall('Descriptive_Notes/Descriptive_Note'):
            note_language = note.find('Note_Language')
            if note_language is not None and note_language.text == "English":
                descriptive_note = note.find('Note_Text').text
                break

        # Concatenate fields for embedding
        
        # Here is an extended description. CLIP wants only 77 tokens per term so I am removing some extra data.
        # combined_text = f"{preferred_term}. Hierarchy: {hierarchy}. Note: {descriptive_note}. Type: {record_type}"
        combined_text = f"{preferred_term}. Description: {descriptive_note}"
        terms_data.append((subject_id, combined_text))

    return terms_data

aat_data = extract_data('AAT.xml')
save_to_csv(aat_data, 'aat_terms.csv')
