import numpy as np
import pandas as pd
import json
from transformers import pipeline,AutoTokenizer

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

file_path = '/Users/jupudi/ar_process/research_agent/company_intelligence.json'
data = read_json_file(file_path)

ner_pipeline = pipeline("ner", model="dslim/bert-base-NER")
# tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")


# Helper function to convert numpy types in a nested structure
def convert_numpy_types(obj):
    """
    Convert numpy types in a nested structure to Python native types
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


# def extract_entities(text):
#     # Get raw NER results
#     ner_results = ner_pipeline(text)
    
#     # Group entities that were split by tokenization
#     grouped_entities = []
#     current_entity = None
    
#     for item in ner_results:
#         # Check if this is a continuation of the previous entity (B- vs I-)
#         if current_entity and item['entity'].startswith('I-') and current_entity['entity'][2:] == item['entity'][2:]:
#             # Update the current entity
#             current_entity['word'] += item['word'].replace('##', '')
#             current_entity['end'] = item['end']
#             # Average the scores (or keep the higher one)
#             current_entity['score'] = (current_entity['score'] + item['score']) / 2
#         else:
#             # If we had a current entity, add it to our results
#             if current_entity:
#                 grouped_entities.append(current_entity)
            
#             # Start a new entity
#             current_entity = item.copy()
    
#     # Don't forget to add the last entity
#     if current_entity:
#         grouped_entities.append(current_entity)
    
#     # Reconstruct proper text for each entity using character spans
#     for entity in grouped_entities:
#         entity['word'] = text[entity['start']:entity['end']]
    
#     return grouped_entities


def extract_entities(text):
    """Extract named entities using Hugging Face NER pipeline with improved handling of split tokens"""
    if not text:
        return []
        
    # Process text in chunks to avoid too-long sequences
    max_length = 512
    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    
    all_entities = []
    for chunk in chunks:
        # Get raw NER results for this chunk
        raw_entities = ner_pipeline(chunk)
        
        # Convert numpy values to Python native types
        raw_entities = convert_numpy_types(raw_entities)
        
        # Group entities that were split by tokenization within this chunk
        chunk_grouped_entities = []
        current_entity = None
        
        for item in raw_entities:
            # Check if this is a continuation of the previous entity (B- vs I-)
            if current_entity and item['entity'].startswith('I-') and current_entity['entity'][2:] == item['entity'][2:]:
                # This is a continuation - update the current entity
                if 'word' in item and item['word'].startswith('##'):
                    current_entity['word'] += item['word'][2:]  # Remove the ## prefix
                else:
                    current_entity['word'] += item['word']
                    
                current_entity['end'] = item['end']
                # Use max score instead of average
                current_entity['score'] = max(current_entity['score'], item['score'])
            else:
                # If we had a current entity, add it to our results
                if current_entity:
                    # Use the original text span for the full word
                    current_entity['word'] = chunk[current_entity['start']:current_entity['end']]
                    chunk_grouped_entities.append(current_entity)
                
                # Start a new entity
                current_entity = item.copy()
        
        # Don't forget to add the last entity
        if current_entity:
            current_entity['word'] = chunk[current_entity['start']:current_entity['end']]
            chunk_grouped_entities.append(current_entity)
        
        all_entities.extend(chunk_grouped_entities)
    
    # Group entities by word and type (across chunks)
    final_grouped_entities = {}
    for entity in all_entities:
        entity_text = entity['word']
        entity_type = entity['entity']
        
        # Remove the B- or I- prefix for consistent grouping
        entity_type_clean = entity_type[2:] if entity_type.startswith(('B-', 'I-')) else entity_type
        
        key = (entity_text, entity_type_clean)
        if key in final_grouped_entities:
            final_grouped_entities[key]['count'] += 1
            final_grouped_entities[key]['score'] = max(final_grouped_entities[key]['score'], entity['score'])
        else:
            final_grouped_entities[key] = {
                'text': entity_text,
                'type': entity_type_clean,  # Use the clean type without B-/I- prefix
                'count': 1,
                'score': entity['score']
            }
    
    return list(final_grouped_entities.values())
# text = 'Jupudi Industries, Kodad - Service Provider of Pipe Bending Service & Plate Work from Kodad'
# entities = extract_entities(text)
    
for item in data:
    item['entities'] = extract_entities(item['text'])

output_file_path = '/Users/jupudi/ar_process/research_agent/company_intelligence_with_entities.json'
with open(output_file_path, 'w') as output_file:
        json.dump(data, output_file, indent=4)


