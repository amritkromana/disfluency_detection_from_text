import os, argparse
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizerFast, BertForTokenClassification

labels = ['FP', 'RP', 'RV', 'RS', 'PW']

def run_language_based(text_file):

    # Load the text 
    with open(text_file, 'r') as fp: 
        text = fp.readline() 
        text = text.strip('\n') 

    # Tokenize the text
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    tokens = tokenizer(text, return_tensors="pt")
    input_ids = tokens['input_ids']

    # Initialize Bert model and load in pre-trained weights
    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=5)
    model.load_state_dict(torch.load('language.pt', map_location='cpu'))
    print('loaded finetuned language model') 

    # Get Bert output at the word-level
    output = model.forward(input_ids=input_ids)
    probs = torch.sigmoid(output.logits)
    preds = (probs > 0.5).int()[0][1:-1]

    # Convert Bert word-level output to a dataframe with word timestamps
    df = pd.DataFrame(preds, columns=labels).astype(int)
    df.insert(loc=0, column='word', value=tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])[1:-1])

    return df


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file', type=str, default=None, required=True, help='path to .txt file containing one line')
    parser.add_argument('--output_file', type=str, default=None, required=True, help='path to output .csv')
    
    args = parser.parse_args()
    
    # Get predictions
    df = run_language_based(args.input_file)

    # Save output
    df.to_csv(args.output_file)

