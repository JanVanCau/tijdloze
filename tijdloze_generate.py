from flask import Flask, render_template, request, redirect, url_for

import random
import re
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerFast

import math
import requests
import gdown

from transformer_class import InputEmbeddings, PositionalEncoding, MultiHeadAttention, FeedForwardSubLayer, DecoderLayer, Decoder

fast_tokenizer = PreTrainedTokenizerFast.from_pretrained("model_7_tokenizer", local_files_only=True)

transformer = Decoder(18000, 256, 4, 4, 512, 0.1, 512) 
transformer.load_state_dict(torch.load("model_7_weights.pth", map_location=torch.device('cpu')))


def split_positions(n, parts):
    base = n // parts
    remainder = n % parts

    numbers = [base + 1] * remainder + [base] * (parts - remainder)
    positions = [sum(numbers) - sum(numbers[i:]) for i in range(len(numbers))]
    positions.append(n)
    
    return positions

def break_lines_and_capitalize(text, max_words=13):

    chunks = text.split('\n')
    new_chunks = []
    
    for i in range(len(chunks)):

        l = len(chunks[i].split())
        
        if l > max_words:

            parts = math.ceil(l/max_words)
            positions = split_positions(l, parts)
            
            for j in range(len(positions)-1):
                spl_chunk = ' '.join(chunks[i].split()[positions[j]:positions[j+1]])
                spl_chunk = spl_chunk[0].upper() + spl_chunk[1:]
                new_chunks.append(spl_chunk)
                
            #for j in range(0, l, max_words):
                #spl_chunk = ' '.join(chunks[i].split()[j:min(j+max_words, l)])
                #spl_chunk = spl_chunk[0].upper() + spl_chunk[1:]
                #new_chunks.append(spl_chunk)
        
        else:
            new_chunks.append(chunks[i])

    return '\n'.join(new_chunks)

def paragraph(text, max_lines=6):

    s = 0
    positions = []
    p_text = text
    
    for i in range(1, len(text)-1):
        if text[i:i+1] == "\n":
            s += 1
            if s % max_lines == 0:
                positions.append(i+1)
    
    for pos in sorted(positions, reverse=True):
        p_text = p_text[:pos] + '\n' + p_text[pos:]

    return p_text


def generate_lousy_text(model, tokenizer, max_length=200, temperature=1.0, top_k=50, start_text=None):
    
    model.eval()

    if start_text:
        input_ids = tokenizer.encode(start_text)
    else:
        input_ids = [tokenizer.convert_tokens_to_ids("\n")]

    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)  # [1, seq_len]
    generated = input_ids.copy()

    for _ in range(max_length):
        
        seq_length = input_tensor.size(1)
        mask = (1-torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1))
        
        with torch.no_grad():
            outputs = model(input_tensor, mask)
            next_token_logits = outputs[0, -1, :]  # take the last time step's logits

        for n in [1, 2, 3]:  # block 1-grams, 2-grams, 3-grams
            if len(generated) >= n:
                ngram_prefix = tuple(generated[-(n-1):]) if n > 1 else ()
                banned_tokens = set()

            if n == 1:
                banned_tokens.add(generated[-1])
            else:
                for i in range(len(generated) - n + 1):
                    if tuple(generated[i:i + n - 1]) == ngram_prefix:
                        banned_tokens.add(generated[i + n - 1])

            next_token_logits[list(banned_tokens)] = float('-inf')   # Set banned token logits to -inf
            

        logits = next_token_logits / temperature

        unk_token_id = tokenizer.convert_tokens_to_ids("[UNK]")
        logits[unk_token_id] = float('-inf')  # Block [UNK] from being sampled

        if top_k > 0:
            topk_vals, topk_indices = torch.topk(logits, top_k)
            probs = F.softmax(topk_vals, dim=-1)
            next_token = topk_indices[torch.multinomial(probs, 1)].item()
        else:
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

        generated.append(next_token)
        input_tensor = torch.tensor(generated, dtype=torch.long).unsqueeze(0)

    text = tokenizer.decode(generated)
    cleaned_text = re.sub(r'\r\n?|\(|\)', '\n', text)
    cleaned_text = re.sub(r'\n\s*', '\n', cleaned_text)
    cleaned_text = re.sub(r'\n{2,}', '\n', cleaned_text)

    final_text = break_lines_and_capitalize(cleaned_text, max_words=12)
    final_text = paragraph(final_text, max_lines=6)
        
    return final_text 



#def download_model(url, output_path):
    #response = requests.get(url, stream=True)
    #if response.status_code == 200:
        #with open(output_path, 'wb') as f:
            #for chunk in response.iter_content(1024 * 1024):  # 1MB chunks
                #f.write(chunk)
        #print(f"Model downloaded to {output_path}")
    #else:
        #raise Exception(f"Failed to download model. Status code: {response.status_code}")

#url = "https://drive.google.com/uc?id=1k0gKmQKZIJpSyC-M6ReTGj168ljvEUKZ"
#output_path = "gpt2-finetuned/model.safetensors"
#download_model(url, output_path)

#gdown.download(id="1k0gKmQKZIJpSyC-M6ReTGj168ljvEUKZ", output="gpt2-finetuned/model.safetensors", quiet=False)

model = AutoModelForCausalLM.from_pretrained("JanVanCau/distilgpt2-finetuned")
tokenizer = AutoTokenizer.from_pretrained("JanVanCau/distilgpt2-finetuned")

def generate_safe_text(model, tokenizer, max_length=150, temperature=1.0, top_k=50, top_p = 0.95, start_text=None):

    if start_text:
        encoding = tokenizer(start_text, return_tensors="pt")
    else:
        encoding = tokenizer(tokenizer.eos_token, return_tensors="pt")
        
    output_ids = model.generate(
        input_ids = encoding["input_ids"],
        attention_mask = encoding["attention_mask"],
        max_length = max_length,
        temperature = temperature,
        top_k = top_k,
        top_p = top_p,
        do_sample = True,
        repetition_penalty = 1.2,
        pad_token_id = tokenizer.eos_token_id)

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    cleaned_text = re.sub(r'(?<![ \n])([A-Z])', r'\n\1', text)

    final_text = break_lines_and_capitalize(cleaned_text, max_words=12)
    final_text = paragraph(final_text, max_lines=6)
    
    return final_text


generator = Flask(__name__)

@generator.route('/', methods=['GET', 'POST'])
def index():

    result = ''
    input_text = ''

    if request.method == 'POST':

        input_text = request.form.get('provided_text', '')
        action = request.form.get('action')
    
        if action == 'lousy_model':
            try:
                result = generate_lousy_text(model=transformer, tokenizer=fast_tokenizer, start_text=input_text, temperature=0.8, top_k=50)
            except Exception as e:
                result = f"Error generating text: {str(e)}"
        
        elif action == 'safe_model':
            try:
                result = generate_safe_text(model, tokenizer, max_length=150, temperature=1.0, top_k=50, top_p = 0.95, start_text=input_text)
            except Exception as e:
                result = f"Error generating text: {str(e)}"

    return render_template('index.html', gen_text=result, prov_text=input_text)

if __name__ == '__main__':
    generator.run(debug=True)
