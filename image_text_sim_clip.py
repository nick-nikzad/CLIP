#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 14:28:20 2024
 
(image, text) similarity CLIP metric

@author: Nick Nikzad
"""
import argparse
import os
import time
import pandas as pd
import numpy as np
import requests
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from memory_profiler import profile
import yaml
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# Loading config
def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config

# Use CUDA if available
def set_device(use_cuda):
    return torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")


# Function to extract the image name from the URL
def extract_image_name(url):
    try:
        return url.split('/')[4]
    except IndexError:
        print(f"Error extracting third part from URL: {url}")
        return None

@profile
def main(config_path='./config.yaml'):
    config = load_config(config_path)
    device = set_device(config["use_cuda"])

    # CLIP Model loading
    model = CLIPModel.from_pretrained(config["pretrained_model"])
    model = model.to(device)

    # Load processor: Image rescaling, normalizing, Text Tokenizer
    pre_processor = CLIPProcessor.from_pretrained(config["pretrained_model"])

    # Read the CSV file
    csv_path = config['csv_path']
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    
    image_urls = df['url'].tolist()
    image_captions = df['caption'].tolist()
    

    img_txt_similarity = []
    process_time =[]
    clip_time =[]
    for i, (url, caption) in enumerate(zip(image_urls, image_captions)):
        
        if config['image_path'].lower() == 'url':
            image = Image.open(requests.get(url,stream=True).raw)
        else:
            # Apply the function to the first column to extract the images' name
            image = Image.open(os.path.join(config['image_path'], extract_image_name(url)+'.png'))
        
        start_time = time.time()

        inputs = pre_processor(text=[caption], images=image, return_tensors="pt",
                               padding=True, truncation = True)
        inputs.to(device)
        
        process_time.append(time.time()-start_time)
        
        start_time = time.time()

        outputs = model(**inputs)
        
        clip_time.append(time.time()-start_time)
        
        similarity = outputs.logits_per_image.squeeze().detach().to('cpu').numpy()
        
        print(f' text-image similarity for image {i}: {similarity}')
        img_txt_similarity.append(similarity)
        
    df['similarity_score'] = img_txt_similarity
    
    
    print(' Avg Time taken for pre-processing (image scaling, tokenizer) = {} sec'.format(np.mean(process_time)))
    print(' Avg Time taken for CLIP metric = {} sec'.format(np.mean(clip_time)))

    # Append the updated DataFrame to the challenge_set CSV file
    df.to_csv(csv_path, header=True, index=False)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='(image, text) similarity CLIP metric')

    # path to config file
    parser.add_argument('--config', default='./config.yaml', metavar='DIR',
                        help='path to config file')

    args = parser.parse_args()
    main(config_path=args.config)