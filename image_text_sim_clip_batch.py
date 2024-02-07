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
import numpy as np
import torch
import yaml
import pandas as pd

from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from memory_profiler import profile
from text_image_dataset import Text_Image_Dataset
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Loading config
def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config

# Use CUDA if available
def set_device(use_cuda):
    return torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")

def to_pil_list(img_tensor_batch):
    # Convert the batch of tensor images to a list of PIL images
    to_pil_image = transforms.ToPILImage()
    pil_image_list = [to_pil_image((img_tensor * 255).byte()) for img_tensor in img_tensor_batch]
    return pil_image_list



@profile
def main(config_path='./config.yaml'):
    config = load_config(config_path)
    device = set_device(config["use_cuda"])


    # Define resize, totensor transformations 
    data_transform = transforms.Compose([
        transforms.Resize((config["img_resize"], config["img_resize"])),
        transforms.ToTensor(),
    ])
    # Create an instance of the text-image Dataset
    dataset = Text_Image_Dataset(config,transform=data_transform)
    
    
    # Use DataLoader for batch processing
    data_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, drop_last=False)
    
    # CLIP Model loading
    model = CLIPModel.from_pretrained(config["pretrained_model"])
    model = model.to(device)

    # Load processor: Image rescaling, normalizing, Text Tokenizer
    pre_processor = CLIPProcessor.from_pretrained(config["pretrained_model"])
  

    img_txt_similarity = []
    process_time =[]
    clip_time =[]
    for batch_idx, (images, captions) in enumerate(data_loader):
        start_time = time.time()
        
        inputs = pre_processor(text=list(captions), images=to_pil_list(images), return_tensors="pt",
                               padding=True, truncation=True)
        
        # load on the device
        inputs.to(device)
    
        process_time.append(time.time() - start_time)
    
        start_time = time.time()
    
        # compute the CLIP model output
        outputs = model(**inputs)
    
        clip_time.append(time.time() - start_time)
        
        ## pariwise similarity
        similarities_pairwise = outputs.logits_per_image.squeeze().detach().to('cpu').numpy()
        
        if similarities_pairwise.size>1:
            similarities = np.diag(similarities_pairwise)
            img_txt_similarity += similarities.tolist()

        else:
            img_txt_similarity.append(similarities_pairwise)
        print(f'Batch {batch_idx + 1}: Text-image similarities: {similarities}')
        
    
    avg_process_time = np.sum(process_time)/len(img_txt_similarity) ## average time pre-process per image
    avg_clip_time = np.sum(clip_time)/len(img_txt_similarity) ## average clip time per image
    print(' Avg Time taken for pre-processing (image scaling, tokenizer) = {} sec'.format(avg_process_time))
    print(' Avg Time taken for CLIP metric = {} sec'.format(avg_clip_time))


    # Read the CSV file
    csv_path = config['csv_path']
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    
    df['similarity_score_batch (img-size:'+str(config["img_resize"])+')'] = img_txt_similarity

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