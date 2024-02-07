#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 09:27:01 2024

@author: nick
"""
import os
import requests
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image

# Function to extract the image name from the URL
def extract_image_name(url):
    try:
        return url.split('/')[4]
    except IndexError:
        print(f"Error extracting third part from URL: {url}")
        return None

class Text_Image_Dataset(Dataset):
    def __init__(self, config, transform=None):
        
        # Read the CSV file
        df = pd.read_csv(config['csv_path'], on_bad_lines='skip')
        
        self.image_urls = df['url'].tolist()
        self.image_captions = df['caption'].tolist()
        self.config = config
        self.transform = transform

    def __len__(self):
        return len(self.image_urls)

    def __getitem__(self, index):
        url = self.image_urls[index]
        caption = self.image_captions[index]

        if self.config['image_path'].lower() == 'url':
            image = Image.open(requests.get(url, stream=True).raw)
        else:
            image = Image.open(os.path.join(self.config['image_path'], extract_image_name(url) + '.png'))

        # Apply transform if provided
        if self.transform:
            image = self.transform(image)

        return image, caption