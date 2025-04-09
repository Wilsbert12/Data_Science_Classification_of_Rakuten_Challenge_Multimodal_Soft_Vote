import os
import numpy as np
import pandas as pd

from PIL import Image
from PIL.ExifTags import TAGS

"""
Image utility functions for extracting data from images.

This module provides functions for tasks such as:
- Data extraction, e. g. size, format, metadata

Planned features:
- Image processing
- Image classification

These functions are designed to prepare image data for machine learning.
"""

def image_data_extractor(df, base_path='./images/image_train/'):
    """
    Extract data from product images in given DataFrame with columns 'productid' and 'imageid'.
    Return a new DataFrame with the extracted information.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'productid' and 'imageid' columns
    base_path : str, optional
        The base directory path where images are stored, e.g. './images/image_train/'
        Default is './images/image_train/'.
        
    Returns:
    --------
    pandas.DataFrame
        A new DataFrame with 'productid', 'imageid', and columns for each extracted data point
    """

    # List to store extracted data for each image
    extracted_data = []
    
    # Process each row in the input DataFrame
    for idx, row in df.iterrows():
        # Extract productid and imageid from the DataFrame
        product_id = row['productid']
        image_id = row['imageid']
        
        # Initialize a dictionary to store data for this image
        image_data = {
            'productid': product_id,
            'imageid': image_id
        }
        
        # Construct the file path
        file_path = f'{base_path}image_{image_id}_product_{product_id}.jpg'
        
        # Check if file exists
        if not os.path.exists(file_path):
            image_data['error'] = "File not found" # f"File not found": {file_path}"
            extracted_data.append(image_data)
            continue
            
        try:
            # Open the image
            img = Image.open(file_path)
            
            # Extract basic image properties
            image_data['width'] = img.size[0] 
            image_data['height'] = img.size[1] 
            image_data['format'] = img.format
            image_data['mode'] = img.mode # RGB, L, etc.
            image_data['file_size_kb'] = os.path.getsize(file_path) / 1024
            image_data['aspect_ratio'] = img.size[0] / img.size[1] if img.size[1] > 0 else None
            
            # Extract metadata from image
            for key, value in img.info.items():
                # Handle binary data and complex objects
                if isinstance(value, (bytes, bytearray)):
                    continue  # Skip binary data
                if isinstance(value, (str, int, float, bool)) or value is None:
                    # Only include simple data types in the DataFrame
                    image_data[f'meta_{key}'] = value
            
            # Extract EXIF data
            if hasattr(img, '_getexif') and img._getexif():
                for tag_id, value in img._getexif().items():
                    tag_name = TAGS.get(tag_id, f"tag_{tag_id}")
                    # Skip binary data and complex objects
                    if isinstance(value, (bytes, bytearray)):
                        continue
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        image_data[f'exif_{tag_name}'] = value
            
            # Color analysis if the image is in RGB mode
            if img.mode == 'RGB':
                img_array = np.array(img)
                
                # Calculate mean RGB values
                mean_rgb = img_array.mean(axis=(0, 1))
                image_data['mean_r'] = mean_rgb[0]
                image_data['mean_g'] = mean_rgb[1]
                image_data['mean_b'] = mean_rgb[2]
                
                # Calculate dominant brightness
                brightness = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
                image_data['mean_brightness'] = brightness.mean()
                
                # Calculate color variance as a simple measure of image complexity
                std_rgb = img_array.std(axis=(0, 1))
                image_data['std_r'] = std_rgb[0]
                image_data['std_g'] = std_rgb[1]
                image_data['std_b'] = std_rgb[2]
                
        except Exception as e:
            image_data['error'] = str(e)
            
        finally:
            # Ensure we close the image file
            if 'img' in locals() and hasattr(img, 'close'):
                img.close()
                
        # Add this image's data to our list
        extracted_data.append(image_data)
    
    # Convert the list of dictionaries to a DataFrame
    df_image_data_extractor = pd.DataFrame(extracted_data)
    
    return df_image_data_extractor

