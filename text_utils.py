import re
import pandas as pd
import html

import numpy as np

"""
Text utility functions for text cleaning and preprocessing.

This module provides functions for tasks such as:
- Removing special characters

Planned features:
- Normalizing text
- Tokenization helpers
- Stopword removal
- Lemmatization and stemming utilities

These functions are designed to prepare text data for machine learning
and natural language processing tasks.
"""

def html_cleaner(df):
    '''requires a df with description column, will remove html markup from description column'''
    from bs4 import BeautifulSoup
    df['description'] = df['description'].apply(lambda x: BeautifulSoup(x, "html.parser").get_text() if pd.notna(x) else np.nan)


def text_cleaner(df):
    """
    Clean text in pandas DataFrame by removing unnecessary characters and normalizing spacing.

    Cleans 'designation' and 'description' columns and stores results in 
    'designation_cleaned' and 'description_cleaned' columns. Also adds boolean flag columns
    to track text characteristics.

    Removes:
    - Leading and trailing spaces 
    - HTML tags, e.g. \<br>, \<br />, \<b>

    Replaces:
    - Multiple spaces with a single space
    - HTML entities with their corresponding characters, e.g. &eacute; → è, &auml; → ä, &ntilde; → ñ 
    - Control characters with empty strings, e.g. Ã, Â©, �
    - Non-ASCII characters with empty strings
    
    Args:
        df (pandas.DataFrame): The DataFrame containing text to clean
        
    Returns:
            pandas.DataFrame: The original DataFrame with additional columns:
                - designation_cleaned, description_cleaned: Cleaned text columns
                - designation_org_clean, description_org_clean: Overall cleanliness flags
                - Additional feature flags for HTML tags, HTML flags, spaces, uppercase, and lowercase text
    """

    # Create a copy to avoid modifying the original DataFrame
    df_clean = df.copy()
    
    # Create new columns for cleaned text
    df_clean['designation_cleaned'] = ''
    df_clean['description_cleaned'] = ''

    # Create flag columns
    for col in ['designation', 'description']:
        
        df_clean[f'{col}_org_clean'] = True  # Assume clean until proven dirty
        
        df_clean[f'{col}_spaces'] = False
        df_clean[f'{col}_upper'] = False
        df_clean[f'{col}_lower'] = False

        df_clean[f'{col}_html_tag'] = False
        df_clean[f'{col}_html_entity'] = False
        
        df_clean[f'{col}_encoding_issue'] = False
        df_clean[f'{col}_control_chars'] = False

    # Map original columns to their cleaned counterparts
    col_mapping = {
        'designation': 'designation_cleaned',
        'description': 'description_cleaned'
    }

    # Process each row
    for idx, row in df_clean.iterrows():
        for orig_col, clean_col in col_mapping.items():
            if orig_col not in row or pd.isna(row[orig_col]):
                continue
                
            text = str(row[orig_col])
            
            # Check uppercase
            if text.isupper():
                df_clean.at[idx, f'{orig_col}_upper'] = True
            
            # Check lowercase
            if text.islower():
                df_clean.at[idx, f'{orig_col}_lower'] = True
            
            # Check HTML tags
            if re.search(r'<[^>]+>', text):
                df_clean.at[idx, f'{orig_col}_html_tag'] = True
                df_clean.at[idx, f'{orig_col}_org_clean'] = False

            # Check HTML entities
            if re.search(r'&[a-zA-Z0-9#]+;', text):
                df_clean.at[idx, f'{orig_col}_html_entity'] = True
                df_clean.at[idx, f'{orig_col}_org_clean'] = False

            # Check for potential encoding issues
            if re.search(r'Ã.|\xef\xbf\xbd|�', text):
                df_clean.at[idx, f'{orig_col}_encoding_issue'] = True
                df_clean.at[idx, f'{orig_col}_org_clean'] = False

            # Check for control characters (except common whitespace)
            if re.search(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', text):
                df_clean.at[idx, f'{orig_col}_control_chars'] = True
                df_clean.at[idx, f'{orig_col}_org_clean'] = False

            # Check extra spaces
            if re.search(r'^\s{1,}|\s{1,}$|\s{2,}', text):
                df_clean.at[idx, f'{orig_col}_spaces'] = True
                df_clean.at[idx, f'{orig_col}_org_clean'] = False
            
            
            # Clean text
            cleaned = text
            cleaned = re.sub(r'<[^>]+>', ' ', cleaned)  # Remove HTML tags
            
            cleaned = html.unescape(cleaned)            # Decode HTML entities and convert to characters
            cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', cleaned)  # Remove control chars
            
            cleaned = re.sub(r'^\s+|\s+$', '', cleaned) # Remove leading/trailing spaces
            cleaned = re.sub(r'\s{2,}', ' ', cleaned)   # Replace multiple spaces
            
            
            # Update the cleaned text column
            df_clean.at[idx, clean_col] = cleaned             

    return df_clean


def text_merger(df):
    """
    Merge text columns and set flags for empty or identical columns.
    
    This function:
    1. Checks if 'designation' or 'description' are empty and sets flags
    2. Checks if 'designation' or 'description' are identical and sets a flag
    3. Checks if cleaned versions 'designation_cleaned' or 'description_cleaned' are identical and sets a flag
    4. Creates a merged 'text_merged' column combining designation and description
    
    Args:
        df (pandas.DataFrame): DataFrame containing designation and description columns
        
    Returns:
        pandas.DataFrame: Original DataFrame with additional columns:
            - text_merged: Merged text from designation and description
            - designation_empty: Flag indicating if designation is empty
            - description_empty: Flag indicating if description is empty
            - identical_original: Flag indicating if original columns are identical
            - identical_cleaned: Flag indicating if cleaned columns are identical
    """
    # Create a copy to avoid modifying the original DataFrame
    df_merge = df.copy()
    
    # Create flag columns
    df_merge['designation_empty'] = False
    df_merge['description_empty'] = False
    df_merge['identical_original'] = False
    df_merge['identical_cleaned'] = False
    
    # Create merged text column
    df_merge['text_merged'] = ''
    
    # Process each row
    for idx, row in df_merge.iterrows():
        # Check for empty columns
        if 'designation_cleaned' not in row or pd.isna(row['designation_cleaned']):
            df_merge.at[idx, 'designation_empty'] = True
            
        if 'description_cleaned' not in row or pd.isna(row['description_cleaned']):
            df_merge.at[idx, 'description_empty'] = True
        
        # Get cleaned designation and cleaned description and handle missing values
        designation = str(row['designation_cleaned']) if 'designation_cleaned' in row and not pd.isna(row['designation_cleaned']) else ''
        description = str(row['description_cleaned']) if 'description_cleaned' in row and not pd.isna(row['description_cleaned']) else ''
        
        # Check if original columns are identical – if both are non-empty
        if designation and description and designation == description:
            df_merge.at[idx, 'identical_original'] = True
        
        # Check if cleaned columns are identical – if both exist
        if ('designation_cleaned' in row and 'description_cleaned' in row and 
            not pd.isna(row['designation_cleaned']) and not pd.isna(row['description_cleaned'])):
            if row['designation_cleaned'] == row['description_cleaned']:
                df_merge.at[idx, 'identical_cleaned'] = True
        
        # Create merged text
        if df_merge.at[idx, 'description_empty']:
            df_merge.at[idx, 'text_merged'] = designation
        elif df_merge.at[idx, 'identical_original']:
            df_merge.at[idx, 'text_merged'] = designation
        else:
            df_merge.at[idx, 'text_merged'] = designation + ' // ' + description
    
    return df_merge


def count_matching_words(str1, str2):
    '''count_matching_words(string1, string2) will return the percentage rounded to 2 decimal places'''
    words1 = str1.lower().split()  
    words2 = str2.lower().split()
    common_words = [word for word in words1 if word in words2]  # Find matching words
    return round(len(common_words) / len(words2) * 100, 2) if words1 else 0  # Avoid division by zero


def clean_description(str1, str2, cutoff = 95):
    if pd.notna(str2):  # Check if str2 is not NaN
            if count_matching_words(str1, str2) > cutoff:
                return np.nan  # Remove highly similar descriptions
    return str2  # Keep the original if not highly similar

def remove_duplicate_description_information(df, cutoff=95):
    '''removes information from description column if the information in description is already in designation'''
    df['description'] = df.apply(lambda row: clean_description(row['designation'], row['description'], cutoff=cutoff), axis=1)

def hw():
    """Test function to print "Hello, world!".
    
    This function serves as a placeholder to demonstrate the module's structure.
    """
    print("Hello, world!")
