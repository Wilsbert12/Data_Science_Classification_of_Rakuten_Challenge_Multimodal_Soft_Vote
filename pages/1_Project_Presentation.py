# Project presentation
import streamlit as st
from streamlit_utils import add_pagination

st.set_page_config(
    page_title="FEB25 BDS // Project Presentation",
    page_icon="images/logos/rakuten-favicon.ico",
    layout="wide",
)

st.progress(1 / 8)
st.title("Project Presentation")
st.sidebar.header(":material/work: Project Presentation")
st.sidebar.image("images/logos/rakuten-logo-red-wide.svg", use_container_width=True)

st.markdown(
    """
## Classification of Rakuten E-commerce Products

This project focuses on classifying Rakuten e-commerce products using both text and image data. 

The goal is to develop machine learning models that can categorize products 
based on their titles, descriptions and images.

**:material/folder_code: GitHub Repository:** [feb25_bds_classification-of-rakuten-e-commerce-products](https://github.com/PeterStieg/feb25_bds_classification-of-rakuten-e-commerce-products)
"""
)

# Pagination and footer
st.markdown("---")
add_pagination("pages/1_Project_Presentation.py")
st.markdown("© 2025 | Peter Stieg, Robert Wilson, Thomas Borer")
