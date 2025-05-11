# Data Preview
import streamlit as st
from streamlit_utils import add_pagination

st.set_page_config(
    page_title="FEB25 BDS // Data Preview",
    page_icon="images/logos/rakuten-favicon.ico",
    layout="wide",
)

st.progress(3 / 7)
st.title("Data Preview")
st.sidebar.header(":material/search: Data Preview")
st.sidebar.image("images/logos/rakuten-logo-red-wide.svg", use_container_width=True)

# Add your data exploration code here
st.write("Welcome to the Data Preview page!")

# Example section
st.header("Section 1")
st.write("This is a sample section for our data exploration content.")

# You can add more sections, visualizations, and interactions below

# Example data loading section
st.header("Data Loading")
st.code(
    """
import pandas as pd

# Load your data
data = pd.read_csv('your_data.csv')
"""
)

# Example data preview section
st.header("Data Preview")
st.write("Add code to display your dataframe here")


# Pagination and footer
st.markdown("---")
add_pagination("pages/3_Data_Preview.py")
st.markdown("© 2025 | Peter Stieg, Robert Wilson, Thomas Borer")
