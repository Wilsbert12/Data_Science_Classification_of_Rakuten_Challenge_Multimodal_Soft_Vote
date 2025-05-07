# Home.py
import streamlit as st
from streamlit_utils import add_pagination

st.set_page_config(
    page_title="FEB25 BDS // Rakuten: Classification of eCommmerce products",
    page_icon="images/logos/rakuten-favicon.ico",
    layout="wide",
)

st.title("FEB25 BDS // Rakuten")

st.sidebar.header(":material/home: Home")
st.sidebar.image("images/logos/rakuten-logo-red-wide.svg", use_container_width=True)

# Home page content
st.write("## eCommerce Products Classification Project")
st.markdown(
    """
    This is a Streamlit app for DataScientest's FEB25 BDS project on **eCommerce product classification**.
    
    The goal of this project is to classify products from the **Rakuten dataset** into different categories.
    """
)

# Pagination and footer
st.markdown("---")
add_pagination("pages/Home.py")
st.markdown("© 2025 | Peter Stieg, Robert Wilson, Thomas Borer")
