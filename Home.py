# Home.py
import streamlit as st
from streamlit_utils import add_pagination_and_footer

st.set_page_config(
    page_title="FEB25 BDS // Rakuten: Classification of eCommmerce products",
    page_icon="images/logos/rakuten-favicon.ico",
    layout="wide",
)

st.progress(0 / 7)
st.title("FEB25 BDS // Rakuten")

st.sidebar.header(":material/home: Home")
st.sidebar.image("images/logos/rakuten-logo-red-wide.svg", use_container_width=True)


# Home page content
st.write("## eCommerce Products Classification Project")
st.markdown(
    """
    This Streamlit app, developed for **DataScientest**'s FEB25 BDS project, addresses **Rakuten**'s challenge of accurately categorizing marketplace listings through **multimodal machine learning** combining text and image data.
    
    The approach aims to leverage competitive advantage by...
    * automating the **marketplace**'s product categorization, reducing operational costs and errors while simultaneously 
    * enhancing **user experience** and consumer product discovery as well as 
    * helping **vendors** reach their target customers more effectively.
    
    Use the sidebar or pagination to browse through presentations of the project and team, the data's exploration, visualization and preprocessing as well as pages regarding modelling and prediction.
    
    **:material/folder_code: GitHub Repository:** [feb25_bds_classification-of-rakuten-e-commerce-products](https://github.com/PeterStieg/feb25_bds_classification-of-rakuten-e-commerce-products)
    
    """
)

# Pagination and footer
st.markdown("---")
add_pagination_and_footer("pages/Home.py")
