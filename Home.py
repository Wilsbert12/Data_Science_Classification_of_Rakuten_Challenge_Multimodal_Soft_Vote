# Home.py
import streamlit as st
from streamlit_utils import add_pagination_and_footer

st.set_page_config(
    page_title="FEB25 BDS // Rakuten: Classification of eCommmerce products",
    page_icon="images/logos/rakuten-favicon.ico",
    layout="wide",
)

# Add Open Graph meta tags
st.markdown(
    """
    <head>
        <meta property="og:description" content="This Streamlit app is part of the final project for DataScientist's training in **Data Science** of the cohort **FEB25 BDS**. The project addresses Rakuten's challenge of accurately **categorizing products** in the marketplace listings. One solution could be an automation via multimodal machine learning combining text and image data." />
    </head>
""",
    unsafe_allow_html=True,
)


st.progress(0 / 7)
st.title("FEB25 BDS // Rakuten")

st.sidebar.header(":material/home: Home")
st.sidebar.image("images/logos/rakuten-logo-red-wide.svg", use_container_width=True)


# Home page content
st.write("## eCommerce Products Classification Project")
st.markdown(
    """
    This Streamlit app is part of the final project for **_DataScientist_**'s training in **Data Science** of the cohort **FEB25 BDS**.
    
    The project addresses **_Rakuten_**'s challenge of accurately **categorizing products** in the marketplace listings.
    
    One solution could be an automation via **multimodal machine learning** combining text and image data.
    
    Our approach aims to leverage the company's competitive advantage by...
    * automating the **marketplace**'s product categorization, reducing operational costs and errors while simultaneously 
    * improving **user experience** thanks to a more efficient consumer product discovery as well as 
    * helping **vendors** reach their respective target audience more effectively.
    
    Use the sidebar or pagination to browse through the presentation of the project and the team, the data's exploration, visualization and preprocessing as well as pages regarding modelling and prediction.
    
    **:material/folder_code: GitHub Repository:** [feb25_bds_classification-of-rakuten-e-commerce-products](https://github.com/PeterStieg/feb25_bds_classification-of-rakuten-e-commerce-products)
    
    """
)

# Pagination and footer
add_pagination_and_footer("pages/Home.py")
