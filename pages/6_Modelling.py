# Modelling
import streamlit as st
from streamlit_utils import add_pagination

st.set_page_config(
    page_title="FEB25 BDS // Modelling",
    page_icon="images/logos/rakuten-favicon.ico",
    layout="wide",
)

st.progress(6 / 8)
st.title("Modelling")
st.sidebar.header(":material/model_training: Modelling")
st.sidebar.image("images/logos/rakuten-logo-red-wide.svg", use_container_width=True)

# Add your modelling code here
st.write("Welcome to the Modelling page!")

# Example section
st.header("Section 1")
st.write("This is a sample section for our modelling content.")

tab_text, tab_image = st.tabs(["Text classification", "Image classification"])

with tab_text:

    # Text model sectionö
    text_model_option = st.selectbox(
        "Select a model:",
        ("Linear Regression", "Random Forest", "Neural Network"),
    )

    st.write("You selected:", text_model_option)

    # Model parameters section
    st.header("Model Parameters")
    st.write("Add model parameter sliders and inputs here")


with tab_image:

    # Text model section
    image_model_option = st.selectbox(
        "Select a model:",
        ("VGG16", "ResNet50", "InceptionV3"),
    )

    st.write("You selected:", image_model_option)

    # Model parameters section
    st.header("Model Parameters")
    st.write("Add model parameter sliders and inputs here")


# Pagination and footer
st.markdown("---")
add_pagination("pages/6_Modelling.py")
st.markdown("© 2025 | Peter Stieg, Robert Wilson, Thomas Borer")
