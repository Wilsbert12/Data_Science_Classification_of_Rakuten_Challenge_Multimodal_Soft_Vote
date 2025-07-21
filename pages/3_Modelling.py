# Modelling
import streamlit as st
from streamlit_utils import add_pagination_and_footer
from streamlit_mermaid import st_mermaid

st.set_page_config(
    page_title="FEB25 BDS // Modelling",
    page_icon="images/logos/rakuten-favicon.ico",
    layout="wide",
)

st.progress(3 / 7)
st.title("Modelling")
st.sidebar.header(":material/model_training: Modelling")
st.sidebar.image("images/logos/rakuten-logo-red-wide.svg", use_container_width=True)

with open("data/methodology.mmd", "r") as file:
    methodology_mermaid = file.read()

st_mermaid(methodology_mermaid, height="auto", pan=True, zoom=True, show_controls=True)

# Pagination and footer
st.divider()
add_pagination_and_footer("pages/3_Modelling.py")
