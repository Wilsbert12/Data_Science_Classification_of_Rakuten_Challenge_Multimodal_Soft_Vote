# Data Visualization
import streamlit as st
from streamlit_utils import add_pagination

st.set_page_config(
    page_title="FEB25 BDS // Data Visualization",
    page_icon="images/logos/rakuten-favicon.ico",
    layout="wide",
)

st.progress(4 / 8)
st.title("Data Visualization")
st.sidebar.header(":material/query_stats: Data Visualization")
st.sidebar.image("images/logos/rakuten-logo-red-wide.svg", use_container_width=True)

# Add your data visualization code here
st.write("Welcome to the Data Visualization page!")

# Example section
st.header("Section 1")
st.write("This is a sample section for our data visualization content.")

# Example visualization section
st.header("Visualizations")
st.write("Add your visualization code here")
st.code(
    """
import matplotlib.pyplot as plt
import seaborn as sns

# Create a visualization
fig, ax = plt.subplots()
# your visualization code
st.pyplot(fig)
"""
)


# Pagination and footer
st.markdown("---")
add_pagination("pages/4_Data_Visualization.py")
st.markdown("© 2025 | Peter Stieg, Robert Wilson, Thomas Borer")
