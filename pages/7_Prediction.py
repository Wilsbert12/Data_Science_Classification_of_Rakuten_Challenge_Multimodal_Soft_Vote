# Prediction
import streamlit as st
import pandas as pd
from streamlit_utils import add_pagination, load_DataFrame, display_image

import os


st.set_page_config(
    page_title="FEB25 BDS // Prediction",
    page_icon="images/logos/rakuten-favicon.ico",
    layout="wide",
)

df_text_test = load_DataFrame("df_text_test.parquet")
df_text_test = df_text_test.sample(1)
row = df_text_test.iloc[0]


product_id = str(row.name)
image_id = str(row["imageid"])
product_title = str(row["designation"])
image_data = [product_id, image_id, product_title]

st.progress(7 / 8)
st.title("Prediction")
st.sidebar.header(":material/category_search: Prediction")
st.sidebar.image("images/logos/rakuten-logo-red-wide.svg", use_container_width=True)

st.markdown(
    """
    Predicting the category of a product listing based on its...
    - text data, e.g. description and/or title
    - image data
    ---
    """
)


# Create a layout with columns for the data
image_col, pid_col, iid_col, text_col = st.columns([1, 1, 1, 2])

# Display product image in first column
with image_col:
    st.write("**Product Image**")
    display_image(image_data, option="prediction")


# Display product id in second column
with pid_col:
    st.write("**Product ID**")
    st.write(product_id)


# Display image id in third column
with iid_col:
    st.write("**Image ID**")
    st.write(image_id)


# Display product details in fourth column
with text_col:
    st.write("**Product Text**")
    st.write(f"{row['designation']}")


# Optional: Display the raw data below
with st.expander("View original test data"):
    st.dataframe(pd.DataFrame(df_text_test))


# Pagination and footer
st.markdown("---")
add_pagination("pages/7_Prediction.py")
st.markdown("© 2025 | Peter Stieg, Robert Wilson, Thomas Borer")
