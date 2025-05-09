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
product_title = row["designation"]
product_description = row["description"]
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
    """
)

# Prediction content
prediction_tab1, prediction_tab2 = st.tabs(["Random test data", "User test data"])

with prediction_tab1:

    # Create a layout with columns for the processed data
    image_proc_col, text_col, prediction_col = st.columns([1, 2, 2])

    # Display product image in first column
    with image_proc_col:
        st.write("**Processed Image**")

        display_image(image_data, option="prediction")

    # Display product details in fourth column
    with text_col:
        st.write("**Cleaned text Data**")
        st.write(f"{row['designation']}")

    with prediction_col:
        st.write("**Predicted category**")

    # ### Create a layout with columns for the original data
    image_org_col, title_col, description_col, pid_col, iid_col = st.columns(
        [1, 1, 1, 1, 1]
    )

    # Display product image in first column
    with image_org_col:
        st.write("**Original Image**")
        display_image(image_data, option="prediction")

    # Display product details in fourth column
    with title_col:
        st.write("**Title**")
        st.write(f"{row['designation']}")

    # Display product ids in second column
    with description_col:
        st.write("**Description**")
        st.write(f"{row['description']}")

        # Display product ids in second column
    with pid_col:
        st.write("**Product ID**")
        st.write(product_id)

    # Display image id in third column
    with iid_col:
        st.write("**Image ID**")
        st.write(image_id)

    # Optional: Display the raw data below
    with st.expander("View original test data"):
        st.dataframe(pd.DataFrame(df_text_test))


with prediction_tab2:
    # Insert title
    user_product_title = st.text_input(
        "**Product title**",
        "Sony WH-1000XM4 Casque Sans Fil à Réduction de Bruit - Noir",
        help="Enter your product listing's title",
    )

    # Insert description
    user_product_description = st.text_input(
        "**Product description**",
        "Casque premium circum-aural avec réduction de bruit leader sur le marché, autonomie de 30 heures, commandes tactiles et microphone intégré pour les appels mains libres. Livré avec étui de transport, câble de charge et câble audio pour écoute filaire. Compatible avec tous les appareils Bluetooth.",
        help="Enter your product listing's description",
    )

    # Upload image
    uploaded_image = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

# Pagination and footer
st.markdown("---")
add_pagination("pages/7_Prediction.py")
st.markdown("© 2025 | Peter Stieg, Robert Wilson, Thomas Borer")
