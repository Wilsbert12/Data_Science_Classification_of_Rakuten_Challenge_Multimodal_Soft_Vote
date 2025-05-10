# Prediction
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

from streamlit_utils import (
    add_pagination,
    load_DataFrame,
    display_image,
    get_img_path,
    load_vgg16,
    predict_vgg16,
)

from image_utils import preprocess_image
from text_utils import text_cleaner, text_merger

st.set_page_config(
    page_title="FEB25 BDS // Prediction",
    page_icon="images/logos/rakuten-favicon.ico",
    layout="wide",
)

# Load trained model
with st.spinner("Loading **VGG16 model**: Duration: approx. **5s**", show_time=True):
    vgg16 = load_vgg16()


# Load DataFrame with text data
with st.spinner("Loading **DataFrame with Text Data**", show_time=True):
    df_text_test = load_DataFrame("df_text_test.parquet")

# Get product IDs from DataFrame with test data
product_ids = df_text_test.index.tolist()

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
prediction_tab1, prediction_tab2, prediction_tab3 = st.tabs(
    ["Specific test data", "Random test data", "User test data"]
)

with prediction_tab1:
    with st.expander("**Options** for test data preview"):
        product_id_selected = st.selectbox("Select specific Product ID", product_ids)

    df_text_test_specific = df_text_test.loc[[product_id_selected]]
    row = df_text_test_specific.iloc[0]

    product_id = str(row.name)
    image_id = str(row["imageid"])
    product_title = str(row["designation"])
    product_description = str(row["description"])
    image_data = [product_id, image_id, product_title]

    product_title_cleaned = text_cleaner(product_title)
    product_description_cleaned = (
        text_cleaner(product_description) if product_description else ""
    )
    merged_text = text_merger([product_title_cleaned, product_description_cleaned])
    merged_text_len = len(merged_text)

    merged_text_char_limit = 240
    if merged_text_len > merged_text_char_limit:
        merged_text = f"{merged_text[:merged_text_char_limit-6]} *[...]*"

    product_title_len = len(product_title)
    product_description_len = len(product_description) if product_description else 0

    # Create a layout with columns for the processed data
    image_proc_col, text_col, prediction_col = st.columns([1, 2, 2])

    # Display product image in first column
    with image_proc_col:
        st.write("**Processed Image**")

        display_image(image_data, option="prediction_proc")

    # Display product details in fourth column
    with text_col:
        st.write("**Cleaned Text Data**")
        st.write(merged_text)

    with prediction_col:
        st.write("**Predicted category**")
        img_cpr_path = get_img_path(product_id, image_id, option="cpr")
        st.write(f"{predict_vgg16(vgg16, img_cpr_path)}")

    # ### Create a layout with columns for the original data
    image_org_col, title_col, description_col, pid_col, iid_col = st.columns(
        [1, 1, 1, 1, 1]
    )

    # Display product image in first column
    with image_org_col:
        st.write("**Original Image**")
        display_image(image_data, option="prediction_org")

    # Display product details in fourth column
    with title_col:
        st.write("**Title**")
        st.write(f"{product_title}")

    # Display product ids in second column
    with description_col:
        st.write("**Description**")
        if (product_description is None) or (
            product_description_len < product_title_len
        ):
            st.write(f"{product_description}")
        else:
            st.markdown(f"{product_description[:product_title_len-6]} *[...]*")

        # Display product ids in second column
    with pid_col:
        st.write("**Product ID**")
        st.write(product_id)

    # Display image id in third column
    with iid_col:
        st.write("**Image ID**")
        st.write(image_id)

    with st.expander("View original test data"):
        st.dataframe(pd.DataFrame(df_text_test_specific))

with prediction_tab2:

    df_text_test_sample = df_text_test.sample(1)
    row = df_text_test_sample.iloc[0]

    product_id = str(row.name)
    image_id = str(row["imageid"])
    product_title = row["designation"]
    product_description = row["description"]
    image_data = [product_id, image_id, product_title]

    product_title_cleaned = text_cleaner(product_title)
    product_description_cleaned = (
        text_cleaner(product_description) if product_description else ""
    )
    merged_text = text_merger([product_title_cleaned, product_description_cleaned])
    merged_text_len = len(merged_text)

    merged_text_char_limit = 240
    if merged_text_len > merged_text_char_limit:
        merged_text = f"{merged_text[:merged_text_char_limit-6]} *[...]*"

    product_title_len = len(product_title)
    product_description_len = len(product_description) if product_description else 0

    # Create a layout with columns for the processed data
    image_proc_col, text_col, prediction_col = st.columns([1, 2, 2])

    # Display product image in first column
    with image_proc_col:
        st.write("**Processed Image**")
        display_image(image_data, option="prediction_proc")

    # Display product details in fourth column
    with text_col:
        st.write("**Cleaned Text Data**")
        st.write(f"{merged_text}")

    with prediction_col:
        st.write("**Predicted category**")
        img_cpr_path = get_img_path(product_id, image_id, option="cpr")
        st.write(f"{predict_vgg16(vgg16, img_cpr_path)}")

    # ### Create a layout with columns for the original data
    image_org_col, title_col, description_col, pid_col, iid_col = st.columns(
        [1, 1, 1, 1, 1]
    )

    # Display product image in first column
    with image_org_col:
        st.write("**Original Image**")
        display_image(image_data, option="prediction_org")

    # Display product details in fourth column
    with title_col:
        st.write("**Title**")
        st.write(f"{product_title}")

    # Display product ids in second column
    with description_col:
        st.write("**Description**")
        if (product_description is None) or (
            product_description_len < product_title_len
        ):
            st.write(f"{product_description}")
        else:
            st.markdown(f"{product_description[:product_title_len-6]} *[...]*")

        # Display product ids in second column
    with pid_col:
        st.write("**Product ID**")
        st.write(product_id)

    # Display image id in third column
    with iid_col:
        st.write("**Image ID**")
        st.write(image_id)

    # Optional: Display the raw data below
    footer_col1, footer_col2 = st.columns([4, 1])

    with footer_col1:
        with st.expander("View original test data"):
            st.dataframe(pd.DataFrame(df_text_test_sample))

    with footer_col2:
        st.button("Change product", type="primary")


with prediction_tab3:
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

    user_product_title_cleaned = (
        text_cleaner(user_product_title) if user_product_title else ""
    )
    user_product_description_cleaned = (
        text_cleaner(user_product_description) if user_product_description else ""
    )
    user_text_merged = text_merger(
        [user_product_title_cleaned, user_product_description_cleaned]
    )

    # Upload image
    uploaded_image = st.file_uploader(
        "Upload a product image...", type=["png", "jpg", "jpeg"]
    )

    elc, button_col, erc = st.columns([2, 1, 2])

    with button_col:
        prediction_button = st.button("Predict category", type="primary")

    if prediction_button:

        st.markdown("---")

        # Create a layout to display the processed image and predictions
        pred_img_col, pred_txt_col, pred_cat_col = st.columns([1, 2, 1])

        with pred_img_col:

            if uploaded_image is not None:
                # Read the uploaded image
                image = Image.open(uploaded_image)

                # Convert to numpy array for OpenCV processing
                image_array = np.array(image)

                # Call preprocessing function
                img_with_bb, img_cpr, img_phash, df_uploaded_image = preprocess_image(
                    image_array
                )

                st.write("**Processed Image**")
                st.image(img_cpr, use_container_width=True)

            else:
                st.write("**Image**")
                st.image(
                    "images/logos/no_product_image.png",
                    use_container_width=True,
                )

        with pred_txt_col:
            st.write("**Cleaned Text Data**")
            st.write(user_text_merged)

        with pred_cat_col:

            if uploaded_image is not None:
                st.write("**Predicted Category**")
                st.write(f"{predict_vgg16(vgg16, img_cpr_path)}")
            else:
                st.write("**Predicted Category**")
                st.write("*Placeholder*")

# Pagination and footer
st.markdown("---")
add_pagination("pages/7_Prediction.py")
st.markdown("© 2025 | Peter Stieg, Robert Wilson, Thomas Borer")
