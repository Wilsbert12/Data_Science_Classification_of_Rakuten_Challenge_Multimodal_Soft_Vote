# Data Preprocessing
import streamlit as st

import numpy as np
import pandas as pd

from PIL import Image
from text_utils import text_cleaner
from image_utils import preprocess_image
from streamlit_utils import add_pagination_and_footer, display_image, load_DataFrame

st.set_page_config(
    page_title="FEB25 BDS // Preprocessing",
    page_icon="images/logos/rakuten-favicon.ico",
    layout="wide",
)


# Constants
BUCKET_NAME = "feb25_bds_classification-of-rakuten-e-commerce-products"
GOOGLE_CLOUD_STORAGE_URL = "https://storage.googleapis.com"
GCP_PROJECT_URL = f"{GOOGLE_CLOUD_STORAGE_URL}/{BUCKET_NAME}"

# File names and URLs of DataFrames
DF_TEXT_TRAIN_CLEAN_FN = "df_text_train_clean.parquet"  # FN as in "file name"
DF_TEXT_TRAIN_CLEAN_URL = f"{GCP_PROJECT_URL}/{DF_TEXT_TRAIN_CLEAN_FN}"

DF_IMAGE_TRAIN_FN = "df_image_train.parquet"  # FN as in "file name"

# Load DataFrames
with st.spinner(
    "Loading **DataFrame with Text Data**: Duration: approx. **5s**", show_time=True
):
    df_text_train_clean = load_DataFrame(DF_TEXT_TRAIN_CLEAN_URL)
    df_text_preprocessing = df_text_train_clean[["designation", "description"]]

with st.spinner("Loading **DataFrame with Image Data**", show_time=True):
    df_image_train = load_DataFrame(DF_IMAGE_TRAIN_FN)
    df_image_train_preprocessing = df_image_train[["imageid", "designation"]]

st.progress(5 / 7)
st.title("Data Preprocessing")
st.sidebar.header(":material/rule: Data Preprocessing")
st.sidebar.image("images/logos/rakuten-logo-red-wide.svg", use_container_width=True)

st.markdown(
    """
    Showcasing of preprocessing steps necessary for...
    - **strings** contained in columns `designation` and `description`
    - **images** available in the directories 'image_train/' and 'image_test/'
    """
)

with st.expander("**Options** for product data preview"):

    # Create a horizontal container with columns
    tpt_col1, tpt_col2 = st.columns([2, 3])  # tpt_ as in "text preprocessing tab"

    # Number input in the first column
    with tpt_col1:

        nr_of_products = st.number_input(
            "**Number** of products",
            value=3,
            min_value=1,
            max_value=48,
            step=3,
            help="Select the number _n_ for product preview. Recommended value is a multiple of 3.",
        )

    # Place checkboxes in remaining columns
    with tpt_col2:
        sort_option = st.radio(
            "**Sorting** option:",
            ["First products", "Random products", "Last products"],
            horizontal=True,
            help="Select the sorting option for product preview. Either first _n_ products, random _n_ products or last _n_ products.",
        )

tab_text, tab_image, tab_showcase = st.tabs(
    ["Text preprocessing", "Image preprocessing", "Try live functions"]
)

with tab_text:

    # Selection text preprocessing step
    text_preprocessing_option = st.selectbox(
        "Select preprocessing step for strings:",
        (
            "Original text data",
            "1. Remove HTML tags and attributes",
            "2. Remove URLs, e.g. Amazon, AWS servers, etc.",
            "3. Remove control characters and problematic whitespace, e.g. '\x00-'",
            "4. Remove multiple question marks and inverted question marks",
            "5. Remove parentheses and quotes, e.g. (), [], {}",
        ),
    )

    if text_preprocessing_option == "1. Remove HTML tags and attributes":

        mask_html = (df_text_train_clean["description_html_tag"] == 1) | (
            df_text_train_clean["description_html_entity"] == 1
        )

        df_text_preprocessing = df_text_train_clean[mask_html][
            [
                "description",
                "description_cleaned",
                "description_html_tag",
                "description_html_entity",
            ]
        ]

    elif text_preprocessing_option == "2. Remove URLs, e.g. Amazon, AWS servers, etc.":

        mask_url = (
            (df_text_train_clean["description_URL"] == 1)
            & (df_text_train_clean["description_html_tag"] == 0)
            & (df_text_train_clean["description_html_entity"] == 0)
        )

        df_text_preprocessing = df_text_train_clean[mask_url][
            ["description", "description_cleaned", "description_URL"]
        ]

    elif (
        text_preprocessing_option
        == "3. Remove control characters and problematic whitespace, e.g. '\x00-'"
    ):
        mask_control_chars = (
            (df_text_train_clean["description_control_chars"] == 1)
            & (df_text_train_clean["description_URL"] == 0)
            & (df_text_train_clean["description_html_tag"] == 0)
            & (df_text_train_clean["description_html_entity"] == 0)
        )

        df_text_preprocessing = df_text_train_clean[mask_control_chars][
            ["description", "description_cleaned", "description_control_chars"]
        ]

    elif (
        text_preprocessing_option
        == "4. Remove multiple question marks and inverted question marks"
    ):
        mask_question_marks = df_text_train_clean["description_question_marks"] == 1

        df_text_preprocessing = df_text_train_clean[mask_question_marks][
            [
                "description",
                "description_cleaned",
                "description_question_marks",
                "description_question_mark_char_count",
            ]
        ]

        df_text_preprocessing = df_text_preprocessing.sort_values(
            by="description_question_mark_char_count", ascending=False
        )

    elif (
        text_preprocessing_option == "5. Remove parentheses and quotes, e.g. (), [], {}"
    ):
        mask_dashes = (
            (df_text_train_clean["description_parentheses"] == 1)
            & (df_text_train_clean["description_URL"] == 0)
            & (df_text_train_clean["description_html_tag"] == 0)
            & (df_text_train_clean["description_html_entity"] == 0)
        )

        df_text_preprocessing = df_text_train_clean[mask_dashes][
            ["description", "description_cleaned", "description_parentheses"]
        ]

    # Display the DataFrame with text data
    rows_df_txt = df_text_preprocessing.shape[0]

    if sort_option == "First products":
        df_text_preprocessing = df_text_preprocessing.head(nr_of_products)
        st.dataframe(df_text_preprocessing)
    elif sort_option == "Last products":
        df_text_preprocessing = df_text_preprocessing.tail(nr_of_products)
        st.dataframe(df_text_preprocessing)
    elif sort_option == "Random products":
        df_text_preprocessing = df_text_preprocessing.sample(nr_of_products)
        st.dataframe(df_text_preprocessing)


with tab_image:

    # Text model section
    image_preprocessing_option = st.selectbox(
        "Select preprocessing step for images:",
        (
            "Original image data",
            "1. Basic image data extraction",
            "2. Bounding box detection",
            "3. Crop, pad and resize",
            "(4. Duplicate search)",
        ),
    )

    if image_preprocessing_option == "1. Basic image data extraction":
        df_image_train_preprocessing = df_image_train[
            [
                "imageid",
                "designation",
                "prdtypecode",
                "file_size_kb",
                "mean_r",
                "mean_g",
                "mean_b",
            ]
        ]
    elif image_preprocessing_option == "2. Bounding box detection":
        df_image_train_preprocessing = df_image_train[
            [
                "imageid",
                "designation",
                "prdtypecode",
                "bb_x",
                "bb_y",
                "bb_w",
                "bb_h",
                "bb_ar",
            ]
        ]
    elif image_preprocessing_option == "3. Crop, pad and resize":
        df_image_train_preprocessing = df_image_train[
            [
                "imageid",
                "designation",
                "prdtypecode",
                "bb_x",
                "bb_y",
                "bb_w",
                "bb_h",
                "bb_ar",
                "downscale",
                "upscale",
                "exclude",
            ]
        ]
    elif image_preprocessing_option == "(4. Duplicate search)":
        df_image_train_preprocessing = df_image_train[
            [
                "imageid",
                "designation",
                "prdtypecode",
                "phash",
                "phash_duplicate",
            ]
        ]

    # Display the DataFrame with image data
    rows_df_img = df_image_train_preprocessing.shape[0]

    if sort_option == "First products":
        df_image_train_preprocessing = df_image_train_preprocessing.head(nr_of_products)
        st.dataframe(df_image_train_preprocessing)
    elif sort_option == "Last products":
        df_image_train_preprocessing = df_image_train_preprocessing.tail(nr_of_products)
        st.dataframe(df_image_train_preprocessing)
    elif sort_option == "Random products":
        df_image_train_preprocessing = df_image_train_preprocessing.sample(
            nr_of_products
        )
        st.dataframe(df_image_train_preprocessing)

    # Display table containing original images and ...
    # ... images with bounding boxes or
    # ... cropped, padded and resized images

    # Iterate over the rows of the DataFrame
    image_data = []

    for product_id, row in df_image_train_preprocessing.iterrows():
        product_id = str(product_id)
        image_id = str(int(row["imageid"]))
        product_title = str(row["designation"])
        image_phash = str(row["phash"]) if "phash" in row else ""

        image_data.append(
            [
                product_id,
                image_id,
                product_title,
                image_preprocessing_option,
                image_phash,
            ]
        )

    # Create a grid layout with 3 columns
    cols = st.columns(3)

    # Display each image in its own column
    for i, image_data in enumerate(image_data):
        with cols[i % 3]:
            display_image(image_data, option="preprocessing")


with tab_showcase:

    # Showcase text cleaning function from text_utils.py
    with st.expander("**Try** text cleaning function", expanded=True):
        text = st.text_input(
            "Try text preprocessing:",
            "<p>Errors e.g. ???  spaces,     <strong>HTML tags, </strong>  and accents: “Caf&eacute; &amp; Canel&eacute;”</p>",
        )
        st.write("The cleaned text is:")
        cleaned_text = text_cleaner(text)
        st.code(cleaned_text, language="None")

    # Showcase image preprocessing function from image_utils.py
    with st.expander("**Try** image preprocessing function"):

        # Upload image
        uploaded_image = st.file_uploader(
            "Upload a product image...", type=["png", "jpg", "jpeg"]
        )

        if uploaded_image is not None:

            # Read the uploaded image
            image = Image.open(uploaded_image)

            # Convert to RGB mode if it's not already (handles grayscale images)
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Convert to numpy array for OpenCV processing
            image_array = np.array(image)

            img_with_bb, img_cpr, img_phash, df_uploaded_image = preprocess_image(
                image_array
            )

            # Preprocessing options
            ipt_col1, ipt_col2, ipt_col3, ipt_col4 = st.columns(
                4
            )  # ipt_ as in "image preprocessing tab"

            with ipt_col1:
                st.markdown("**Uploaded image**")
                st.image(uploaded_image, use_container_width=True)
            with ipt_col2:
                st.markdown("**Bounding box**")
                st.image(img_with_bb, use_container_width=True)
            with ipt_col3:
                st.markdown("**Crop, pad and resize**")
                st.image(img_cpr, use_container_width=True)
            with ipt_col4:
                st.markdown("**Perceptual hash**")
                st.image(img_phash, use_container_width=True)

            st.dataframe(df_uploaded_image, use_container_width=True)

# Pagination and footer
st.divider()
add_pagination_and_footer("pages/5_Data_Preprocessing.py")
