import streamlit as st
import pandas as pd

from image_utils import display_phash

# Constants
BUCKET_NAME = "feb25_bds_classification-of-rakuten-e-commerce-products"
GOOGLE_CLOUD_STORAGE_URL = "https://storage.googleapis.com"
GCP_PROJECT_URL = f"{GOOGLE_CLOUD_STORAGE_URL}/{BUCKET_NAME}"

PAGE_SEQUENCE = [
    {"name": "Home", "path": "Home.py"},
    {"name": "1. Project Presentation", "path": "pages/1_Project_Presentation.py"},
    {"name": "2. Team Presentation", "path": "pages/2_Team_Presentation.py"},
    {"name": "3. Data Exploration", "path": "pages/3_Data_Exploration.py"},
    {"name": "4. Data Visualization", "path": "pages/4_Data_Visualization.py"},
    {"name": "5. Data Preprocessing", "path": "pages/5_Data_Preprocessing.py"},
    {"name": "6. Modelling", "path": "pages/6_Modelling.py"},
    {"name": "7. Prediction", "path": "pages/7_Prediction.py"},
    {"name": "8. Thank you", "path": "pages/8_Thank_you.py"},
]


@st.cache_data()
def load_DataFrame(URL):
    """
    Fetch a DataFrame from a given URL.

    Parameters:
    - URL: The URL to fetch the DataFrame from.

    Returns:
    - DataFrame: The fetched DataFrame.
    """
    df = pd.read_parquet(URL, engine="pyarrow")
    return df


def add_pagination(current_page_path):
    # Find current page index in sequence
    current_index = next(
        (
            i
            for i, page in enumerate(PAGE_SEQUENCE)
            if page["path"] == current_page_path
        ),
        0,
    )

    # Create columns for previous, current page indicator, next
    prev_butt, elc, pagination, erc, next_butt = st.columns(
        [2, 2, 1, 2, 2]
    )  # elf* and erc* as in "empty left column" and "empty right column"

    # Previous button
    with prev_butt:
        if current_index > 0:  # Not on first page
            prev_page = PAGE_SEQUENCE[current_index - 1]
            if st.button("← Previous", use_container_width=True):
                st.switch_page(prev_page["path"])

    # Page indicator
    with pagination:
        st.markdown(f"**Page {current_index}/{len(PAGE_SEQUENCE) - 1}**")

    # Next button
    with next_butt:
        if current_index < len(PAGE_SEQUENCE) - 1:  # Not on last page
            next_page = PAGE_SEQUENCE[current_index + 1]
            if st.button("Next →", use_container_width=True):
                st.switch_page(next_page["path"])


def display_image(id, option):  # id as in "image data"

    if option == "preprocessing":

        dataset = "train"
        pi = id[0]  # product ID
        ii = id[1]  # image ID
        pt = id[2]  # product title, aka designation
        ipo = id[3]  # image preprocessing option
        iph = id[4]  # image perceptual hash
        fais = ""

        if ipo == "2. Bounding box detection":
            fais = "_bb"
        elif ipo == "3. Crop, pad and resize":
            fais = "_cpr"
        else:
            pass

        # Create the full public URL for the image
        if ipo != "(4. Duplicate search)":
            image_path = f"image_{ii}_product_{pi}{fais}.jpg"
            image_url = f"{GCP_PROJECT_URL}/images/image_{dataset}{fais}/{image_path}"
        else:
            image_url = display_phash(iph, size=8, scale=32)

        # Display the image with product ID and image ID as caption
        st.image(
            image_url,
            caption=f"Title: {pt[:33]} - Product: {pi} - Image: {ii}",
            use_container_width=True,
        )

    # Test data: Display the image without caption
    elif option == "prediction":

        dataset = "test"
        pi = id[0]  # product ID
        ii = id[1]  # image ID
        pt = id[2]  # product title, aka designation
        fais = "_cpr"

        image_path = f"image_{ii}_product_{pi}{fais}.jpg"
        image_url = f"{GCP_PROJECT_URL}/images/image_{dataset}{fais}/{image_path}"

        st.image(
            image_url,
            use_container_width=True,
        )
