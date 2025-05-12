# Data Overview
import streamlit as st
from streamlit_utils import add_pagination_and_footer, load_DataFrame
import time

st.set_page_config(
    page_title="FEB25 BDS // Data Overview",
    page_icon="images/logos/rakuten-favicon.ico",
    layout="wide",
)


# Constants
BUCKET_NAME = "feb25_bds_classification-of-rakuten-e-commerce-products"
GOOGLE_CLOUD_STORAGE_URL = "https://storage.googleapis.com"
GCP_PROJECT_URL = f"{GOOGLE_CLOUD_STORAGE_URL}/{BUCKET_NAME}"

# File names and URLs of DataFrames
DF_TEXT_TRAIN_CLEAN_SAVED_CHARS_FN = (
    "df_text_train_clean_saved_chars.parquet"  # FN as in "file name"
)
DF_TEXT_TRAIN_CLEAN_SAVED_CHARS_URL = (
    f"{GCP_PROJECT_URL}/{DF_TEXT_TRAIN_CLEAN_SAVED_CHARS_FN}"
)

DF_TEXT_TRAIN_CLEAN_INFO_FN = "data/df_text_train_clean_info.parquet"
DF_TEXT_TRAIN_CLEAN_INFO_GRAPH_FN = "data/df_text_train_clean_info_graph.parquet"
DF_TEXT_TRAIN_CLEAN_MISSING_FN = "data/df_text_train_clean_missing.parquet"
DF_TEXT_TRAIN_CLEAN_DUPLICATES_FN = "data/df_text_train_clean_duplicates.parquet"
DF_TEXT_TRAIN_CLEANED_FN = "data/df_text_train_cleaned.parquet"

DF_TEXT_TRAIN_DESCR_PRIM_CAT_FN = "data/df_text_train_descr_prim_cat.parquet"
DF_TEXT_TRAIN_MISS_DESCR_PRIM_CAT_FN = "data/df_text_train_miss_descr_prim_cat.parquet"
DF_TEXT_TRAIN_DUPL_DESCR_PRIM_CAT_FN = "data/df_text_train_dupl_descr_prim_cat.parquet"

DF_TEXT_TRAIN_DESCR_SUBCAT_FN = "data/df_text_train_descr_subcat.parquet"
DF_TEXT_TRAIN_MISS_DESCR_SUBCAT_FN = "data/df_text_train_miss_descr_subcat.parquet"
DF_TEXT_TRAIN_DUPL_DESCR_SUBCAT_FN = "data/df_text_train_dupl_descr_subcat.parquet"

DF_TEXT_TRAIN_DUPL_DES_TOP5_FN = "data/df_text_train_clean_dupl_des_top5.parquet"
DF_TEXT_TRAIN_DUPL_DESCR_TOP5_FN = "data/df_text_train_clean_dupl_descr_top5.parquet"

DF_IMAGE_TRAIN_FN = "data/df_image_train.parquet"

# Load DataFrames
with st.spinner(
    "Loading **DataFrame with Text Data**: Duration: approx. **6s**", show_time=True
):
    # Load primary DataFrame reduced to saved chars
    df_text_train_clean_saved_chars = load_DataFrame(
        DF_TEXT_TRAIN_CLEAN_SAVED_CHARS_URL
    )

    # Load secondary DataFrames for presentation
    df_text_train_clean_info = load_DataFrame(DF_TEXT_TRAIN_CLEAN_INFO_FN)
    df_text_train_clean_missing = load_DataFrame(DF_TEXT_TRAIN_CLEAN_MISSING_FN)
    df_text_train_duplicates = load_DataFrame(DF_TEXT_TRAIN_CLEAN_DUPLICATES_FN)
    df_text_train_cleaned = load_DataFrame(DF_TEXT_TRAIN_CLEANED_FN)

    df_text_train_clean_descr_prim_cat = load_DataFrame(DF_TEXT_TRAIN_DESCR_PRIM_CAT_FN)
    df_text_train_clean_descr_subcat = load_DataFrame(DF_TEXT_TRAIN_DESCR_SUBCAT_FN)

    df_text_train_clean_miss_descr_prim_cat = load_DataFrame(
        DF_TEXT_TRAIN_MISS_DESCR_PRIM_CAT_FN
    )
    df_text_train_clean_miss_descr_subcat = load_DataFrame(
        DF_TEXT_TRAIN_MISS_DESCR_SUBCAT_FN
    )

    df_text_train_clean_dupl_descr_prim_cat = load_DataFrame(
        DF_TEXT_TRAIN_DUPL_DESCR_PRIM_CAT_FN
    )
    df_text_train_clean_dupl_descr_subcat = load_DataFrame(
        DF_TEXT_TRAIN_DUPL_DESCR_SUBCAT_FN
    )

    df_text_clean_dupl_des_top5 = load_DataFrame(DF_TEXT_TRAIN_DUPL_DES_TOP5_FN)
    df_text_clean_dupl_descr_top5 = load_DataFrame(DF_TEXT_TRAIN_DUPL_DESCR_TOP5_FN)

    # Load additional DataFrame for visualization
    df_text_train_clean_info_graph = load_DataFrame(DF_TEXT_TRAIN_CLEAN_INFO_GRAPH_FN)


st.progress(4 / 7)
st.title("Data Overview")
st.sidebar.header(":material/search: Data Overview")
st.sidebar.image("images/logos/rakuten-logo-red-wide.svg", use_container_width=True)

st.markdown(
    """
Data Exploration and Data Visualization of training data for the **Rakuten E-Commerce Product Classification** project.
    """
)

with st.expander("**Options** for product category level", expanded=False):
    radio_category = st.radio(
        "Show **DataFrames and charts** grouped by...",
        (
            "Primary Category",
            "Subcategory",
        ),
        horizontal=True,
    )


# Distribute Data Overview content over self-explanatory tabs
tab_basic_info, tab_miss_values, tab_dupl_values = st.tabs(
    ["Basic information", "Missing values", "Duplicate values"]
)

with tab_basic_info:

    st.markdown(
        """
        The original dataset contains a total of **339,664 cells** distributed over **84,916 products with 4 columns**.
            """
    )

    # Display the DataFrame with basic information
    st.dataframe(df_text_train_clean_info, use_container_width=True)

    # Display a bar chart with the basic information
    with st.expander("Show **bar chart** with basic information", expanded=True):
        st.bar_chart(
            df_text_train_clean_info_graph,
            horizontal=True,
            use_container_width=True,
        )

    if radio_category == "Primary Category":
        with st.expander(
            "Show **DataFrame** with product count for primary category", expanded=False
        ):
            st.dataframe(df_text_train_clean_descr_prim_cat, use_container_width=True)

        with st.expander(
            "Show **bar chart** with product count for primary category", expanded=False
        ):
            st.bar_chart(
                df_text_train_clean_descr_prim_cat,
                horizontal=True,
                use_container_width=True,
            )

    elif radio_category == "Subcategory":
        with st.expander(
            "Show **DataFrame** with product count for subcategory", expanded=False
        ):
            st.dataframe(df_text_train_clean_descr_subcat, use_container_width=True)

        with st.expander(
            "Show **bar chart** with product count for subcategory", expanded=False
        ):
            st.bar_chart(
                df_text_train_clean_descr_subcat,
                horizontal=True,
                use_container_width=True,
            )


with tab_miss_values:
    st.markdown(
        """
                The training data has **29,800 missing values** in the column `description`.
                """
    )

    # Display the DataFrame with missing values
    st.dataframe(df_text_train_clean_missing, use_container_width=True)

    with st.expander("Show **bar chart** with missing values", expanded=True):
        df_text_train_clean_missing_graph = df_text_train_clean_missing.copy()
        df_text_train_clean_missing_graph = (
            df_text_train_clean_missing_graph.reset_index(drop=True).set_index("Column")
        )
        st.bar_chart(
            df_text_train_clean_missing_graph, horizontal=True, use_container_width=True
        )

    if radio_category == "Primary Category":
        with st.expander(
            "Show **DataFrame** with product count for primary category", expanded=False
        ):
            st.dataframe(
                df_text_train_clean_miss_descr_prim_cat, use_container_width=True
            )

        with st.expander(
            "Show **bar chart** with product count for primary category", expanded=False
        ):
            st.bar_chart(
                df_text_train_clean_miss_descr_prim_cat,
                horizontal=True,
                use_container_width=True,
            )

    elif radio_category == "Subcategory":
        with st.expander(
            "Show **DataFrame** with product count for subcategory", expanded=False
        ):
            st.dataframe(
                df_text_train_clean_miss_descr_subcat, use_container_width=True
            )

        with st.expander(
            "Show **bar chart** with product count for subcategory", expanded=False
        ):
            st.bar_chart(
                df_text_train_clean_miss_descr_subcat,
                horizontal=True,
                use_container_width=True,
            )

with tab_dupl_values:
    st.markdown(
        """
            The training data has **40,060 duplicate values** in the columns `designation` and `description`.
            """
    )

    # Display the DataFrame with duplicate values
    st.dataframe(df_text_train_duplicates, use_container_width=True)

    # Dropdown menu for duplicate values
    dupl_value_opt = st.selectbox(
        "Data exploration steps for duplicate values:",
        (
            "Duplicate values: Overview",
            "1. Duplicate values: Titles",
            "2. Duplicate values: Descriptions",
            "(3. Duplicate values: Dupl. title & miss. descr.)",
            "4. Cleaned data: Text length",
            "(5. Cleaned data: Duplicates)",
        ),
    )

    if dupl_value_opt == "Duplicate values: Overview":

        if radio_category == "Primary Category":
            with st.expander(
                "Show **DataFrame** with product count for primary category",
                expanded=False,
            ):
                st.dataframe(
                    df_text_train_clean_dupl_descr_prim_cat, use_container_width=True
                )

            with st.expander(
                "Show **bar chart** with product count for primary category",
                expanded=False,
            ):
                st.bar_chart(
                    df_text_train_clean_dupl_descr_prim_cat,
                    horizontal=True,
                    use_container_width=True,
                )

        elif radio_category == "Subcategory":
            with st.expander(
                "Show **DataFrame** with product count for subcategory", expanded=False
            ):
                st.dataframe(
                    df_text_train_clean_dupl_descr_subcat, use_container_width=True
                )

            with st.expander(
                "Show **bar chart** with product count for subcategory", expanded=False
            ):
                st.bar_chart(
                    df_text_train_clean_dupl_descr_subcat,
                    horizontal=True,
                    use_container_width=True,
                )

    elif dupl_value_opt == "1. Duplicate values: Titles":

        st.markdown(
            """
            The most frequent duplicate values for product titles in column `designation` are:
            """
        )

        # Display the DataFrame with duplicate titles
        st.dataframe(
            df_text_clean_dupl_des_top5,
            use_container_width=True,
        )

    elif dupl_value_opt == "2. Duplicate values: Descriptions":

        st.markdown(
            """
            The most frequent duplicate values for product descriptions in column `description` are:
            """
        )

        # Display the DataFrame with duplicate descriptions
        st.dataframe(
            df_text_clean_dupl_descr_top5,
            use_container_width=True,
        )

    elif dupl_value_opt == "(3. Duplicate values: Dupl. title & miss. descr.)":

        st.markdown(
            """
            :material/cruelty_free: Only **120 products** with a missing description have a duplicate title (designation).
            
            Compared to a total of **29,800 products** with a missing description, this is only **0.40%** of the total.
            """
        )

        # Display the code block with mask and methods
        st.code(
            """
        mask_description_nan = df_text_train["description"].isna()
            
        df_text_train_descr_nan = df_text_train[mask_description_nan]
            
        df_text_train_descr_nan_title_dupl = (
            df_text_train_descr_nan["designation"].duplicated().sum()
        )
        """
        )

        time.sleep(0.5)
        st.toast(
            "This is a rabbit hole, so be careful!", icon=":material/cruelty_free:"
        )

    elif dupl_value_opt == "4. Cleaned data: Text length":

        st.markdown(
            """
            We reduced text length by **6.42%** after removing errors, spaces, special chars., and HTML tags.
            """
        )

        # Display the DataFrame with cleaned data
        st.dataframe(df_text_train_cleaned, use_container_width=True)

        # Display expander with cleaned DataFrame sorted by saved text length
        with st.expander(
            "Show **cleaned text data** sorted by saved chars", expanded=False
        ):

            st.dataframe(
                df_text_train_clean_saved_chars,
                use_container_width=True,
            )

    elif dupl_value_opt == "(5. Cleaned data: Duplicates)":

        st.markdown(
            """
            :material/cruelty_free: Only **211 products** additional duplicate values can be found after text cleaning.
            """
        )

        # Display the code block with mask and methods
        st.code(
            """
        text_cols = ['designation_cleaned', 'description_cleaned']

        for cn_clean in text_cols:
            cn_duplicates = df_text_train_clean[cn_clean].duplicated().sum()
            pct = cn_duplicates / rows

            # Get the previous value from the dictionary
            cn = re.sub('_cleaned', '', cn_clean)
            pct_incr = pct - duplicate_values[cn][1]

            print(f'The cleaned column \033[1m\'{cn_clean}\'\033[0m has...')
            print(f'\t{cn_duplicates:,} duplicates compared to {duplicate_values[cn][0]:,} (+{cn_duplicates - duplicate_values[cn][0]:,})')
            print(f'\tAn increase of {pct_incr:.2%} from {duplicate_values[cn][1]:.2%} to {pct:.2%}', end='\n\n')
            """
        )

        time.sleep(0.5)
        st.toast(
            "This is another rabbit hole, you really should be more careful...",
            icon=":material/cruelty_free:",
        )

# Pagination and footer
st.markdown("---")
add_pagination_and_footer("pages/4_Data_Overview.py")
