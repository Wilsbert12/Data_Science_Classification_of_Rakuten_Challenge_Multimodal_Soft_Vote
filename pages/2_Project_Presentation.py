# Project presentation
import streamlit as st
import pandas as pd
from streamlit_utils import add_pagination

st.set_page_config(
    page_title="FEB25 BDS // Project Presentation",
    page_icon="images/logos/rakuten-favicon.ico",
    layout="wide",
)

st.progress(2 / 7)
st.title("Project Presentation")
st.sidebar.header(":material/work: Project Presentation")
st.sidebar.image("images/logos/rakuten-logo-red-wide.svg", use_container_width=True)

# Project presentation content
project_tab1, project_tab2, project_tab3, project_tab4 = st.tabs(
    ["Description", "Methodology", "Models", "Results"]
)

with project_tab1:
    st.markdown(
        """
        
        The project leverages **machine learning** to:
        * process product **titles and descriptions** for textual understanding of items
        * analyze associated product **images** to extract visual features for classification
        * combine text and image data in **multimodal input** to create more accurate product categorization

        **Objective:** The goal is to create machine learning models that classify a product listings' category accurately.
        
        **:material/folder_code: GitHub Repository:** [feb25_bds_classification-of-rakuten-e-commerce-products](https://github.com/PeterStieg/feb25_bds_classification-of-rakuten-e-commerce-products)
        """
    )

with project_tab2:
    st.markdown(
        """
        :material/search: **Data Exploration**
        >> Analysis of dataset with product titles, descriptions, and images.
        
        :material/query_stats: **Data Visualization**
        >> Distribution of product categories, text length, text content, errors, etc.
        
        :material/rule: **Data Preprocessing**
        >> Text cleaning by e.g. removing redundant chars/ words and finding bounding boxes.
        
      
        :material/model_training: **Modelling**
        >> **1) Feature Extraction**: Extraction of relevant features from the text and image data with e.g. TF-IDF and CNN.
        
        >> **2) Evaluation**: Assessment of model performance using cross-validation and hyperparameter tuning as well as F1-score.
        
        :material/category_search: **Prediction**
        >> Testing models on unseen test data as well as custom user input.
        """
    )

with project_tab3:
    models_col1, models_col2 = st.columns(2)

    with models_col1:
        st.markdown("### Text Models")
        st.markdown(
            """
        - **TF-IDF with Classical ML:**
          - Random Forest
          - SVM
          - Gradient Boosting
        - **BERT-based models:**
          - Fine-tuned camemBERT
        """
        )

    with models_col2:
        st.markdown("### Image Models")
        st.markdown(
            """
        - **Deep Learning:**
          - VGG16 (Transfer learning)
        """
        )

with project_tab4:

    # Example results visualization
    results_data = {
        "Model": ["TF-IDF + RF", "TF-IDF + SVM", "BERT", "VGG16", "Combined"],
        "Accuracy": ["?", "?", "?", "?", "?"],
        "Precision": ["?", "?", "?", "?", "?"],
        "Recall": ["?", "?", "?", "?", "?"],
        "F1-Score": ["?", "?", "?", "?", "?"],
    }

    results_df = pd.DataFrame(results_data)

    # Add a color column for the bar chart
    st.dataframe(results_df, use_container_width=True)

    st.markdown(
        """
        **Key Findings:**
        - BERT-based models performed best for text classification
        - VGG16 showed strong performance for image classification
        - The combined multi-modal approach achieved the highest overall accuracy
        - Certain product categories benefited more from text data, while others from image data
        """
    )


# Pagination and footer
st.markdown("---")
add_pagination("pages/2_Project_Presentation.py")
st.markdown("© 2025 | Peter Stieg, Robert Wilson, Thomas Borer")
