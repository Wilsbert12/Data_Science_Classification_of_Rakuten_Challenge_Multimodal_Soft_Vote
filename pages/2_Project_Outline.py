# Project Outline
import streamlit as st
import pandas as pd
from streamlit_utils import add_pagination_and_footer

st.set_page_config(
    page_title="FEB25 BDS // Project Outline",
    page_icon="images/logos/rakuten-favicon.ico",
    layout="wide",
)

st.progress(2 / 7)
st.title("Project Outline")
st.sidebar.header(":material/work: Project Outline")
st.sidebar.image("images/logos/rakuten-logo-red-wide.svg", use_container_width=True)

# Project Outline content
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

        >> **1) Text cleaning**: removing html code/remnants and redundant chars/ words, solving encoding issues
        
        >> **2) Language detection**: Using the langdetect library as well as Gemini API
        
        >> **2) Localization**: Using the DeepL API
      
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
          - Gradient Boosting
          - Logistic Regression
          - Random Forest
          - Support Vector Machine
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

    # Performance metrics
    models_results_data = {
        "Type": [
            "Text classifier",
            "Text classifier",
            "Text classifier",
            "Text classifier",
            "Text classifier",
            "Image classifier",
        ],
        "Classifier": [
            "SVM",
            "LogisticRegression",
            "XGBClassifier",
            "RandomForestClassifier",
            "CamemBERT",
            "VGG16",
        ],
        "Accuracy": [
            0.77,
            0.74,
            0.65,
            0.55,
            '-',
            0.58,
        ],
        "Precision (weighted)": [
            0.78,
            0.75,
            0.72,
            0.73,
            0.76,
            0.61,
        ],
        "Recall (weighted)": [
            0.76,
            0.74,
            0.65,
            0.55,
            0.75,
            0.58,
        ],
        "F1 (weighted)": [
            0.76,
            0.74,
            0.66,
            0.63,
            0.75,
            0.58,
        ],
    }

    # Best model information
    best_model_info_data = {
        "Model": ["SVC"],
        "random_state": [42],
        "classifier__C": [1],
        "classifier__kernel": ["linear"],
        "vectorizer": ["TfidfVectorizer"],
        "max_features": [5000],
        "ngram_range": ["(1, 1)"],
        "Cross-Validation Score": [0.75],
    }

    models_results_df = pd.DataFrame(models_results_data).set_index("Classifier")
    best_model_info_df = pd.DataFrame(best_model_info_data).set_index("Model")

    st.dataframe(models_results_df, use_container_width=True)

    st.markdown("**SVC // Best performing hyperparameters**")
    st.dataframe(best_model_info_df, use_container_width=True)

    st.markdown(
        """
        _Key Findings_
        - Underrepresented categories achieved high classification performance despite class imbalance
        - Model performance showed high sensitivity to hyperparameter selection, with significant risk of overfitting
        - Computational complexity presented scalability challenges, particularly for ensemble methods
        - Text vector representations exhibited high dimensionality but demonstrated linear separability, eliminating the need for kernel transformations
        """
    )

    st.markdown("**CamemBERT**")

    st.markdown(
        """
        _Key Findings_
        * Easy to use library from HuggingFace
        * Training is somewhat resource intensive - limited number of parameters tried
            * Colab L4 GPU - ~3h
            * tokenizer max length - 256
            * class weight - balanced
            * training epochs - 10
            * training and validation loss starting to move opposite directions after 10 epochs
        * potential next step - Use LLM to generate more data
        """
    )


# Pagination and footer
st.markdown("---")
add_pagination_and_footer("pages/2_Project_Outline.py")
