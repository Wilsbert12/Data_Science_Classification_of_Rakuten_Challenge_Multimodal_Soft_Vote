# Team Presentation
import streamlit as st
from streamlit_utils import add_pagination_and_footer

st.set_page_config(
    page_title="FEB25 BDS // Team Presentation",
    page_icon="images/logos/rakuten-favicon.ico",
    layout="wide",
)

st.progress(1 / 7)
st.title("Team Presentation")
st.sidebar.header(":material/diversity_3: Team Presentation")
st.sidebar.image("images/logos/rakuten-logo-red-wide.svg", use_container_width=True)


# Create three columns for team members
tp_col1, tp_col2, tp_col3 = st.columns(3)  # tp_ as in "team presentation"

with tp_col1:
    st.image("images/profile_pictures/peter_stieg.jpg", use_container_width=True)
    st.info("Peter Stieg")

    # Display primary contribution information directly
    st.write("**Primary Contributions:**")
    st.write("• Preprocessing of text and images")
    st.write("• Image classification with VGG16")
    st.write("• Streamlit presentation")
    st.write("• GitHub Management")
    st.write("• Cloud Deployment")

    with st.expander("… more info"):
        st.markdown(
            """
        **Peter Stieg**
        ---
        
        **Former positions:**
        - Marketing Director
        - Head of Marketing
        - COO
        
        **Skills & Expertise:**
        - Project Management 
        - User Experience
        - Marketing
        
        **Links:**
        - [GitHub Profile](https://github.com/peterstieg/)
        - [LinkedIn Profile](https://www.linkedin.com/in/PeterStieg/)
        """
        )

with tp_col2:
    st.image("images/profile_pictures/robert_wilson.jpg", use_container_width=True)
    st.info("Robert Wilson")

    # Display primary contribution information directly
    st.write("**Primary Contributions:**")
    st.write("• Model Development and Evaluation")
    st.write("• Classical Machine Learning")
    st.write("• Training data enrichment")
    st.write("• Target data exploration")
    st.write("• Data Visualization")

    with st.expander("… more info"):
        st.markdown(
            """
        **Robert Wilson**
        ---
        
        **Former positions:**
        - Director of Sales and Marketing
        - Senior Account Executive
        - Sales Manager
        
        **Skills & Expertise:**
        - SaaS B2B Enterprise Sales
        - Performance Marketing
        - Pre-Sales
        
        **Links:**
        - [GitHub Profile](https://github.com/Wilsbert12)
        - [LinkedIn Profile](https://www.linkedin.com/in/robert-wilson-17081983/)
        """
        )

with tp_col3:
    st.image("images/profile_pictures/thomas_borer.jpg", use_container_width=True)
    st.info("Thomas Borer")

    # Display primary contribution information directly
    st.write("**Primary Contributions:**")
    st.write("• Text cleaning and localization")
    st.write("• Model Development and Evaluation")
    st.write("• Text classification with BERT")
    st.write("• LLM workflow integration")
    st.write("• Voting Classifier")

    with st.expander("… more info"):
        st.markdown(
            """
        **Thomas Borer**
        ---
        
        **Former positions:**
        - Principal Speech Scientist
        - Senior Speech Scientist
        - Teaching Assistant
        
        
        **Skills & Expertise:**
        - Natural Language Processing
        - Regression Testing
        - CI/CD Pipelines
        
    
        **Links:**
        - [GitHub Profile](https://github.com/thomas-borer)
        - [LinkedIn Profile](https://www.linkedin.com/in/thomas-borer-066714111/)
        """
        )


# Pagination and footer
st.markdown("---")
add_pagination_and_footer("pages/1_Team_Presentation.py")
