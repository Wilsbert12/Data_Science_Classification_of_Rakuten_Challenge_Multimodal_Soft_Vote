# Team Presentation
import streamlit as st
from streamlit_utils import add_pagination

st.set_page_config(
    page_title="FEB25 BDS // Team Presentation",
    page_icon="images/logos/rakuten-favicon.ico",
    layout="wide",
)

st.progress(2 / 8)
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
    st.write("• Data preprocessing for text and images")
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
        - Python, PHP, JavaScript
        - Project Management 
        - User Experience
        - Data Science
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
    st.write("• ?")
    st.write("• ?")
    st.write("• ?")

    with st.expander("… more info"):
        st.markdown(
            """
        **Robert Wilson**
        ---
        
        **Former positions:**
        - Senior Account Executive
        - Sales Manager
        - Director of Sales and Marketing
        
        **Skills & Expertise:**
        - ?
        - ?
        - ?
        - ?
        - ?
        
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
    st.write("• Text preprocessing: Cleaning and localization")
    st.write("• Model Development and Evaluation")
    st.write("• BERT")
    st.write("• ?")
    st.write("• ?")

    with st.expander("… more info"):
        st.markdown(
            """
        **Thomas Borer**
        ---
        
        **Former positions:**
        - Principal Speech Scientist
        - Senior Speech Scientist
        - ?
        
        
        **Skills & Expertise:**
        - Natural Language Processing
        - ?
        - ?
        - ?
        - ?
    
        **Links:**
        - [GitHub Profile](https://github.com/thomas-borer)
        - [LinkedIn Profile](https://www.linkedin.com/in/thomas-borer-066714111/)
        """
        )


# Pagination and footer
st.markdown("---")
add_pagination("pages/2_Team_Presentation.py")
st.markdown("© 2025 | Peter Stieg, Robert Wilson, Thomas Borer")
