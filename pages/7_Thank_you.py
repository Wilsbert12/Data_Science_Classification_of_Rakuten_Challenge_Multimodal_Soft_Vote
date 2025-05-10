# Thank you
import streamlit as st
from streamlit_utils import add_pagination

st.set_page_config(
    page_title="FEB25 BDS // Thank You",
    page_icon="images/logos/rakuten-favicon.ico",
    layout="wide",
)

st.progress(7 / 7)
st.title("Thank You")
st.sidebar.header(":material/folded_hands: Thank You")
st.sidebar.image("images/logos/rakuten-logo-red-wide.svg", use_container_width=True)

# Main content
st.markdown(
    """
## Thank You for Exploring Our Project!

We appreciate you taking the time to review our work on the classification of Rakuten e-commerce products. 
This project represents significant effort and collaboration from our entire team.

### Key Contributions

Our team has worked diligently to:
* Collect and process complex e-commerce data
* Develop robust classification models
* Create this interactive Streamlit application to showcase our work

### Feedback and Contact

Your feedback is valuable to us! If you have any questions, suggestions, or would like to discuss this project further,
please don't hesitate to reach out to our team members.

### Acknowledgements

We would like to extend our sincere gratitude to:
* Our instructors and mentors who guided us throughout this project
* Rakuten for providing the dataset that made this research possible
* The open-source community for the wonderful tools and libraries that powered our analysis
"""
)

# Add a decorative element - a simple progress completion bar
st.subheader("Project Exploration Progress")
st.progress(100)
st.success("You've completed the entire project tour! Thank you for your interest!")


# Final touch - a simple animation
st.balloons()


# Pagination and footer
st.markdown("---")
add_pagination("pages/7_Thank_you.py")
st.markdown("© 2025 | Peter Stieg, Robert Wilson, Thomas Borer")
