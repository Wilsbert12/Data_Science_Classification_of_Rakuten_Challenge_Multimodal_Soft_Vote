import streamlit as st

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

    # Create three columns for previous, current page indicator, next
    prev_butt, elc, pagination, erc, next_butt = st.columns(
        [2, 2, 1, 2, 2]
    )  # elf and erc as in "empty left column" and "empty right column"

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
