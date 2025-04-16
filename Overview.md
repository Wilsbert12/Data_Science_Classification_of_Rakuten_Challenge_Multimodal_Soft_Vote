# Overview

## File structure overview

- .ipynb_checkpoints
  - Rakuten_eCommerce_Products
- MISC
  - Training_Python_Plotly
- language_analysis
  - deepL_output_backup.csv
  - deepL_result.csv
  - df_lang.csv
  - df_langdetect.csv
  - df_localization.csv
  - gemini_result.json
  - merged_output.csv
- .gitignore
- Business_Impact.md
- DeepL.ipynb
- Picture Experiments.ipynb
- README.md
- Rakuten_eCommerce_Products
- X_test.csv
- X_train.csv
- data_viz.py
- df_image.csv
- frequent_word.json
- gemini.ipynb
- image_utils.py
- language_analysis.ipynb
- text_utils.py
- text_utils_test.py
- y_train.csv

## File content

- .ipynb_checkpoints
  - Rakuten_eCommerce_Products-checkpoint.ipynb\
    > Main file including links to issues (Github Kanban board)
- MISC
  - Training_Python_Plotly\
    > A test file including a generic 3d plot (from @Peter?)
- language_analysis
  - deepL_output_backup.csv
    > A csv backup file containing all deepL translations
  - deepL_result.csv
    > The final deepL translations
  - df_lang.csv
    > A csv containing the detected language of each description
  - df_langdetect.csv
    > ???
  - df_localization.csv
    > ???
  - gemini_result.json
    > ???
  - merged_output.csv
    > ???
- .gitignore
  > ignores specific files when executing Git
- Business_Impact.md
  > Detailed description of business impact of project
- DeepL.ipynb
  > Importing cleaned text and translating it via DeepL
- Picture Experiments.ipynb
  > Experimentation of image represenattion for Data Viz:
  > 1. Grid of random product pictures
  > 2. Visualisation of dimensionality reduction on product images to a 2d scatterplot (Isomap Algorithm)
- README.md
  > Md file containing weekly project meeting notes and tasks
- **Rakuten_eCommerce_Products**
  > **The main file containing most of the code. Most other files are executed and loaded into this file**
- X_test.csv
  > Test feature data 
- X_train.csv
  > Train feature data
- data_viz.py
  > File to remove data Viz execution code from main file. Does only contain wordcloud code so far.
- df_image.csv
  > CSV file containing image ids and respective meta data for each image (aspect ration etc.)
- frequent_word.json
  > saves the most frequent word per category for the wordcloud. Is created in data_viz.py file.
- gemini.ipynb
  > gemini translation???
- image_utils.py
  > extracts metadata from images. Planned: Immage Processing, classification
- language_analysis.ipynb
  > In depth analysis of languages detected
- text_utils.py
  > text cleaning and processing
- text_utils_test.py
  > Testing text_utils.py
- y_train.csv
  > Training target data
