# feb25_bds_classification-of-rakuten-e-commerce-products

## Meeting notes: March 25th
1. **DataFrame:** Emphasis on using _[langdetect](https://pypi.org/project/langdetect/)_ to get probability of language
2. **DataFrame:** Merge _designation_ and _description_ in additional text field
3. **Order of steps:**
   1. Cleaning
   2. Translation
   3. Normalization
4. **Images:**
   1. Delete blank space
   2. Normalize ratio of images
   3. Goal for next week: Set bounding box
5. **Approach:**
  1. Choose one language, e.g. English
  2. Translate other languages via _DeepL_'s API
  3. TF-IDF: Define importance of word (or sentence)
  4. Apply text mining techniques only after, e.g. lemmatization
  5. Define threshold for designation and description via mean, median, etc. to exclude text, which is too long (@RW, @TB: Did I miss anything important?)
  6. Goal is to Word2Vec (CBOW, skip-gram), BERT and Meta's alternative (?)

## Additional Notes from Yaniv:
1. Continue your work on the preprocessing tasks on the images and the text as :
      1. Delete HTML tags and special caracters on the text. Pay attention to some outliers.
      2. Merge description and designation columns to remove missing values
      3. For the translation, we can try to do everything in english thanks to the DeepL API.
      4. For the images, we can work on the ratio of the images as the size of the bouding box around the image. You should find something arount 20% of images below a ratio of 0.8.
2. These images will be zoomed if it's possible.
   1. I liked the data viz you have presented but I can propose to do some others about :
   2. Class distribution
   3. images by category, notice that they are oddly made
   4. Occurrence of words by category
   5. Word cloud (module 131 Text Mining) on each category or on the whole column designation+description

## Meeting notes: March 18th

**A. To-Dos // Preprocessing**
1. **DataFrame:** Detect languages e.g. FRA, ENG, GER, etc. in designation and description with _[langdetect](https://pypi.org/project/langdetect/)_
2. **DataFrame:** Are there mixed languages in designation and description, e.g. FRA & ENG, FRA & GER, etc.
3. **DataFrame:** Merge product type with product type names if possible with [Liste des catégories (categorymap)](https://global.fr.shopping.rakuten.com/developpeur/liste-des-categories-categorymap/)
4. **DataFrame:** Are there duplicate items, e.g. product id, designation, description?
5. **DataFrame:** Add TF-IDF (Term Frequency-Inverse Document Frequency) as proxy for text quality.
6. **DataFrame:** Are there short _(meaningless)_ text (designations or descriptions)?
7. **DataFrame:** Are there long texts due to keyword spamming meaning duplicated, repetitive text?
8. **DataFrame:** Are there formatting errors and html in the text, e.g. '&uuml;' or à?

**B. Backlog**
- **Images:** Sizes, resolution, ratios, blank images (borders).
- **Plot:** Visualize NaNs per product type.
- **Research:** Reduce vector size of text.

**C. Verschiedenes**
- [Normalform von Datenbanken](https://www.tinohempel.de/info/info/datenbank/normalisierung.htm)


## Meeting notes: March 18th

**A. WordCloud**
1. Stopwords filtern
2. Mehrsprachigkeit ggf. beachten

**B. Visualizations:**
1. Fünfte Grafik mit Matrix aus numerischen Werten aufgrund fehlender Daten nicht möglich

**C. Weitere mgl. Fragen & Grafiken:**
1. Verteilung von NaNs in Beschreibung pro Product Type
2. Gibt es fehlende Bilder?
3. Verteilung von Auflösung?
4. Wahrscheinlichkeit, dass fehlende Textinhalten (s. o.) zu fehlenden Bildern bzw. niedrig-auflösenden Grafiken führen?

**D. Verschiedenes**
1. [GitHub Projects](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects): Projektmanagement à la Kanban mit GANTT-Charts
2. [GitHub Codespace](https://github.com/features/codespaces): Programmierumgebung
3. [Formattierungsmöglichkeiten in Markdown](http://markdownguide.org/)

**E. Tasks**
1. Thomas: Wordclouds
2. Robert: Distribution of Product Type Codes (Count plot), Distribution of Title Lengths
3. Peter: Sample images from different product types, Missing images, Image resolution
