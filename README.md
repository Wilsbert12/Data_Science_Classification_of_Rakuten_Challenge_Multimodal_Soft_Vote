# feb25_bds_classification-of-rakuten-e-commerce-products

## Meeting notes: March 25th
1. **DataFrame:** Emphasis on using _[langdetect](https://pypi.org/project/langdetect/)_ to get probability of language
2. **DataFrame:** Merge _designation_ and _description_ in additional text field
3. **Images:**
   1. Delete blank space
   2. Normalize ratio of images
   3. Goal for next week: Set bounding box
4. **Approach:**
  1. Choose one language
  2. Translate other languages via _DeepL_'s API
  3. TF-IDF: Define importance of word (or sentence)
  4. Apply text mining techniques only after, e.g. lemmatization
  5. Define threshold for designation and description via mean, median, etc. to exclude text, which is too long (@RW, @TB: Did I miss anything important?)
  6. Goal is to Word2Vec (CBOW, skip-gram), BERT and Meta's alternative (?)

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
