import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def word_cloud(df, target_column, text_column):
    '''requies to specify the data frame as well as the target column. The function is used to generate one wordcloud per target category.'''

    nltk.download("stopwords")

    # Get the list of French stopwords
    french_stopwords = set(stopwords.words("french"))
    frequent_words = {}
    categories = df[target_column].unique()
    categories = [int(x) for x in categories]
    for category in categories:

        # Create a text string from the description column for the given category
        text = " ".join(
            word for word in " ".join(
                df[df[target_column] == category][text_column].dropna()
            ).split()
            if word.lower() not in french_stopwords
        )
        # removing potential html markup from text
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()

        # Generate the word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        frequent_words[category] = wordcloud.words_
        # Plot the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')  # Remove axis
        plt.title(f'Word cloud for product type code {category}')
        plt.show()
