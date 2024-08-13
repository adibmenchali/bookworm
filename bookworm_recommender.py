import pandas as pd
import requests
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from sentence_transformers import SentenceTransformer
import torch

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load pre-trained sentence transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def load_book_data(csv_file):
    return pd.read_csv(csv_file)

def fetch_book_info(title):
    url = f'https://www.googleapis.com/books/v1/volumes?q=intitle:{title}'
    response = requests.get(url)
    data = response.json()
    if 'items' in data and len(data['items']) > 0:
        return data['items'][0]['volumeInfo']
    else:
        return None

def preprocess_text(text):
    if isinstance(text, str):  # Check if the text is a string
        # Tokenization
        tokens = word_tokenize(text)

        # Lowercasing
        tokens = [token.lower() for token in tokens]

        # Removing punctuation
        tokens = [token for token in tokens if token not in string.punctuation]

        # Removing stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

        # Join tokens back into text
        preprocessed_text = ' '.join(tokens)

        return preprocessed_text
    else:
        return ''  # Return an empty string if the input is not a string or is NaN

def get_similar_books(book_df, title):
    fetched_book_info = fetch_book_info(title)

    if fetched_book_info:
        fetched_title = fetched_book_info.get('title', '')
        fetched_authors = fetched_book_info.get('authors', [])
        fetched_description = fetched_book_info.get('description', '')

        fetched_description = preprocess_text(fetched_description)
        fetched_authors_str = ', '.join(fetched_authors)
        fetched_text = f"{fetched_title} {fetched_authors_str} {fetched_description}"

        book_df['preprocessed_description'] = book_df['description'].apply(preprocess_text)

        fetched_embedding = model.encode([fetched_text])

        batch_size = 1
        similarity_scores = []

        for i in range(0, len(book_df), batch_size):
            batch_descriptions = book_df['preprocessed_description'][i:i+batch_size]
            batch_embeddings = model.encode(batch_descriptions.tolist())

            similarity_scores.extend(torch.cosine_similarity(torch.tensor(fetched_embedding), torch.tensor(batch_embeddings)).tolist())

        similar_books = []

        top_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)[:10]

        for idx in top_indices:
            similar_books.append({
                'title': book_df.iloc[idx]['title'],
                'author': book_df.iloc[idx]['authors'],
                'description': book_df.iloc[idx]['description'],
                'similarity_score': similarity_scores[idx]
            })
        print("Top 10 Most Similar Books:")
        for book in similar_books:
            print(f"Title: {book['title']}")
            print(f"Author: {book['author']}")
            print(f"Description: {book['description']}")
            print(f"Similarity Score: {book['similarity_score']}")
            print()
        return similar_books

if __name__ == "__main__":
    # Load book data
    book_df = load_book_data('book_data.csv')

    # Get similar books
    similar_books = get_similar_books(book_df, 'a thousand splendid suns')

    # Print similar books
    print("Top 10 Most Similar Books:")
    for book in similar_books:
        print(f"Title: {book['title']}")
        print(f"Author: {book['author']}")
        print(f"Description: {book['description']}")
        print(f"Similarity Score: {book['similarity_score']}")
        print()