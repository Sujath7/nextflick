import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# Load dataset
data = pd.read_csv('db.csv')
# Print the first few rows of the dataset to confirm it's loaded correctly
print(data.head())

# Group by Year and count the number of Titles
titles_by_year = data.groupby('Year')['Title'].count()
# Display the summary
print(titles_by_year)

# Plot the distribution
titles_by_year.plot(kind='bar', figsize=(10, 5), color='skyblue')
plt.title('Number of Titles by Year')
plt.xlabel('Year')
plt.ylabel('Number of Titles')
plt.xticks(rotation=45)
plt.show()

# Check for null values in each column
null_summary = data.isnull().sum()
# Display null values
print(null_summary)

# Visualize null values (if any)
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Null Value Heatmap')
plt.show()

# Check whether data types are correct
print(data.dtypes)

# Summary statistics for numerical columns
print(data.describe())

# Plot the distribution of the top 10 genres
plt.figure(figsize=(10, 6))
top_10_genres = data['Genre'].value_counts().head(10)
top_10_genres.plot(kind='bar', color='coral')
plt.title('Top 10 Movies by Genre')
plt.xlabel('Genre')
plt.ylabel('Number of Movies')
plt.xticks(rotation=45)
plt.show()

# Convert titles and genres to lowercase
data['Title'] = data['Title'].str.lower()
data['Genre'] = data['Genre'].str.lower()

# Split comma-separated genres into lists
data['Genre'] = data['Genre'].apply(lambda x: x.split(',') if isinstance(x, str) else [])

# Normalize director names (convert to lowercase)
data['Director'] = data['Director'].str.lower()

# Split and lowercase cast names
data['Cast'] = data['Cast'].apply(lambda x: [name.lower() for name in x.split(',')] if isinstance(x, str) else [])

# Initialize NLTK components
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Tokenization and lemmatization for descriptions
def preprocess_description(desc):
    tokens = word_tokenize(desc.lower())
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

# Apply to descriptions
data['Description'] = data['Description'].apply(preprocess_description)

# Initialize TF-IDF Vectorizers
title_vectorizer = TfidfVectorizer()
description_vectorizer = TfidfVectorizer()
director_vectorizer = TfidfVectorizer()
cast_vectorizer = TfidfVectorizer()

# Fill NaN and lowercase text columns
text_columns = ['Title', 'Description', 'Director', 'Cast']
for col in text_columns:
    data[col] = data[col].astype(str).fillna('').str.lower()

# Fit-transform for each column
title_vectors = title_vectorizer.fit_transform(data['Title'])
description_vectors = description_vectorizer.fit_transform(data['Description'])
director_vectors = director_vectorizer.fit_transform(data['Director'])
cast_vectors = cast_vectorizer.fit_transform(data['Cast'])

# Function to detect column type based on user query
def detect_column(query):
    query = query.lower()
    if "starring" in query or "cast" in query:
        return 'Cast', cast_vectorizer, cast_vectors
    elif "directed by" in query or "director" in query:
        return 'Director', director_vectorizer, director_vectors
    elif "about" in query or "description" in query:
        return 'Description', description_vectorizer, description_vectors
    else:
        return 'Title', title_vectorizer, title_vectors

# Search for movies based on the user query
def search_movies(user_query):
    column, vectorizer, vectors = detect_column(user_query)
    query_vector = vectorizer.transform([user_query])
    similarity_scores = cosine_similarity(query_vector, vectors).flatten()
    
    keywords = user_query.lower().split()  
    filtered_movies = data[
        data['Title'].str.contains('|'.join(keywords), case=False) |
        data['Description'].str.contains('|'.join(keywords), case=False) |
        data['Cast'].apply(lambda cast: any(keyword in str(cast).lower() for keyword in keywords)) |
        data['Genre'].apply(lambda genres: any(keyword in str(genres).lower() for keyword in keywords)) |
        data['Director'].apply(lambda director: any(keyword in str(director).lower() for keyword in keywords))
    ]
    
    if filtered_movies.empty:
        print("No matches found. Using the full dataset instead.")
        filtered_movies = data
    
    relevant_vectors = vectors[filtered_movies.index]
    similarity_scores = cosine_similarity(query_vector, relevant_vectors).flatten()
    
    filtered_movies['Rating_Norm'] = filtered_movies['Rating'] / 10
    filtered_movies['Votes_Norm'] = filtered_movies['Votes'] / filtered_movies['Votes'].max()

    weighted_scores = similarity_scores * (
        0.6 * filtered_movies['Rating_Norm'] + 0.4 * filtered_movies['Votes_Norm']
    )

    top_indices = weighted_scores.argsort()[-5:][::-1]
    return filtered_movies.iloc[top_indices]

# Function for random movie recommendation
def recommend_random_movie():
    # Filter movies with rating > 7
    filtered_data = data[data['Rating'] > 7]
    # Randomly select a movie from the filtered list
    movie = filtered_data.sample(n=1).iloc[0]
    return movie

# Test "Recommend me a movie" button - Random recommendation
print("Your Next Flick:")
recommended_movie = recommend_random_movie()
print(f"Title: {recommended_movie['Title']}")
print(f"Genre: {recommended_movie['Genre']}")
print(f"Director: {recommended_movie['Director']}")
print(f"Cast: {recommended_movie['Cast']}")
print(f"Rating: {recommended_movie['Rating']} | Votes: {recommended_movie['Votes']}")
print(f"Description: {data.loc[recommended_movie.name, 'Description']}\n")


# Example user query for searching movies - Replace this with any query you want to test
user_query = "History adventure"  # Replace with your test query
print(f"User Query: {user_query}\n")

# Get search results based on the user query
results = search_movies(user_query)

# Display results
for index, row in results.iterrows():
    print(f"Title: {row['Title']} ({row['Year']})")
    print(f"Genre: {row['Genre']}")
    print(f"Director: {row['Director']}")
    print(f"Cast: {row['Cast']}")
    print(f"Rating: {row['Rating']} | Votes: {row['Votes']}")
    print(f"Description: {row['Description']}\n")
