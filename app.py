from flask import Flask, jsonify, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import os

# Initialize Flask app
app = Flask(__name__, root_path=os.path.abspath(os.getcwd()))

# Load dataset
data = pd.read_csv('db.csv')

# Convert columns to lowercase
data['Title'] = data['Title'].str.lower()
data['Genre'] = data['Genre'].str.lower()
data['Director'] = data['Director'].str.lower()

# Split comma-separated genres into lists
data['Genre'] = data['Genre'].apply(lambda x: x.split(',') if isinstance(x, str) else [])

# Split and lowercase cast names
data['Cast'] = data['Cast'].apply(lambda x: [name.lower() for name in x.split(',')] if isinstance(x, str) else [])

# Initialize TF-IDF Vectorizers for relevant columns
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

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')  # Make sure index.html exists in the templates folder

@app.route('/recommend')
def recommend():
    movie = recommend_random_movie()  # Get a random movie recommendation
    return jsonify(movie.to_dict())  # Convert the movie data to dictionary and return as JSON

@app.route('/search', methods=['GET'])
def search():
    user_query = request.args.get('query', '')  # Extract query from the URL
    if user_query:
        results = search_movies(user_query)  # Call the search function
        return jsonify(results.to_dict(orient='records'))  # Return results as a list of dictionaries
    else:
        return jsonify({"error": "No query provided"}), 400

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
