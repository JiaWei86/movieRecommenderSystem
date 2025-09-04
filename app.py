# app.py
import streamlit as st
import pandas as pd
from surprise import SVD
import joblib
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

# ========================
# Load Data
# ========================
@st.cache_data
def load_data():
    ratings = pd.read_csv('ml-100k/u.data', sep='\t', header=None, names=['userId','movieId','rating','timestamp'])
    movies = pd.read_csv('ml-100k/u.item', sep='|', header=None, usecols=[0,1], names=['movieId','title'], encoding='latin-1')
    return ratings, movies

ratings, movies = load_data()

# ========================
# Load SVD Model
# ========================
model = joblib.load('model.joblib')

# ========================
# Compute Movie Similarity (using ratings)
# ========================
movie_ratings = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
similarity_matrix = cosine_similarity(movie_ratings.T)
similarity_df = pd.DataFrame(similarity_matrix, index=movie_ratings.columns, columns=movie_ratings.columns)

# ========================
# Helper Functions
# ========================
def get_top_n_recommendations(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, _, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

def recommend_movies(user_id, n=10):
    all_movie_ids = ratings['movieId'].unique()
    user_movies = ratings[ratings['userId']==user_id]['movieId'].values
    testset = [(user_id, mid, 0) for mid in all_movie_ids if mid not in user_movies]
    predictions = model.test(testset)
    top_n = get_top_n_recommendations(predictions, n)
    top_movies = [(movies[movies['movieId']==mid]['title'].values[0], est) for mid, est in top_n[user_id]]
    return top_movies

def recommend_similar_movies(movie_name, top_n=5):
    movie_row = movies[movies['title'].str.contains(movie_name, case=False)]
    if movie_row.empty:
        return f"No movies found matching '{movie_name}'"
    movie_id = movie_row.iloc[0]['movieId']
    similar_scores = similarity_df[movie_id].sort_values(ascending=False).drop(movie_id)
    top_movies = [(movies[movies['movieId']==mid]['title'].values[0], score) for mid, score in similar_scores.head(top_n).items()]
    return top_movies

# ========================
# Streamlit UI
# ========================
st.title("Movie Recommendation System 🎬")

# create left and right column
col1, col2 = st.columns(2)

# 左栏：User ID 推荐
with col1:
    st.subheader("1️⃣ User-Based Recommendation")
    user_id = st.number_input(
        "Enter your User ID:",
        min_value=int(ratings['userId'].min()),
        max_value=int(ratings['userId'].max()),
        step=1,
        key="user_input"
    )
    if st.button("Get User Recommendations", key="user_btn"):
        recommendations = recommend_movies(user_id, n=10)
        st.subheader(f"Top 10 movies for User {user_id}:")
        for i, (title, score) in enumerate(recommendations, start=1):
            st.write(f"{i}. {title} (Predicted Rating: {score:.2f})")

# 右栏：电影搜索 & 相似电影推荐
with col2:
    st.subheader("2️⃣ Search Movie & Recommend Similar Movies")
    movie_input = st.text_input("Enter a movie name:", key="movie_input")
    if st.button("Search Similar Movies", key="movie_btn"):
        results = recommend_similar_movies(movie_input, top_n=5)
        if isinstance(results, str):
            st.write(results)
        else:
            st.subheader(f"Top 5 movies similar to '{movie_input}':")
            for i, (title, score) in enumerate(results, start=1):
                st.write(f"{i}. {title} (Similarity: {score:.2f})")
