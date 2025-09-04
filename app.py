# app.py
import streamlit as st
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from collections import defaultdict
import joblib

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
# Load Model
# ========================
model = joblib.load('model.joblib')  

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
    # Create testset: all movies for this user
    all_movie_ids = ratings['movieId'].unique()
    user_movies = ratings[ratings['userId']==user_id]['movieId'].values
    testset = [(user_id, mid, 0) for mid in all_movie_ids if mid not in user_movies]
    
    predictions = model.test(testset)
    top_n = get_top_n_recommendations(predictions, n)
    top_movies = [(movies[movies['movieId']==mid]['title'].values[0], est) for mid, est in top_n[user_id]]
    return top_movies

# ========================
# Streamlit UI
# ========================
st.title("Movie Recommendation System 🎬")
st.write("Enter your User ID to get movie recommendations.")

user_id = st.number_input("User ID", min_value=int(ratings['userId'].min()), max_value=int(ratings['userId'].max()), step=1)

if st.button("Get Recommendations"):
    recommendations = recommend_movies(user_id, n=10)
    st.subheader(f"Top 10 movie recommendations for User {user_id}:")
    for i, (title, score) in enumerate(recommendations, start=1):
        st.write(f"{i}. {title} (Predicted Rating: {score:.2f})")
