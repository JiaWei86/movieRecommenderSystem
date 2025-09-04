# app.py
import streamlit as st
import pandas as pd
from surprise import SVD, Dataset, Reader
from collections import defaultdict
import joblib
import random
import matplotlib.pyplot as plt

# ========================
# Load Data
# ========================
@st.cache_data
def load_data():
    ratings = pd.read_csv('ml-100k/u.data', sep='\t', header=None,
                          names=['userId','movieId','rating','timestamp'])
    movies = pd.read_csv('ml-100k/u.item', sep='|', header=None, usecols=[0,1,2],
                         names=['movieId','title','year'], encoding='latin-1')
    ratings['movieId'] = ratings['movieId'].astype(int)
    movies['movieId'] = movies['movieId'].astype(int)
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
    all_movie_ids = ratings['movieId'].unique()
    user_movies = ratings[ratings['userId']==user_id]['movieId'].values
    testset = [(user_id, mid, 0) for mid in all_movie_ids if mid not in user_movies]
    predictions = model.test(testset)
    top_n = get_top_n_recommendations(predictions, n)
    
    top_movies = []
    for mid, est in top_n[user_id]:
        movie_row = movies[movies['movieId']==mid]
        title = movie_row['title'].values[0]
        year = movie_row['year'].values[0] if 'year' in movie_row.columns else ''
        top_movies.append((title, year, est))
    return top_movies

def top_rated_movies_overall(n=10):
    merged = pd.merge(ratings, movies, on='movieId')
    avg_ratings = merged.groupby(['movieId', 'title', 'year'])['rating'].mean().reset_index()
    avg_ratings = avg_ratings.sort_values('rating', ascending=False).head(n)
    return avg_ratings

# ========================
# Streamlit UI
# ========================
st.title("🎬 Movie Recommendation System")

# ----- User Recommendations -----
st.subheader("Personalized Recommendations")
user_groups = ['1-200', '201-400', '401-600', '601-800', '801-943']
selected_group = st.selectbox("Select User ID Range", user_groups)

# Filter user list by group
start, end = map(int, selected_group.split('-'))
user_list = sorted([uid for uid in ratings['userId'].unique() if start <= uid <= end])
user_id = st.selectbox("Choose User ID", user_list)

top_n = st.slider("Select number of recommended movies", 5, 20, 10)

if st.button("Get Recommendations"):
    recommendations = recommend_movies(user_id, n=top_n)
    
    st.subheader(f"Top {top_n} Recommended Movies for User {user_id}:")
    titles = []
    scores = []
    for i, (title, year, score) in enumerate(recommendations, start=1):
        st.write(f"{i}. {title} ({year}) - Predicted Rating: {score:.2f}")
        titles.append(f"{title} ({year})")
        scores.append(score)
    
    # Show bar chart
    fig, ax = plt.subplots()
    ax.barh(titles[::-1], scores[::-1], color='skyblue')
    ax.set_xlabel("Predicted Rating")
    ax.set_title(f"Top {top_n} Recommended Movie Ratings")
    st.pyplot(fig)

# ----- Random User Recommendation -----
if st.button("Random User Recommendation"):
    random_user = random.choice(user_list)
    st.subheader(f"🎉 Recommendations for Random User {random_user}")
    recommendations = recommend_movies(random_user, n=top_n)
    for i, (title, year, score) in enumerate(recommendations, start=1):
        st.write(f"{i}. {title} ({year}) - Predicted Rating: {score:.2f}")

# ----- Top Rated Movies Overall -----
st.subheader("🔥 Top Rated Movies Overall")
top_movies = top_rated_movies_overall(n=10)
for i, row in top_movies.iterrows():
    st.write(f"{i+1}. {row['title']} ({row['year']}) - Average Rating: {row['rating']:.2f}")
