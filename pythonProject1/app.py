import pickle
import streamlit as st
import requests
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# Streamlit UI setup
st.set_page_config(page_title="🎬 Hybrid Movie Recommender System", layout="wide")

# Function to fetch movie poster from TMDB
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"
        else:
            return None  # No poster available
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching poster: {e}")
        return None

# Function to fetch movie trailer from TMDB
def fetch_trailer(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data['results']:
            for video in data['results']:
                if video['site'] == 'YouTube' and video['type'] == 'Trailer':
                    return f"https://www.youtube.com/watch?v={video['key']}"
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching trailer: {e}")
        return None

# Function to fetch movie details, including homepage
def fetch_movie_details(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Extract relevant details
        title = data.get('title', 'Unknown Title')
        release_date = data.get('release_date', 'N/A')
        rating = data.get('vote_average', 'N/A')
        genres = [genre['name'] for genre in data.get('genres', [])]
        homepage = data.get('homepage')  # Official movie homepage
        if not homepage:
            homepage = f"https://www.themoviedb.org/movie/{movie_id}"  # Fallback to TMDB link

        return title, release_date, rating, genres, homepage
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching movie details: {e}")
        return 'Unknown Title', 'N/A', 'N/A', [], None

# Function to recommend similar movies based on cosine similarity (Content-based Filtering)
def content_based_recommend(movie):
    try:
        index = movies[movies['title'] == movie].index[0]
        distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
        recommended_movie_details = []

        for i in distances[1:6]:  # Top 5 recommendations
            movie_id = movies.iloc[i[0]].movie_id
            poster_url = fetch_poster(movie_id)
            title, release_date, rating, genres, homepage = fetch_movie_details(movie_id)
            trailer_url = fetch_trailer(movie_id)

            # Append all relevant movie details in a dictionary
            recommended_movie_details.append({
                'title': title,
                'poster': poster_url,
                'release_date': release_date,
                'rating': rating,
                'genres': genres,
                'trailer_url': trailer_url,
                'homepage': homepage  # Add homepage link
            })

        return recommended_movie_details
    except IndexError:
        st.error("Movie not found in the dataset.")
        return []

# Function to recommend movies using collaborative filtering (SVD)
def collaborative_filtering_recommend(user_id):
    predictions = []
    for movie_id in movies['movie_id'].unique():
        predicted_rating = svd.predict(user_id, movie_id).est
        predictions.append((movie_id, predicted_rating))
    recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]

    recommended_movie_details = []
    for movie_id, rating in recommendations:
        poster_url = fetch_poster(movie_id)
        title, release_date, rating, genres, homepage = fetch_movie_details(movie_id)
        trailer_url = fetch_trailer(movie_id)

        recommended_movie_details.append({
            'title': title,
            'poster': poster_url,
            'release_date': release_date,
            'rating': rating,
            'genres': genres,
            'trailer_url': trailer_url,
            'homepage': homepage
        })

    return recommended_movie_details

# Hybrid recommender system (combining content-based and collaborative filtering)
def hybrid_recommend(movie, user_id=None):
    if user_id:
        # Use both content-based and collaborative filtering
        content_based_recs = content_based_recommend(movie)
        collaborative_recs = collaborative_filtering_recommend(user_id)

        # Combine both recommendations (here, we just return both sets for simplicity)
        return content_based_recs + collaborative_recs
    else:
        # Fallback to content-based if no user_id is provided
        return content_based_recommend(movie)

# Load the movie data and similarity matrix
@st.cache_data
def load_data():
    movies = pickle.load(open('movie_list.pkl', 'rb'))
    similarity = pickle.load(open('similarity.pkl', 'rb'))
    return movies, similarity

# SVD (Collaborative Filtering) Model Training
def train_svd(df):
    reader = Reader()
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.25)
    svd = SVD()
    svd.fit(trainset)
    return svd

# Load data and model
movies, similarity = load_data()

# Define a sample DataFrame `df` for SVD model training (replace this with your actual user data)
# Assuming `df` is in the format with columns: ['userId', 'movieId', 'rating']
df = pd.DataFrame({
    'userId': [1, 1, 2, 2, 3, 3],
    'movieId': [1, 2, 2, 3, 1, 3],
    'rating': [5, 3, 4, 2, 1, 5]
})

svd = train_svd(df)

# Custom CSS styling for buttons, headers, and cards
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .movie-title {
        font-weight: bold;
        font-size: 18px;
        margin-bottom: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description with emojis
st.title('🎥 Hybrid Movie Recommender System')
st.markdown("✨ **Welcome to your personalized movie recommender!** Choose a movie from the dropdown, and we’ll suggest similar movies just for you!")

# Movie selection dropdown
movie_list = movies['title'].values
selected_movie = st.selectbox("🎬 Select a Movie from the dropdown", movie_list)

# User ID input for collaborative filtering
user_id = st.text_input("👤 Enter your User ID (Optional for collaborative filtering)")

# Show recommendations when the button is clicked
if st.button('🍿 Show Recommendation'):
    with st.spinner('🍿 Fetching movie recommendations...'):
        recommended_movie_details = hybrid_recommend(selected_movie, user_id=user_id)

        # If we have recommendations, display them
        if recommended_movie_details:
            st.subheader(f"Here are movies similar to **{selected_movie}** 🎥")
            cols = st.columns(5)

            for idx, col in enumerate(cols):
                with col:
                    # Display poster
                    movie = recommended_movie_details[idx]
                    if movie['poster']:
                        st.image(movie['poster'], use_column_width=True)
                    else:
                        st.text("Poster not available")

                    # Display movie details
                    st.markdown(f"<div class='movie-title'>{movie['title']}</div>", unsafe_allow_html=True)
                    st.markdown(f"**Release Date**: {movie['release_date']}")
                    st.markdown(f"**Rating**: ⭐ {movie['rating']}/10")
                    st.markdown(f"**Genres**: {', '.join(movie['genres']) if movie['genres'] else 'N/A'}")

                    # Display movie link
                    st.markdown(f"[🔗 Full Movie Details]({movie['homepage']})", unsafe_allow_html=True)

                    # Display trailer link if available
                    if movie['trailer_url']:
                        st.markdown(f"[🎬 Watch Trailer]({movie['trailer_url']})", unsafe_allow_html=True)
                    else:
                        st.text("Trailer not available")
        else:
            st.error("❌ No recommendations found.")
