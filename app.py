# Copyright © 2024 [Your Name or Organization Name]  # <-- REPLACE THIS
# All rights reserved.

# --- Start of your actual code ---
import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import json
# Import necessary ML libraries for content-based filtering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import project modules
from models.hybrid_recommender import HybridRecommender
from utils.data_loader import PROCESSED_DATA_DIR # Get path to processed data

# Define paths relative to app.py
MOVIES_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'movies_processed.parquet')
RATINGS_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'ratings_processed.parquet')
USER_MAP_PATH = os.path.join(PROCESSED_DATA_DIR, 'user_map.json')


# --- Data and Model Loading (Cached) ---

@st.cache_resource(show_spinner="Loading recommendation models...")
def load_hybrid_model():
    """Loads the Hybrid Recommender system."""
    try:
        recommender = HybridRecommender(mf_weight=0.5, dcn_weight=0.5)
        recommender.load_models()
        return recommender
    except FileNotFoundError:
        st.error("🔴 Error: Trained model files not found. "
                 "Please run `python train.py` successfully before launching the app.")
        return None
    except Exception as e:
        st.error(f"🔴 An unexpected error occurred loading the models: {e}")
        st.exception(e)
        return None

@st.cache_data(show_spinner="Loading movie and rating data...")
def load_processed_data():
    """Loads processed movies and ratings data."""
    if not os.path.exists(MOVIES_DATA_PATH) or not os.path.exists(RATINGS_DATA_PATH):
        st.error(f"🔴 Error: Processed data files not found at '{PROCESSED_DATA_DIR}'. "
                    "Please run `python train.py` successfully.")
        return None, None
    try:
        movies_df = pd.read_parquet(MOVIES_DATA_PATH)
        # Ensure list columns are actually lists (Parquet sometimes saves them differently)
        for col in ['genres', 'directors', 'actors']:
            if col in movies_df.columns:
                # Apply conversion only if not already list/object type suggesting list
                if len(movies_df) > 0 and not pd.api.types.is_list_like(movies_df[col].iloc[0]):
                    # Added check for empty df
                    movies_df[col] = movies_df[col].apply(lambda x: list(x) if isinstance(x, (np.ndarray, pd.Series)) else x if pd.api.types.is_list_like(x) else [])
                    # Ensure result is list-like or empty list

        ratings_df = pd.read_parquet(RATINGS_DATA_PATH)
        print(f"Loaded processed movies: {movies_df.shape}")
        print(f"Loaded processed ratings: {ratings_df.shape}")
        return movies_df, ratings_df
    except Exception as e:
        st.error(f"🔴 Error loading processed data: {e}")
        return None, None

@st.cache_data
def load_user_ids():
    """Loads the list of user IDs."""
    if os.path.exists(RATINGS_DATA_PATH):
        try:
            ratings_df = pd.read_parquet(RATINGS_DATA_PATH)
            if 'userId' in ratings_df.columns:
                user_ids = sorted(ratings_df['userId'].unique())
                print(f"Loaded {len(user_ids)} unique user IDs from ratings file.")
                return user_ids
            else:
                st.error("🔴 'userId' column not found in ratings_processed.parquet.")
                return []
        except Exception as e:
            st.error(f"🔴 Error loading user IDs from ratings file: {e}")
            return []
    else:
        st.error("🔴 Processed ratings data file not found, cannot load user IDs.")
        return []

# --- Content-Based Similarity Calculation ---

# Helper to create feature string, handling lists
def create_feature_string(row):
    # Join list elements, convert others to string, handle None/NaN
    try:
        genres = ' '.join(map(str, row.get('genres', [])))
    except TypeError: genres = ''
    try:
        directors = ' '.join(map(str, row.get('directors', []))).replace(' ','') # Remove spaces in names
    except TypeError: directors = ''
    try:
        actors = ' '.join(map(str, row.get('actors', []))).replace(' ','') # Remove spaces in names
    except TypeError: actors = ''
    year = str(row.get('year', ''))
    runtime = str(row.get('runtimeMinutes', ''))
    # Combine features - adjust weighting by repeating terms if needed
    return f"{genres} {directors} {actors} {year} {runtime}"

@st.cache_resource(show_spinner="Calculating movie content similarity...")
def build_tfidf_matrix(_movies_df):
    """Builds the TF-IDF matrix from movie features."""
    if _movies_df is None:
        return None, None

    print("Building TF-IDF matrix for content-based similarity...")
    # Create a copy WITH A RESET INDEX to ensure 0-based positional indexing aligns
    movies_copy = _movies_df.copy().reset_index(drop=True)
    # Ensure required columns exist
    if not all(col in movies_copy.columns for col in ['genres', 'directors', 'actors', 'year', 'runtimeMinutes']):
        st.error("🔴 Missing required columns (genres, directors, actors, year, runtimeMinutes) for TF-IDF calculation.")
        return None, None

    # Apply the helper function to create the combined feature string
    movies_copy['combined_features'] = movies_copy.apply(create_feature_string, axis=1)

    tfidf = TfidfVectorizer(stop_words='english', max_features=10000) # Limit features
    tfidf_matrix = tfidf.fit_transform(movies_copy['combined_features'])
    print(f"TF-IDF matrix built with shape: {tfidf_matrix.shape}")
    # Return matrix and the df copy (which now has a 0-based index)
    return tfidf_matrix, movies_copy

# --- *** UPDATED get_content_recommendations function *** ---
@st.cache_data(show_spinner="Finding similar movies...") # Added spinner message
def get_content_recommendations(movie_title, _movies_df_for_tfidf, _tfidf_matrix, top_n=10):
    """Finds similar movies using TF-IDF and cosine similarity."""
    if _movies_df_for_tfidf is None or _tfidf_matrix is None:
        st.warning("TF-IDF data not available for content recommendations.")
        return pd.DataFrame()

    # --- Find the POSITIONAL index (row number) in the TF-IDF DataFrame ---
    # Use the DataFrame's index which was reset to 0-based during build_tfidf_matrix
    matching_indices = _movies_df_for_tfidf.index[_movies_df_for_tfidf['title'].str.lower() == movie_title.lower()]

    if matching_indices.empty:
        st.warning(f"Movie '{movie_title}' not found for content similarity.")
        return pd.DataFrame()

    # Get the first matching POSITIONAL index (since index is now 0-based)
    positional_idx = matching_indices[0]
    # --- End Finding Positional Index ---

    # --- Compute cosine similarity using the positional index ---
    try:
        # Check if positional_idx is within the bounds of the matrix rows
        if positional_idx >= _tfidf_matrix.shape[0]:
            st.error(f"Internal Error: Positional index {positional_idx} is out of bounds for TF-IDF matrix shape {_tfidf_matrix.shape}.")
            return pd.DataFrame()

        cosine_sim = cosine_similarity(_tfidf_matrix[positional_idx], _tfidf_matrix).flatten()
    except IndexError:
        st.error(f"IndexError during cosine similarity for positional index {positional_idx}. Matrix shape: {_tfidf_matrix.shape}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error during cosine similarity calculation: {e}")
        return pd.DataFrame()
    # --- End Cosine Similarity Calculation ---

    # Get indices of top N similar movies (argsort returns indices relative to cosine_sim array)
    # These are also positional indices for the _tfidf_matrix and _movies_df_for_tfidf
    sim_indices = cosine_sim.argsort()[::-1][1:top_n+1]

    # --- Retrieve results using iloc (positional lookup) ---
    try:
        # Ensure sim_indices are within bounds before using iloc
        valid_sim_indices = [i for i in sim_indices if i < len(_movies_df_for_tfidf)]
        if not valid_sim_indices:
            st.warning("No valid similar movie indices found.")
            return pd.DataFrame()

        similar_movies = _movies_df_for_tfidf.iloc[valid_sim_indices].copy()
        # Get similarity scores corresponding to the valid indices
        similar_movies['similarity_score'] = cosine_sim[valid_sim_indices]
        return similar_movies
    except IndexError:
        st.error(f"IndexError retrieving similar movies using iloc with indices: {valid_sim_indices}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error retrieving similar movies: {e}")
        return pd.DataFrame()
# --- *** END UPDATED FUNCTION *** ---


# --- Helper Function for Display ---
def display_movie_details(movie_series, show_score=False): # Added optional score display
    """Formats and displays details for a single movie."""
    title = movie_series.get('title', 'N/A')
    year = movie_series.get('year', 'N/A')
    if isinstance(year, (int, float)) and year > 0:
        st.markdown(f"**{title} ({int(year)})**")
    else:
        st.markdown(f"**{title}**")

    genres = movie_series.get('genres', [])
    try: # Ensure genres is iterable list of strings
        genres_list = list(map(str, genres)) if pd.api.types.is_list_like(genres) else []
    except TypeError:
        genres_list = []
    if len(genres_list) > 0:
        st.caption(f"Genres: {', '.join(genres_list)}")

    directors = movie_series.get('directors', [])
    try:
        directors_list = list(map(str, directors)) if pd.api.types.is_list_like(directors) else []
    except TypeError:
        directors_list = []
    if len(directors_list) > 0 and directors_list != ['Unknown']:
        st.caption(f"Director(s): {', '.join(directors_list)}")

    actors = movie_series.get('actors', [])
    try:
        actors_list = list(map(str, actors)) if pd.api.types.is_list_like(actors) else []
    except TypeError:
        actors_list = []
    if len(actors_list) > 0:
        max_actors = 4
        actors_display = ', '.join(actors_list[:max_actors])
        if len(actors_list) > max_actors:
            actors_display += ", ..."
        st.caption(f"Actors: {actors_display}")

    runtime = movie_series.get('runtimeMinutes', 0)
    if isinstance(runtime, (int, float)) and runtime > 0:
        st.caption(f"Runtime: {int(runtime)} min")

    if show_score and 'similarity_score' in movie_series:
        st.caption(f"Similarity: {movie_series['similarity_score']:.3f}")

    st.markdown("---")


# --- Main App Logic ---

st.set_page_config(layout="wide", page_title="Movie Recommender")
st.title("🎬 Hybrid & Content-Based Movie Recommendation System")


# --- Load Data and Models ---
hybrid_placeholder = st.empty()
data_placeholder = st.empty()
user_id_placeholder = st.empty()
tfidf_placeholder = st.empty()

with hybrid_placeholder: hybrid_recommender = load_hybrid_model()
with data_placeholder: movies_df, ratings_df = load_processed_data()
with user_id_placeholder: available_user_ids = load_user_ids()
# Build TF-IDF matrix using the loaded movies_df
with tfidf_placeholder: tfidf_matrix, movies_df_for_tfidf = build_tfidf_matrix(movies_df)

hybrid_placeholder.empty()
data_placeholder.empty()
user_id_placeholder.empty()
tfidf_placeholder.empty()

# --- Check if essential components loaded ---
if movies_df is None:
    st.error("🔴 Failed to load movie data. Cannot start the application.")
    st.stop()
# Content search requires both matrix and the specific df used to build it
content_search_enabled = tfidf_matrix is not None and movies_df_for_tfidf is not None
if not content_search_enabled:
    st.warning("Content search components failed to load or build. Content search tab disabled.")


# --- Main Application Tabs ---
tab1, tab2 = st.tabs(["**✨ Hybrid Recommendations (For User)**", "**🔍 Find Similar Movies (Content-Based)**"])


# ============================ TAB 1: Hybrid Recommendations ============================
with tab1:
    # ... (Code for Tab 1 remains the same as previous correct version) ...
    st.header("Hybrid Recommendations (SVD + DCN)")
    st.write("Get personalized movie suggestions for a selected user.")

    if hybrid_recommender and hybrid_recommender.models_loaded and ratings_df is not None and available_user_ids:

        if not hybrid_recommender.data_set:
            try:
                all_movie_ids_list = movies_df['movieId'].unique()
                hybrid_recommender.set_data(all_movie_ids_list, ratings_df)
            except Exception as e:
                st.error(f"🔴 Error setting data for recommender: {e}")
                st.stop() # Stop this tab if setup fails

        # --- Sidebar for User Input ---
        st.sidebar.header("👤 User Selection (Hybrid)")
        if not available_user_ids:
            st.sidebar.warning("No users available.")
            # st.stop() # Don't stop the whole app, just this tab's logic
        else:
            MAX_USERS_DROPDOWN = 5000
            user_selection_list = available_user_ids[:MAX_USERS_DROPDOWN]
            if len(available_user_ids) > MAX_USERS_DROPDOWN:
                st.sidebar.info(f"Showing first {MAX_USERS_DROPDOWN} users.")

            selected_user_id = st.sidebar.selectbox(
                "Select a User ID:",
                user_selection_list,
                index=0,
                key="hybrid_user_select" # Unique key for this widget
            )

            st.sidebar.header("⚙️ Recommendation Settings (Hybrid)")
            num_hybrid_recommendations = st.sidebar.slider(
                "Number of recommendations:", min_value=5, max_value=30, value=10, step=1,
                key="hybrid_num_recs"
                )
            exclude_seen = st.sidebar.checkbox(
                "Exclude movies already rated?", value=True,
                key="hybrid_exclude_seen"
                )

            st.sidebar.markdown("---")
            st.sidebar.header("💡 Model Weights (Hybrid)")
            mf_weight = st.sidebar.slider(
                "SVD Weight:", min_value=0.0, max_value=1.0, value=hybrid_recommender.mf_weight, step=0.05,
                key="hybrid_mf_weight"
                )
            dcn_weight = 1.0 - mf_weight
            st.sidebar.caption(f"DCN Weight: {dcn_weight:.2f}")

            # --- Main Area for Hybrid Recommendations ---
            st.subheader(f"Recommendations for User: {selected_user_id}")

            if st.button(f"Get Top {num_hybrid_recommendations} Hybrid Recommendations", key="hybrid_get_recs"):
                print(f"Button clicked: Get hybrid recommendations for user {selected_user_id}")
                start_rec_time = time.time()
                with st.spinner(f"⏳ Generating hybrid recommendations..."):
                    try:
                        recommended_movie_ids = hybrid_recommender.recommend(
                            user_id=selected_user_id,
                            n=num_hybrid_recommendations,
                            exclude_seen=exclude_seen
                        )
                        print(f"Hybrid recommended IDs: {recommended_movie_ids}")

                        if recommended_movie_ids:
                            st.success("Recommendations generated!")
                            valid_indices = [idx for idx in recommended_movie_ids if idx in movies_df['movieId'].values]
                            if not valid_indices:
                                st.warning("Could not find details for any recommended movie IDs.")
                            else:
                                recommended_movies_df = movies_df.set_index('movieId').loc[valid_indices]
                                recommended_movies_df = recommended_movies_df.reindex(recommended_movie_ids).reset_index()

                                num_cols = 5
                                cols = st.columns(num_cols)
                                for i, (_, movie_row) in enumerate(recommended_movies_df.iterrows()):
                                    with cols[i % num_cols]:
                                        display_movie_details(movie_row) # Don't show score here
                        else:
                            st.warning("Could not generate recommendations for this user.")

                    except Exception as e:
                        st.error(f"🔴 An error occurred during hybrid recommendation:")
                        st.exception(e)

                end_rec_time = time.time()
                st.info(f"Hybrid recommendation took {end_rec_time - start_rec_time:.2f} seconds.")

            # --- Optional: Show User's Rated Movies ---
            st.markdown("---")
            if st.expander(f"View Movies Rated by User {selected_user_id}"):
                # ... (Code for showing user ratings remains the same) ...
                user_ratings = ratings_df[ratings_df['userId'] == selected_user_id].sort_values('rating', ascending=False)
                if not user_ratings.empty:
                    user_ratings_with_titles = user_ratings.merge(movies_df[['movieId', 'title']], on='movieId', how='left')
                    display_cols = ['title', 'rating']
                    if 'timestamp' in user_ratings_with_titles.columns:
                        display_cols.append('timestamp')
                    st.dataframe(user_ratings_with_titles[display_cols].head(20))
                else:
                    st.write("This user has no ratings in the processed dataset.")


    else:
        st.warning("Hybrid recommendation components failed to load. This feature is disabled.")


# ============================ TAB 2: Content-Based Search ============================
with tab2:
    st.header("Find Similar Movies (Content-Based)")
    st.write("Search for a movie and find others with similar genres, cast, crew, etc.")

    # Use the flag defined earlier
    if content_search_enabled:

        # --- Search Input ---
        search_term = st.text_input("Search for a movie title:", key="content_search_term")

        # Filter movies based on search term for the selectbox
        if search_term:
            mask = movies_df_for_tfidf['title'].str.contains(search_term, case=False, na=False)
            search_results_titles = movies_df_for_tfidf.loc[mask, 'title'].tolist()
        else:
            search_results_titles = []

        if search_term and not search_results_titles:
            st.info("No movies found matching your search term in the processed dataset.")

        if search_results_titles:
            max_search_results = 100
            if len(search_results_titles) > max_search_results:
                st.info(f"Showing top {max_search_results} matches for '{search_term}'.")
                search_results_titles = search_results_titles[:max_search_results]

            selected_movie_title = st.selectbox(
                "Select a movie from search results:",
                search_results_titles,
                key="content_movie_select"
            )

            num_content_recommendations = st.slider(
                "Number of similar movies to find:", min_value=5, max_value=30, value=10, step=1,
                key="content_num_recs"
                )

            if st.button(f"Find Top {num_content_recommendations} Similar Movies", key="content_get_recs"):
                print(f"Button clicked: Get content recommendations for '{selected_movie_title}'")
                start_rec_time = time.time()
                with st.spinner(f"⏳ Finding movies similar to '{selected_movie_title}'..."):
                    try:
                        # Call the updated function
                        similar_movies_df = get_content_recommendations(
                            movie_title=selected_movie_title,
                            _movies_df_for_tfidf=movies_df_for_tfidf, # Pass the df used to build matrix
                            _tfidf_matrix=tfidf_matrix,              # Pass the matrix
                            top_n=num_content_recommendations
                        )

                        if not similar_movies_df.empty:
                            st.success("Similar movies found!")
                            st.subheader(f"Movies similar to '{selected_movie_title}':")

                            num_cols = 5
                            cols = st.columns(num_cols)
                            for i, (_, movie_row) in enumerate(similar_movies_df.iterrows()):
                                    with cols[i % num_cols]:
                                        display_movie_details(movie_row, show_score=True) # Show similarity score
                        else:
                            # Warnings are now handled inside get_content_recommendations
                            pass # Or add a generic message here if needed

                    except Exception as e:
                        st.error(f"🔴 An error occurred during content-based recommendation:")
                        st.exception(e)

                end_rec_time = time.time()
                st.info(f"Similarity search took {end_rec_time - start_rec_time:.2f} seconds.")
        elif search_term:
            pass
        else:
            st.info("Enter a movie title in the search bar above to find similar movies.")

    else:
        # Message moved to the top check
        st.warning("Content-based search is disabled as components failed to load.") # Explicit message here


# --- Footer or general info ---
st.markdown("---")
st.caption("Movie Recommendation Demo | SVD + DCN Hybrid & Content-Based TF-IDF")