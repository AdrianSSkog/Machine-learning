import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

def load_data():
    movieDF = pd.read_pickle(r"labb1\movieDF.pkl")
    ratingsDF = pd.read_pickle(r"labb1\ratingsDF.pkl")
    movieTags = pd.read_pickle(r"labb1\movieTags.pkl")
    tfidf_matrix = vectorize_text(movieTags["text"])

    return movieDF, ratingsDF, movieTags, tfidf_matrix

#Content based functions

def vectorize_text(texts):
    vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 3),
    min_df= 4,
    sublinear_tf=True
    )

    return vectorizer.fit_transform(texts)

def get_movie_index(movie_title, movieDF):
    matches = movieDF.index[movieDF["title"].str.lower() == movie_title.lower()].tolist()

    if not matches:
        raise ValueError(f"The movie title: {movie_title} was not found")
    return matches[0]
    

def get_movie_titles(movie_indices, movieDF):
    return movieDF.iloc[movie_indices]["clean_title"].tolist()

def rertieval(movie_title, tfidf_matrix, movieDF, top_n=200):
    movie_indx = get_movie_index(movie_title, movieDF)

    movie_vector = tfidf_matrix[movie_indx]
    similarities = cosine_similarity(movie_vector, tfidf_matrix).flatten()
    simMovieIndx = similarities.argsort()[::-1][1:top_n+1]

    simMovies = get_movie_titles(simMovieIndx, movieDF)
    return simMovies, simMovieIndx

#Collaborative filtering functions

def get_ratings(movies_byIndex, input_title, ratingsDF, movieTags):

    candidates_Id = movieTags.iloc[movies_byIndex]["movieId"]
    inputMovie_id = movieTags.loc[movieTags["title"].str.lower() == input_title.lower(), "movieId"].iloc[0]
    movies_to_keep = candidates_Id.tolist() + [inputMovie_id]
    candidateRatings = ratingsDF[ratingsDF["movieId"].isin(movies_to_keep)]
    candidateRatings = candidateRatings.drop("timestamp", axis=1)
    
    return candidateRatings

def build_sparse_matrix(ratings):
    rows = ratings["userId"].astype("category").cat.codes
    cols = ratings["movieId"].astype("category").cat.codes
    data = ratings["rating"]

    return csr_matrix((data, (rows, cols)))

def get_top5_from_ratings(rating_matrix, ratings, inputMovie_id, movieDF):
    movie_similarity = cosine_similarity(rating_matrix.T)

    movie_categories = ratings["movieId"].astype("category")
    movieId_lookup = movie_categories.cat.categories

    if inputMovie_id in movieId_lookup:
        input_index = list(movieId_lookup).index(inputMovie_id)

        similarities = movie_similarity[input_index]

    else:
        similarities = None

    if similarities is None:
        return []

    else:
        sorted_sim_indx = np.argsort(similarities)[::-1]

        sim_movieIds = movieId_lookup[sorted_sim_indx]

        rec_Ids = sim_movieIds.drop(inputMovie_id)

    return movieDF.loc[movieDF["movieId"].isin(rec_Ids[:5]), "title"] 

def ranking(candidateIndex, inputMovieTitle, movieDF, ratingsDF, movieTags):
    candRatings = get_ratings(candidateIndex, inputMovieTitle, ratingsDF, movieTags)
    rating_matrix = build_sparse_matrix(candRatings)

    inputMovie_id = movieTags.loc[movieTags["title"].str.lower() == inputMovieTitle.lower(), "movieId"].iloc[0]

    if len(candRatings) > 0:
        return get_top5_from_ratings(rating_matrix, candRatings, inputMovie_id, movieDF)
    else:
        return get_movie_titles(candidateIndex, movieDF)[:5]

#Combined hybrid filter

def hybrid_recommender(movieTitle, tfidf_matrix, movieDF, ratingsDF, movieTags):

    candidatesIndx = rertieval(movieTitle, tfidf_matrix, movieDF, top_n=200)[1]

    return ranking(candidatesIndx, movieTitle, movieDF, ratingsDF, movieTags).tolist()


    