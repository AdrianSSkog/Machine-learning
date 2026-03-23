import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

movieDF = pd.read_pickle("movieDF.pkl")
ratingsDF = pd.read_pickle("ratingsDF.pkl")
movieTags = pd.read_pickle("movieTags.pkl")

#Content based functions

def vectorize_text(texts):
    vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df= 2
    )

    return vectorizer.fit_transform(texts)

def get_movie_index(movie_title):
    matches = movieDF.index[movieDF["title"].str.lower() == movie_title.lower()].tolist()

    if not matches:
        raise ValueError(f"The movie title: {movie_title} was not found")
    return matches[0]
    

def get_movie_titles(movie_indices):
    return movieDF.iloc[movie_indices]["title"].tolist()

def rertieval(movie_title, tfidf_matrix, top_n=200):
    movie_indx = get_movie_index(movie_title)

    movie_vector = tfidf_matrix[movie_indx]
    similarities = cosine_similarity(movie_vector, tfidf_matrix).flatten()
    simMovieIndx = similarities.argsort()[::-1][1:top_n+1]

    simMovies = get_movie_titles(simMovieIndx)
    return simMovies, simMovieIndx

#Collaborative filtering functions

def get_ratings(movies_byIndex, input_title):

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

def get_top5_from_ratings(rating_matrix, ratings, inputMovie_id):
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

def ranking(candidateIndex, inputMovieTitle):
    candRatings = get_ratings(candidateIndex, inputMovieTitle)
    rating_matrix = build_sparse_matrix(candRatings)

    inputMovie_id = movieTags.loc[movieTags["title"].str.lower() == inputMovieTitle.lower(), "movieId"].iloc[0]

    if len(candRatings) > 0:
        return get_top5_from_ratings(rating_matrix, candRatings, inputMovie_id)
    else:
        return get_movie_titles(candidateIndex)[:5]

#Combined hybrid filter

def twostage_RetrievalRanking(movieTitle, tfidf_matrix):

    candidatesIndx = rertieval(movieTitle, tfidf_matrix, top_n=200)[1]

    return ranking(candidatesIndx, movieTitle).tolist()


def get_Input_Movie():


    while True:
        user_input = input("Title of the movie: ").strip().lower()

        if user_input.startswith("the "):
            user_input = user_input[4:]+", the"
        
        matches = movieDF[movieDF["title"].str.lower() == user_input]
        if len(matches) >= 1:
            return matches.iloc[0]["title"]
        
        partial = movieDF[movieDF["title"].str.lower().str.contains(user_input)]

        if len(partial) == 1:
            return partial.iloc[0]["title"]
        elif len(partial) > 1:
            if partial.iloc[0]["title"] == partial.iloc[1]["title"]:
                return partial.iloc[0]["title"]
            else:
                print("Multiple matches found: ")
                for m in partial["title"]:
                    print(m)
                continue
        
        print("Movie was not found! Check spelling or try another one.")
    

def main():
    tfidf_matrix = vectorize_text(movieTags["text"])

    while True:
       

        input_movie = get_Input_Movie()

        recommendations = twostage_RetrievalRanking(input_movie, tfidf_matrix)

        print()
        print("Recomendations: ")

        for m in recommendations:
            print(m)

if __name__ == "__main__":
    main()