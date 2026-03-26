import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_Key = os.getenv("TMDB_API_KEY")

size = "w300"
Base_URL = "https://api.themoviedb.org/3"
Image_URL = f"https://image.tmdb.org/t/p/{size}"

poster_cache = {}

def get_movie_poster(movie_title, year):
    if movie_title in poster_cache:
        return poster_cache[movie_title]
    
    url = f"{Base_URL}/search/movie"
    params = {
        "api_key" : API_Key,
        "query" : movie_title,
    }
    if year:
        params["year"] = year

    response = requests.get(url, params=params)

    if response.status_code != 200:
        return None
    
    data = response.json()
    results = data.get("results")

    if results:
        file_path = results[0].get("poster_path")
        if file_path:
            poster_url = Image_URL + file_path
            poster_cache[movie_title] = poster_url
            return poster_url
        
    return "https://via.placeholder.com/200x300?text=No+Image"