# Lab 1: Movie recommendation system

This project implements a hybrid recommendation system using data from movie lens.
It combines content based filtering and collaborative filtering to recommend 5 movies based on a given input.

## Data

- movies.csv
- ratings.csv
- tags.csv

## Method

1. Content based filtering:
- TF-IDF voctorization on movie metadata
- Cosine similarity to retrive candidates

2. Collaborative filtering: 
- User Rating matrix
- Item - item Cosine similarity based on ratings to get top movies

3. Hybrid 
- Top-n candidates are ranked to get top 5 movies

## How to run

1. Clone repository

2. Install requirements:
pip install -r requirements.txt

3. Add your tmdb API key to a .env file:
TMDB_API_KEY = "your key here"

4. Run data preprocessing:
python prepare_data.py

5. run dash app:
python dashApp.py

