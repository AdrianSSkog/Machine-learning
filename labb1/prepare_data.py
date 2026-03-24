import pandas as pd

Movie_path = r"C:\Users\adria\Desktop\Python\Machine-learning\labb1\ml-latest\movies.csv"
Rating_path = r"C:\Users\adria\Desktop\Python\Machine-learning\labb1\ml-latest\ratings.csv"
tags_path = r"C:\Users\adria\Desktop\Python\Machine-learning\labb1\ml-latest\tags.csv"

#load datasets
movieDF = pd.read_csv(Movie_path)
ratingsDF = pd.read_csv(Rating_path)
tagsDF = pd.read_csv(tags_path)

#Flytta år till egen kolumn
movieDF["year"] = movieDF["title"].str.extract(r"\((\d{4})\)")
movieDF["title"] = movieDF["title"].str.replace(r"\((\d{4})\)", "", regex=True)

#Flytta "the" från slutet till början
movieDF["title"] = movieDF["title"].str.strip()

mask = movieDF["title"].str.lower().str.endswith(", the", na=False)

new_titles ="The " + movieDF.loc[mask, "title"].str[:-6]

movieDF.loc[mask, "title"] = new_titles

#kategorisera filmer efter årtionde
movieDF["decade"] = movieDF["year"].str[:3] + "0s"

#Slå ihop movies och tags
tags_grouped = tagsDF.groupby("movieId")["tag"].apply(lambda x: " ".join(x.dropna().astype(str))).reset_index()

movieTags = movieDF.merge(tags_grouped, on="movieId", how="left")

#Fyll i Nan med tom sträng
movieTags["tag"] = movieTags["tag"].fillna("")
movieTags["year"] = movieTags["year"].fillna("")
movieTags["decade"] = movieTags["decade"].fillna("")

#lägg ihop titel, decade, genrar och tags till en text sträng i en egen kolumn 
movieTags["text"] = (movieTags["title"].astype(str) + " " + 
                     movieTags["decade"].astype(str) + " " +
                     movieTags["genres"].astype(str) + " " +
                     movieTags["tag"].astype(str))

movieTags["text"] = (movieTags["text"].str.lower().str.replace("|", " ", regex=False))

#save
movieTags.to_pickle("movieTags.pkl")
movieDF.to_pickle("movieDF.pkl")
ratingsDF.to_pickle("ratingsDF.pkl")
