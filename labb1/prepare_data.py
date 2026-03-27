import pandas as pd

Movie_path = r"C:\Users\adria\Desktop\Python\Machine-learning\labb1\ml-latest\movies.csv"
Rating_path = r"C:\Users\adria\Desktop\Python\Machine-learning\labb1\ml-latest\ratings.csv"
tags_path = r"C:\Users\adria\Desktop\Python\Machine-learning\labb1\ml-latest\tags.csv"

#load datasets
movieDF = pd.read_csv(Movie_path, engine="python")
ratingsDF = pd.read_csv(Rating_path)
tagsDF = pd.read_csv(tags_path, engine="python")

#Flytta år till egen kolumn
movieDF["year"] = movieDF["title"].str.extract(r"\((\d{4})\)")
movieDF["title"] = movieDF["title"].str.replace(r"\((\d{4})\)", "", regex=True)

#Flytta "the" från slutet till början
movieDF["title"] = movieDF["title"].str.strip()

mask1 = movieDF["title"].str.lower().str.endswith(", the", na=False)
new_titlesA ="The " + movieDF.loc[mask1, "title"].str[:-5]
movieDF.loc[mask1, "title"] = new_titlesA

mask2 = movieDF["title"].str.contains(r", The \(", na=False)
new_titlesB = "The " + movieDF.loc[mask2, "title"].str.replace(", The", "", regex=True)
movieDF.loc[mask2, "title"] = new_titlesB

movieDF["clean_title"] = movieDF["title"]

#inkludera årtal på filmtitlar som förekommer upprepade gånger
mask3 = movieDF["title"].duplicated(keep=False)
TitlesYear = movieDF.loc[mask3, "title"] + " (" + movieDF.loc[mask3, "year"].fillna("").astype(str) + ")"
movieDF.loc[mask3, "title"] = TitlesYear

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

ratingsDF.drop("timestamp", axis=1, inplace=True)

#save
movieTags.to_pickle(r"labb1\movieTags.pkl")
movieDF.to_pickle(r"labb1\movieDF.pkl")
ratingsDF.to_pickle(r"labb1\ratingsDF.pkl")

