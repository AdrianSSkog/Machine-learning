import pandas as pd

Movie_path = r"C:\Users\adria\Desktop\Python\Machine-learning\labb1\ml-latest\movies.csv"
Rating_path = r"C:\Users\adria\Desktop\Python\Machine-learning\labb1\ml-latest\ratings.csv"
tags_path = r"C:\Users\adria\Desktop\Python\Machine-learning\labb1\ml-latest\tags.csv"

#load datasets
movieDF = pd.read_csv(Movie_path)
ratingsDF = pd.read_csv(Rating_path)
tagsDF = pd.read_csv(tags_path)

#processing
movieDF["year"] = movieDF["title"].str.extract(r"\((\d{4})\)")
movieDF["title"] = movieDF["title"].str.replace(r"\((\d{4})\)", "", regex=True)

tags_grouped = tagsDF.groupby("movieId")["tag"].apply(lambda x: " ".join(x.dropna().astype(str))).reset_index()

movieDF["title"] = movieDF["title"].str.strip()

movieDF["decade"] = movieDF["year"].str[:3] + "0s"

movieTags = movieDF.merge(tags_grouped, on="movieId", how="left")

movieTags["tag"] = movieTags["tag"].fillna("")
movieTags["year"] = movieTags["year"].fillna("")
movieTags["decade"] = movieTags["decade"].fillna("")

movieTags["text"] = (movieTags["title"].astype(str) + " " + 
                     movieTags["decade"].astype(str) + " " +
                     movieTags["genres"].astype(str) + " " +
                     movieTags["tag"].astype(str))

movieTags["text"] = (movieTags["text"].str.lower().str.replace("|", " ", regex=False))

#save
movieTags.to_pickle("movieTags.pkl")
movieDF.to_pickle("movieDF.pkl")
ratingsDF.to_pickle("ratingsDF.pkl")
