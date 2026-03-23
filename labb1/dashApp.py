import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
from recommender import load_data, twostage_RetrievalRanking

data = {}
data["movieDF"], data["ratingsDF"], data["movieTags"], data["tfidf_matrix"] = load_data()

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Movie recommender", className="title"),

    dcc.Dropdown(
        id="movie_input",
        placeholder="Type to search movie..",
        searchable=True,
        className="dropdown"
    ),

    html.Button("Get recommendations", id="submit-btn", className="btn"),

    html.Div(id="output", className="output")
], className="container")


@app.callback(
    Output("movie_input", "options"),
    Input("movie_input", "search_value"),
)
def update_dropdown(search_value):

    if not search_value:
        return []
    df = data["movieDF"]
    filtered = df[df["title"].str.contains(search_value, case=False, na=False)]["title"].dropna().unique()[:20]
    return [{"label" : t, "value" : t} for t in filtered]

@app.callback(
        Output("output", "children"),
        Input("submit-btn", "n_clicks"),
        State("movie_input", "value")
)
def update_recommendations(n_clicks, movie_title):
    if not n_clicks:
        return ""

    if not movie_title:
        return "Skriv en filmtitel."
    try:
        recs = twostage_RetrievalRanking(movie_title, 
                                         data["tfidf_matrix"],
                                         data["movieDF"],
                                         data["ratingsDF"],
                                         data["movieTags"])

        return html.Ul([
            html.Li(movie) for movie in recs
        ])
    except Exception as e:
        return f"Error: {str(e)}"



if __name__ == "__main__":
    app.run(debug=True)
