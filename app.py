from flask import Flask,render_template,request,url_for
from sklearn.neighbors import NearestNeighbors
import pickle
import numpy as np
import pandas as pd

app=Flask(__name__)
model1=pickle.load(open('engine_tu.pkl','rb'))
movie_tag_pivot=pickle.load(open('movie_tag_pivot_table_tu.pkl','rb'))
movie = pd.read_csv("movie.csv")

scores_movie=movie[movie['movieId'].isin(movie_tag_pivot.index)]


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    search=request.form.get('search')
    search_results=scores_movie[scores_movie['title'].str.contains(search,case=False)]
    if len(search_results.index)>0 :
        return render_template('search.html',id=list(search_results['movieId']),m=list(search_results['title']),l=len(search_results.index))
    else:
        return render_template('error.html')

@app.route('/error')
def error():
    return render_template('error.html')

def rec(movie_id):
    movie_id = int(movie_id)
    distances,suggestions=model1.kneighbors(movie_tag_pivot.loc[movie_id,:].values.reshape(1,-1),n_neighbors=16)
    return movie_tag_pivot.iloc[suggestions[0]].index


@app.route('/recommend/<string:type>')
def recommend(type): #type variable is nothing but the movie id
    recommendations = rec(type)
    movies_rec = []
    for movie_id in recommendations[1:]:
        movies_rec.append(movie[movie['movieId']==movie_id]['title'].values[0])
    if len(movies_rec)>0:
        mr = movies_rec
        return render_template('recommend.html', a=mr[0],b=mr[1:])
    else:
        return render_template('error.html')


if __name__ == '__main__':
    app.run(debug = True)