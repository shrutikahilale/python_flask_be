import flask
from flask import request
from flask import jsonify 
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

app = flask.Flask(__name__)

# Load the data and train the model
product_descriptions = pd.read_csv('footwear_v3.csv')
df = product_descriptions[['id', 'description','name']]
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['description'])
num_clusters = 4

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)


@app.route('/featureProducts', methods=['POST'])
def featured_products():
    search_history = request.json['search_history']
    keyword = ' '.join(search_history)    
    vec = vectorizer.transform([keyword])    
    cluster = kmeans.predict(vec)[0]    
    same_cluster = df[kmeans.labels_ == cluster]    
    similarity_scores = cosine_similarity(vec, X[same_cluster.index])
    sorted_indices = similarity_scores.argsort().flatten()[::-1]
    recommended = same_cluster['id'].iloc[sorted_indices[:10]]
    recommended_names = same_cluster['name'].iloc[sorted_indices[:10]]
    return jsonify(ids=recommended.tolist(), names=recommended_names.tolist())

@app.route('/recommend', methods=['POST'])
def recommend_products():
    search_history = request.json['product_name_keywords']
    keyword = ' '.join(search_history)    
    vec = vectorizer.transform([keyword])    
    cluster = kmeans.predict(vec)[0]    
    same_cluster = df[kmeans.labels_ == cluster]    
    similarity_scores = cosine_similarity(vec, X[same_cluster.index])
    sorted_indices = similarity_scores.argsort().flatten()[::-1]
    recommended = same_cluster['id'].iloc[sorted_indices[:5]]
    recommended_names = same_cluster['name'].iloc[sorted_indices[:5]]
    return jsonify(ids=recommended.tolist(), names=recommended_names.tolist())


if __name__ == '__main__':
    app.run(host='192.168.238.222')