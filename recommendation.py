import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd


product_descriptions = pd.read_csv('footwear_v3.csv')

df = product_descriptions[['id', 'description','name']]
# Sample user search history
search_history = [ "nike", "sandal"]

# Step 1: Data Preprocessing
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['description'])

# Step 2: K-means Clustering
num_clusters = 4  # Choose the number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

def recommend_products(search_history):
    keyword = ' '.join(search_history)    
    vec = vectorizer.transform([keyword])    
    cluster = kmeans.predict(vec)[0]    
    same_cluster = df[kmeans.labels_ == cluster]    
    similarity_scores = cosine_similarity(vec, X[same_cluster.index])
    sorted_indices = similarity_scores.argsort().flatten()[::-1]
    recommended = same_cluster['id'].iloc[sorted_indices[:10]]
    recommended_names = same_cluster['name'].iloc[sorted_indices[:10]]
    print(recommended_names)
    print("-----------------")
    return recommended

print(recommend_products(search_history))