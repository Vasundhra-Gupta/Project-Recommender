from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from flask_cors import CORS



app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])

fileObject = open("project_list_dict.pkl", "rb")
df_dict = pickle.load(fileObject)
df = pd.DataFrame(df_dict)
# with open("project_list.pkl", "rb") as f:
# df = pickle.load(f)
cv = CountVectorizer(stop_words='english') 
vectors = cv.fit_transform(df['combined_text'])

def recommend_projects(project_title):
    query_vec = cv.transform([project_title])
    similarity = cosine_similarity(query_vec, vectors).flatten()
    project_list = sorted(list(enumerate(similarity)),reverse=True,key=lambda x:x[1])[1:11]
    recommendations = []
    for i in project_list:
        recommendations.append({
            "title": df.iloc[i[0]].title,
            "description": df.iloc[i[0]].description,
        })
    return recommendations

# API route to get recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        print("data recieved: ", data)
        title = data.get('title')
        if not title:
            return jsonify({'error': 'No title provided'}), 400

        results = recommend_projects(title)
        return jsonify(results) 
    except Exception as e:
        return jsonify({'error': 'Invalid JSON', 'message': str(e)}), 400

# Health check route
@app.route('/')
def home():
    return "✅ Project Recommendation API is running."

if __name__ == '__main__':
    app.run(debug=True)
