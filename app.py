from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "https://peer-connect-chat.vercel.app"])

fileObject = open("project_list_dict.pkl", "rb")
df_dict = pickle.load(fileObject)

df = pd.DataFrame(df_dict)
# with open("project_list.pkl", "rb") as f:
# df = pickle.load(f)
cv = CountVectorizer(stop_words='english') 
vectors = cv.fit_transform(df['combined_text'])

def recommend_projects(project_title, min_similarity=0.35, high_similarity_threshold=0.7):
    query_vec = cv.transform([project_title])
    similarity = cosine_similarity(query_vec, vectors).flatten()
    filtered_indices = [
        (i, score) for i, score in list(enumerate(similarity))
        if score>min_similarity
    ]
    project_list = sorted(filtered_indices, reverse=True,key=lambda x:x[1])
    recommendations = []
    for i in project_list:
        similarity_tag = "high" if i[1] >= high_similarity_threshold else "moderate"
        project = df.iloc[i[0]]
        recommendations.append({
            "title": project.title,
            "description": project.description,
            "tags": project.tags,
            "tech_stack": project.tech_stack,
            "domain": project.domain,
            "createdAt": project.created_date,
            "difficulty": project.difficulty_level,
            "similarity_tag": similarity_tag,
        })
    return recommendations

@app.route('/getProjects', methods=['GET'])
def getProjects():
    try:
        limited_projects = df.head(100)
        return jsonify(limited_projects.to_dict(orient="records"))
    except Exception as e:
        return jsonify({'message': 'Something went wrong while getting projects'})
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
        return jsonify({'message': 'Invalid JSON', 'message': str(e)}), 400

# Health check route
@app.route('/')
def home():
    return "✅ Project Recommendation API is running."

if __name__ == '__main__':
    app.run(debug=True)
