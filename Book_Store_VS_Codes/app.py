from flask import Flask, jsonify, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the dataset
df = pd.read_csv('cleaned_book_store_data.csv')

# Split the dataset
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Batch processing for TF-IDF
batch_size = 10000
batches = np.array_split(df, len(df) // batch_size)
tfidf = TfidfVectorizer(stop_words='english')

for i, batch in enumerate(batches):
    combined_text = batch['Book-Title'] + ' ' + batch['Book-Author'] + ' ' + batch['Publisher']
    tfidf_matrix = tfidf.fit_transform(combined_text)
    batch['tfidf_matrix'] = list(tfidf_matrix)
    print(f'Processed batch {i + 1}/{len(batches)}')

df_combined = pd.concat(batches)
indices = pd.Series(df_combined.index, index=df_combined['Book-Title']).drop_duplicates()

def recommend_books_by_author_batch(df, title, top_n=5):
    idx = indices[title]
    author = df.loc[idx, 'Book-Author']
    author_books = df[df['Book-Author'] == author]
    return author_books[['Book-Title', 'Book-Author', 'Image-URL-M']].head(top_n)

def collaborative_recommendation(df, user_id, top_n=5, batch_size=5000, similarity_threshold=0.1):
    user_ids = df['User-ID'].unique()
    num_batches = int(np.ceil(len(user_ids) / batch_size))
    user_id_batches = np.array_split(user_ids, num_batches)
    
    all_recommendations = []

    for batch_num, user_batch in enumerate(user_id_batches):
        print(f"Processing batch {batch_num + 1} of {num_batches}...")
        batch_train_df = df[df['User-ID'].isin(user_batch)]
        user_book_matrix = batch_train_df.pivot_table(index='User-ID', columns='ISBN', values='Book-Rating', aggfunc='mean').fillna(0)
        user_similarity = cosine_similarity(user_book_matrix)
        
        if user_id in user_book_matrix.index:
            target_user_index = user_book_matrix.index.get_loc(user_id)
            user_similarities = user_similarity[target_user_index]
            similar_user_indices = user_similarities.argsort()[::-1][1:]

            recommended_books = []
            similarity_scores = []

            for user_index in similar_user_indices:
                if user_similarities[user_index] >= similarity_threshold:
                    rated_by_similar_user = user_book_matrix.iloc[user_index]
                    not_rated_by_target_user = (rated_by_similar_user != 0) & (user_book_matrix.iloc[target_user_index] == 0)

                    for book in user_book_matrix.columns[not_rated_by_target_user][:top_n]:
                        recommended_books.append(book)
                        similarity_scores.append(user_similarities[user_index])

            unique_recommendations = list(set(recommended_books))
            all_recommendations.extend(zip(unique_recommendations, similarity_scores))

    recommended_books_df = pd.DataFrame(all_recommendations, columns=['ISBN', 'Similarity Score'])
    print(recommended_books_df)  # Debug print to check the recommended books and similarity scores
    recommended_books_info = df[df['ISBN'].isin(recommended_books_df['ISBN'])][['ISBN', 'Book-Title', 'Book-Author', 'Image-URL-M']].drop_duplicates()
    recommended_books_info = recommended_books_info.merge(recommended_books_df, on='ISBN', how='left')

    return recommended_books_info.head(top_n)

def hybrid_recommendation(df, user_id, book_name, top_n=5, content_weight=0.5, collab_weight=0.5):
    # 1. Get collaborative recommendations
    collaborative_recs = collaborative_recommendation(df, user_id, top_n)

    # 2. Get content-based recommendations
    content_recs = recommend_books_by_author_batch(df_combined, book_name, top_n)
    
    # 3. Combine recommendations with weights
    combined_scores = {}
    
    # Add content-based scores
    for index, row in content_recs.iterrows():
        combined_scores[row['Book-Title']] = content_weight
    
    # Add collaborative-based scores
    for index, row in collaborative_recs.iterrows():
        if row['Book-Title'] in combined_scores:
            combined_scores[row['Book-Title']] += collab_weight * row['Similarity Score']
        else:
            combined_scores[row['Book-Title']] = collab_weight * row['Similarity Score']
    
    # Sort combined recommendations by score
    sorted_combined_scores = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    top_recommendations = [item[0] for item in sorted_combined_scores[:top_n]]
    
    # Get detailed info about the top recommendations
    final_recommendations = df[df['Book-Title'].isin(top_recommendations)][['Book-Title', 'Book-Author', 'Image-URL-M']]
    return final_recommendations



@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = int(request.args.get('user_id'))
    top_n = int(request.args.get('top_n', 5))
    recommendations = collaborative_recommendation(train_df, user_id, top_n)
    print(recommendations)  # Debug print to check the final recommendations
    return jsonify(recommendations[['Book-Title', 'Book-Author', 'Image-URL-M']].to_dict(orient='records'))


def recommend_top_books(df, top_n=5):
    book_stats = df.groupby('ISBN').agg(
        avg_rating=('Book-Rating', 'mean'),
        user_count=('User-ID', 'count'),
        book_title=('Book-Title', 'first'),
        book_author=('Book-Author', 'first'),
        image_url=('Image-URL-M', 'first')
    ).reset_index()

    book_stats['total_score'] = book_stats['avg_rating'] + book_stats['user_count']
    sorted_books = book_stats.sort_values(by=['avg_rating', 'total_score'], ascending=[False, False])
    top_books = sorted_books.head(top_n)

    return top_books[['book_title', 'book_author', 'avg_rating', 'user_count', 'image_url']]

@app.route('/rating_based', methods=['GET'])
def rating_based():
    top_n = int(request.args.get('top_n', 10))
    top_books_recommended = recommend_top_books(train_df, top_n=top_n)
    response = top_books_recommended.to_dict(orient='records')
    return jsonify(response)

@app.route('/content_based', methods=['GET'])
def content_based():
    book_name = request.args.get('book_name')
    top_n = int(request.args.get('top_n', 5))
    content_based_rec = recommend_books_by_author_batch(df_combined, book_name, top_n=top_n)
    print(content_based_rec)  # Debug print to check the response
    return jsonify(content_based_rec.to_dict(orient='records'))

@app.route('/hybrid', methods=['GET'])
def hybrid():
    user_id = int(request.args.get('user_id'))
    book_name = request.args.get('book_name')
    top_n = int(request.args.get('top_n', 5))
    recommendations = hybrid_recommendation(train_df, user_id, book_name, top_n)
    print("Hybrid Recommendations:")
    print(recommendations)  # Debug print to check the final hybrid recommendations
    return jsonify(recommendations.to_dict(orient='records'))


if __name__ == '__main__':
    app.run(debug=True)
