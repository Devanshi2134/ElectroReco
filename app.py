
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 🔹 Sample dataset
data = {
    'ProductName': ['iPhone 13', 'Galaxy S21', 'Dell Inspiron 15', 'MacBook Air M1', 'Sony WH-1000XM4'],
    'Category': ['Smartphone', 'Smartphone', 'Laptop', 'Laptop', 'Headphones'],
    'Description': [
        'Apple smartphone with A15 chip and dual camera',
        'Samsung phone with Android and powerful zoom',
        'Windows laptop with Intel i5 and 8GB RAM',
        'Lightweight Apple laptop with M1 chip and Retina display',
        'Noise-canceling wireless headphones with long battery life'
    ],
    'Price': [799, 699, 550, 999, 349]
}

df = pd.DataFrame(data)

# 🔹 TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['Description'])

# 🔹 Recommender function
def recommend(product_name, top_n=3, category=None, min_price=None, max_price=None):
    index = df[df['ProductName'] == product_name].index[0]
    selected_vector = tfidf_matrix[index]

    filtered_df = df.copy()
    if category:
        filtered_df = filtered_df[filtered_df['Category'].str.lower() == category.lower()]
    if min_price is not None:
        filtered_df = filtered_df[filtered_df['Price'] >= min_price]
    if max_price is not None:
        filtered_df = filtered_df[filtered_df['Price'] <= max_price]

    if filtered_df.empty:
        return []

    filtered_indices = filtered_df.index.tolist()
    similarities = cosine_similarity(selected_vector, tfidf_matrix[filtered_indices]).flatten()
    sim_scores = list(zip(filtered_indices, similarities))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    return [(df.iloc[i]['ProductName'], df.iloc[i]['Category'], df.iloc[i]['Price'], score)
            for i, score in sim_scores[:top_n]]

# 🌐 Streamlit UI
st.set_page_config(page_title="ElectroReco", layout="centered")
st.title("🔌 ElectroReco: Electronics Recommender System")

product = st.selectbox("Choose a product:", df['ProductName'].tolist())

category_filter = st.selectbox("Filter by category (optional):", ["All"] + df['Category'].unique().tolist())
min_price, max_price = st.slider("Filter by price range ($):", 0, 1500, (0, 1500))

if st.button("🎯 Recommend"):
    cat = None if category_filter == "All" else category_filter
    results = recommend(product, category=cat, min_price=min_price, max_price=max_price)

    if results:
        st.subheader(f"🔍 Recommendations for **{product}**:")
        for name, cat, price, score in results:
            st.markdown(f"**➤ {name}**")  
_Category_: {cat}  
_Price_: ${price}  
_Score_: {round(score, 2)}")
    else:
        st.warning("No products match the selected filters.")
