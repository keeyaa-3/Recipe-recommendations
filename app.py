import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Loading and preparing data!!! (reading dataset from Kaggle)
def load_data():
  df=pd.read_csv('Cleaned_Indian_Food_Dataset.csv')
df.dropna(subset=['TranslatedRecipeName','TranslatedIngredients', 'TranslatedInstructions'],
inplace=True)
df.reset_index(drop=True, inplace=True)
return df
df = load_data()
# Creating TF-IDF Model
def create_model(df):
  tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['TranslatedIngredients'])
return tfidf, tfidf_matrix
tfidf, tfidf_matrix = create_model(df)
# Getting Recipe recommendations
def get_recommendations(user_input, recipes_df, tfidf_model, matrix):
query_vec = tfidf_model.transform([user_input])
similarity_scores = cosine_similarity(query_vec, matrix).flatten()
top_indices = similarity_scores.argsort()[-5:][::-1]
return recipes_df.iloc[top_indices][['TranslatedRecipeName', 'TranslatedIngredients',
'TranslatedInstructions']]
# Streamlit Dashboard UI
st.set_page_config(page_title="Recipe Recommender", layout="centered")
st.title("Smart Recipe Recommender")
st.write("Enter ingredients you have or prefer, and get recipe ideas instantly!")
user_input = st.text_input("Ingredients (comma-separated):", placeholder="e.g. tomato, onion, cheese")
if user_input:
with st.spinner("Finding delicious recipes..."):
results = get_recommendations(user_input, df, tfidf, tfidf_matrix)
st.success(f"Top {len(results)} recipe recommendations:")

for i, (index, row) in enumerate(results.iterrows()):
st.markdown(f"### {i+1}. {row['TranslatedRecipeName']}")
st.markdown(f"**Ingredients:** {row['TranslatedIngredients']}")
st.markdown(f"**Instructions:** {row['TranslatedInstructions'][:1000000]}{'...' if
len(row['TranslatedInstructions']) > 1000000 else ''}")
st.markdown("---")
