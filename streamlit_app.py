import streamlit as st
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise  import cosine_similarity, euclidean_distances
from sklearn.feature_extraction.text import CountVectorizer


from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from typing import List


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

st.set_page_config(layout="wide")
st.title('Skill Similarity Job Search')

df_orig = pd.read_pickle("df_eng.pickle")

with st.sidebar:
    st.header("Configuration")
    st.text_input(label="Data URL", value="https://jobboerse.phoenixcontact.com/jobPublication/list.xml", disabled=True)
    data_count = st.slider("Jobcount", min_value=0, max_value=df_orig.shape[0], value=50)
    show_raw_data = st.checkbox('Show raw data for debugging')

def load_job_listings(count: int):
    df = df_orig.sample(count, random_state=42)
    return df

data_load_state = st.text('Loading job listings...')
data = load_job_listings(int(data_count))
data_load_state.text(f"Done loading {data.shape[0]} job listings!")

if show_raw_data:
    st.subheader('Raw data')
    st.write(data.columns)
    st.write(data)

english_stopwords = stopwords.words('english')
cv = CountVectorizer(stop_words=english_stopwords, analyzer=lambda x: x, binary=True)
count_matrix = cv.fit_transform(data["profile_annotations_matches"].array)
count_array = count_matrix.toarray()
df_bow_profile = pd.DataFrame(data=count_array,columns = cv.get_feature_names())

if show_raw_data:
    st.write(df_bow_profile)

def plot_skills_barh(df):
  f = plt.figure()
  f.set_figwidth(5)
  f.set_figheight(23)

  df_skills = df.sum().sort_values()
  df_skills.plot.barh(log=True)


  st.pyplot(f)

# plot_skills_barh(df_bow_profile)

df_skills = df_bow_profile.sum().sort_values(ascending=False)
# st.write(df_skills)


#
#  Question Answering
#

df_matches = data.copy()

query_vec = {
    # "programming": 0.8,
    # "computer science": 1.0,
    # "english": 0.9
}

col1, col2 = st.columns(2, gap="large")

with col1:

    st.write("Please select some skills you have or are interesed in to get personalized job recommondations.")
    
    s = df_bow_profile.sum().sort_values(ascending=False).copy()
    skill_search = st.text_input("Filter skills", placeholder="e.g. business")
    df_skills = pd.DataFrame(s).reset_index()
    if skill_search:
        df_skills = df_skills[df_skills["index"].str.contains(skill_search, case=False)]
    
    if df_skills.shape[0] == 0:
        st.write("No skill found! Please change the search term.")

    skills = df_skills.head(20).to_numpy()
    for name, count in skills:
        slider = st.slider(name + f" [{count}]", min_value=0.0, max_value=1.0, value=0.5)
        query_vec[name] = slider


df_query_vec = pd.DataFrame(columns=df_bow_profile.columns)
df_query_vec.loc[0,:] = 0.5
for k,v in query_vec.items():
  df_query_vec.loc[0, k] = v  



query_vec = df_query_vec.iloc[0].to_numpy().reshape((1,-1))
db_vecs = df_bow_profile.to_numpy()
distances = euclidean_distances(query_vec, db_vecs)

with col2:
    # st.write(query_vec)

    st.subheader("Job Results")

    df_matches["distances"] = distances.reshape((-1))
    results = df_matches.sort_values(by="distances", ascending=True).head(20)
    st.write(results.loc[:, ["distances", "position", "profile_annotations_matches", "jobPublicationURL"]])


    for index, row in results.iterrows():
        st.subheader(row["position"])
        url = row["jobPublicationURL"]
        st.markdown(f"[Open job listing]({url})")
