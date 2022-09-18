import streamlit as st
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise  import cosine_similarity, euclidean_distances
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from typing import List
from umap import UMAP
import plotly.express as px


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

st.set_page_config(layout="wide")
st.title('Skill Similarity Job Search')

df_orig = pd.read_pickle("df_eng.pickle")

if 'skill_values' not in st.session_state:
    st.session_state['skill_values'] = {}

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


def get_tfidf_array(texts):
  min3chars_word = r"(?u)\b[a-zA-ZäöüÄÖÜß][a-zA-ZäöüÄÖÜß][a-zA-ZäöüÄÖÜß]+\b"
  german_stop_words = stopwords.words('german')
  tv = TfidfVectorizer(stop_words=german_stop_words, min_df=0, token_pattern=min3chars_word)
  count_matrix = tv.fit_transform(texts)
  tfidf_array = count_matrix.toarray()
  return tfidf_array


tfidf_array = get_tfidf_array(data["position_cleaned"].array)

umap_2d = UMAP(n_components=2, init='random', random_state=0)
proj_2d = umap_2d.fit_transform(tfidf_array)



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

def reset_skill_selection():
    st.session_state['skill_values'].clear()

col1, col2 = st.columns(2, gap="large")

with col1:

    st.write("Please select some skills you have or are interesed in to get personalized job recommondations.")
    
    s = df_bow_profile.sum().sort_values(ascending=False).copy()
    skill_search = st.text_input("Filter skills", placeholder="e.g. business")
    st.button("Reset", on_click=reset_skill_selection)
    df_skills = pd.DataFrame(s).reset_index()
    if skill_search:
        df_skills = df_skills[df_skills["index"].str.contains(skill_search, case=False)]
    
    if df_skills.shape[0] == 0:
        st.write("No skill found! Please change the search term.")

    skills = df_skills.head(20).to_numpy()
    for name, count in skills:
        if name in st.session_state['skill_values']:
            value = float(st.session_state['skill_values'][name])
        else:
            value = 0.5
        slider = st.slider(name + f" [{count}]", min_value=0.0, max_value=1.0, value=value)
        st.session_state['skill_values'][name] = slider


df_query_vec = pd.DataFrame(columns=df_bow_profile.columns)
df_query_vec.loc[0,:] = 0.5
for k,v in st.session_state['skill_values'].items():
  df_query_vec.loc[0, k] = v  



query_vec = df_query_vec.iloc[0].to_numpy().reshape((1,-1))
db_vecs = df_bow_profile.to_numpy()
distances = euclidean_distances(query_vec, db_vecs)

with col2:
    # st.write(query_vec)

    st.subheader("Job Results")

    df_matches["distances"] = distances.reshape((-1))
    df_matches = df_matches.reset_index()
    results = df_matches.sort_values(by="distances", ascending=True).head(10)
    st.write(results.loc[:, ["distances", "position", "profile_annotations_matches", "jobPublicationURL"]])


    for index, row in results.iterrows():
        st.subheader(row["position"])
        url = row["jobPublicationURL"]
        st.markdown(f"[Open job listing]({url})")



    # Map of Jobs
    st.markdown("""---""")
    st.subheader("Map of jobs")
    st.text("Explore the map of jobs to find jobs with similar descritions.")

    colors = np.zeros((df_matches.shape[0]))
    colors[results.index.values] = 1
    
    colors = [["blue", "red"][int(v)] for v in colors]

    our_color_discrete_map={
                    "unknown": "rgba(180, 180, 180, 0.24)",
                    0: "rgba(5, 192, 5, 0.74)",
                    1: "rgba(255, 112, 0, 0.74)",
                  }

    fig_2d = px.scatter(
        proj_2d, x=0, y=1,
        hover_data=[data["position"].values],
        color=colors
    )
    st.plotly_chart(fig_2d)