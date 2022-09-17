import re
import html
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.cluster import AgglomerativeClustering

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from typing import List

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

st.title('Similarity Job Search')

with st.sidebar:
    data_percentage = st.slider("Data percentage", min_value=0.0, max_value=1.0, value=0.2)
    show_raw_data = st.checkbox('Show raw job listings data')
    keyword_search = st.text_input("Keyword search", "Developer")


TAG_RE = re.compile(r'<[^>]+>')
def remove_tags_and_unescape(text):
    s = TAG_RE.sub('', text)
    return html.unescape(s)

@st.cache
def load_job_listings(percentage: float):
    df = pd.read_xml("https://jobboerse.phoenixcontact.com/jobPublication/list.xml")
    content_columns = ["position", "introduction", "tasks", "profile"]
    for c in content_columns:
        df[c+"_cleaned"] = df[c].apply(remove_tags_and_unescape).astype('string') 
    
    df = df.head(int(df.shape[0]*percentage))
    return df

data_load_state = st.text('Loading job listings...')
data = load_job_listings(data_percentage)
data_load_state.text("Done loading job listings! (using st.cache)")

if show_raw_data:
    st.subheader('Raw data')
    st.write(data)


min3chars_word = r"(?u)\b[a-zA-ZäöüÄÖÜß][a-zA-ZäöüÄÖÜß][a-zA-ZäöüÄÖÜß]+\b"
german_stop_words = stopwords.words('german')

@st.cache
def get_cos_matrix(texts: List[str]):
  tv = TfidfVectorizer(stop_words=german_stop_words, min_df=0, token_pattern=min3chars_word)
  count_matrix = tv.fit_transform(texts)
  tfidf_array = count_matrix.toarray()

  print("\nSimilarity Scores:")
  cos_matrix = cosine_similarity(count_matrix)
  print(cos_matrix)
  return cos_matrix

@st.cache
def plot_dendrogram(model, **kwargs):
    print("Calculating linkage matrix for Dendrogram")
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    print("Plotting Dendrogram")
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

    print("Finished plotting Dendrogram")


def plot_clusters(df: pd.DataFrame, column_name: str):
  model = AgglomerativeClustering(affinity='cosine', linkage='complete', distance_threshold=0, n_clusters=None, compute_full_tree=True)
  cos_matrix = get_cos_matrix(df[column_name].array)
  model.fit(cos_matrix)

  f = plt.figure()
  f.set_figwidth(5)
  f.set_figheight(10)

  plt.title("Hierarchical Clustering Dendrogram")
  # plot the top three levels of the dendrogram
  plot_dendrogram(model, labels=df["position_cleaned"].array, truncate_mode="level", orientation="right")
  plt.xlabel("By Column: " + column_name)

  return f

fig = plot_clusters(data, "profile_cleaned")
print("Showing pyplot figure")
fig.savefig("figure.png", dpi=300)

st.image("figure.png")
