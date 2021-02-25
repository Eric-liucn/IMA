# latent Dirichlet allocation on the reviews data

# on first time run
# !pip install --user scikit-learn
# use the dataframe we made earlier!!!!

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

max_words = 5000 # use only the top 500 words
k = 3 # set number of topics as 10
n_top_words = 8 # print the top 20 words for each topic

# helper function to plot topics
# see Grisel et al.
# https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html
def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(1, 3, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx +1}',
                     fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()

from process import *
df = data_pre_process(get_dataframe_from_csv("selected_app_info"), "description")

# vectorise the data into word counts
tf_vectorizer = CountVectorizer(max_features=max_words, stop_words='english')
tf = tf_vectorizer.fit_transform(df['filtered_review'])

# fit LDA - we'll cover online learning later in the module
lda = LDA(n_components=k, max_iter=5, learning_method='online')
lda.fit(tf)

# get the list of words (feature names)
tf_feature_names = tf_vectorizer.get_feature_names()

# print the top words per topic
plot_top_words(lda, tf_feature_names, n_top_words, 'Topics in LDA model')