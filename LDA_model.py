import matplotlib.pyplot as plt

from process import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA


def plot_top_words(model, feature_names, n_top_words, title, row_number: int, column_number: int):
    fig, axes = plt.subplots(row_number, column_number, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx + 1}',
                     fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()


def lda_process(csv_file_name: str, data_column_name: str, number_of_topics=6,
                title="LAD Model Result", max_words=500, n_top_words=10):
    data_frame = get_dataframe_from_csv(csv_file_name)
    data_frame_processed = data_pre_process(data_frame, data_column_name)
    tf_vectorizer = CountVectorizer(max_features=max_words, stop_words="english")
    tf = tf_vectorizer.fit_transform(data_frame_processed["filtered_review"])

    lda = LDA(n_components=number_of_topics, max_iter=5, learning_method="online")
    lda.fit(tf)

    i = number_of_topics % 3
    j = number_of_topics // 3
    if i != 0:
        j += 1

    plot_top_words(lda, tf_vectorizer.get_feature_names(), n_top_words, title, j, 3)


def lda_process_by_data_frame(data_frame_input: DataFrame, column_name: str, number_of_topics=6,
                title="LAD Model Result", max_words=500, n_top_words=10):
    tf_vectorizer = CountVectorizer(max_features=max_words, stop_words="english")
    tf = tf_vectorizer.fit_transform(data_frame_input[column_name])
    lda = LDA(n_components=number_of_topics, max_iter=5, learning_method="online")
    lda.fit(tf)

    i = number_of_topics % 3
    j = number_of_topics // 3
    if i != 0:
        j += 1

    plot_top_words(lda, tf_vectorizer.get_feature_names(), n_top_words, title, j, 3)


lda_process("selected_app_reviews", "content")
