# nltk.download('stopwords')
import re
import string

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from pandas import DataFrame


def get_dataframe_from_csv(file: str):
    return pd.read_csv("data/{}.csv".format(file))


def data_pre_process(data_frame_input: DataFrame, column: str):
    # drop data with missing values in the 'Review' column
    data_frame_input = data_frame_input.dropna(axis=0, subset=[column])
    # convert the relevant column to lowercase
    data_frame_input = data_frame_to_lowercase(data_frame_input, column_name=column)
    # tokenize
    words = tokenize_words(data_frame_input, column)
    # remove stopwords
    reviews_no_stopwords = remove_stopwords(words)
    # remove emoji
    filtered_reviews = remove_emoji(reviews_no_stopwords)
    # stemming
    stemmed = stemmed_words(filtered_reviews)
    data_frame_input["tokens"] = stemmed
    # rejoin
    data_frame_input["filtered_review"] = rejoin_words(stemmed)
    return data_frame_input


def data_frame_to_lowercase(data_frame_input: DataFrame, column_name: str):
    data_frame_input[column_name] = data_frame_input[column_name].str.lower()
    return data_frame_input


def tokenize_words(data_frame_input: DataFrame, column_name: str):
    words = []
    for word in data_frame_input[column_name].tolist():
        words.append(word_tokenize(word))
    return words


def remove_stopwords(word_tokens: list):
    stops = set(stopwords.words("english"))
    filtered_reviews = []
    for token in word_tokens:
        filtered_reviews.append([w for w in token if not w in stops])
    return filtered_reviews


def remove_emoji(words: list):
    new_words = []
    regex_pattern = re.compile("[" u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", re.UNICODE)
    for review in words:
        temp = []
        for word in review:
            temp.append(regex_pattern.sub(r'', word))
        new_words.append(temp)
    return new_words


def stemmed_words(filtered_reviews: list):
    stemmed = []
    ps = PorterStemmer()
    for review in filtered_reviews:
        stemmed.append([ps.stem(w) for w in review])
    return stemmed


def rejoin_words(stemmed_reviews: list):
    rejoin = []
    for review in stemmed_reviews:
        x = ",".join(review)  # join the text back together
        x = x.replace(",", " ")  # replace commas with spaces
        # remove punctuation from the reviews using the string package
        rejoin.append(x.translate(str.maketrans('', '', string.punctuation)))
    return rejoin


# data_frame = get_dataframe_from_csv("selected_app_reviews")
# data_frame_out = data_pre_process(data_frame, 'content')
# for i in data_frame_out["tokens"].tolist():
#     print(i)
# print("------------------------")
# for i in data_frame_out["filtered_review"].tolist():
#     print(i)
