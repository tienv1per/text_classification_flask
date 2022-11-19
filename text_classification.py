import re
import pickle
import numpy as np
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


lemmatizer = WordNetLemmatizer()
file_path = 'data/training.1600000.processed.noemoticon.csv'
# nltk.download('omw-1.4')
# nltk.download('wordnet')


def process_data(file_path):
    DATASET_COLUMNS = ["sentiment", "ids", "date", "flag", "user", "text"]
    DATASET_ENCODING = "ISO-8859-1"
    dataset = pd.read_csv(file_path, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)

    dataset = dataset[['sentiment', 'text']]
    dataset['sentiment'] = dataset['sentiment'].replace(4, 1)

    text, sentiment = list(dataset['text']), list(dataset['sentiment'])

    return text, sentiment

emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad',
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked', ':-$': 'confused', ':\\': 'annoyed',
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink',
          ';-)': 'wink', 'O:-)': 'angel', 'O*-)': 'angel', '(:-D': 'gossip', '=^.^=': 'cat'}

stopwords = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and', 'any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from',
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most', 'my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',
             's', 'same', 'she', "shes", 'should', "shouldve", 'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd", "youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']


def preprocess(textdata):
    processed_texts = []

    url_pattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    user_pattern = '@[^\s]+'
    alpha_pattern = "[^a-zA-Z0-9]"
    sequence_pattern = r"(.)\1\1+"
    seq_replace_pattern = r"\1\1"

    for tweet in textdata:
        tweet = tweet.lower()
        tweet = re.sub(url_pattern, ' URL', tweet)

        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])

        tweet = re.sub(user_pattern, ' USER', tweet)
        tweet = re.sub(alpha_pattern, " ", tweet)
        tweet = re.sub(sequence_pattern, seq_replace_pattern, tweet)

        preprocessed_words = []
        for word in tweet.split():
            if len(word) > 1 and word not in stopwords:
                word = lemmatizer.lemmatize(word)
                preprocessed_words.append(word)

        processed_texts.append(' '.join(preprocessed_words))

    return processed_texts


def prepare_data():
    text, sentiment = process_data(file_path=file_path)
    processedtext = preprocess(text)
    X_train, X_test, y_train, y_test = train_test_split(processedtext, sentiment,
                                                    test_size=0.05, random_state=0)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=500000)
    vectorizer.fit(X_train)

    X_train = vectorizer.transform(X_train)
    X_test = vectorizer.transform(X_test)

    return X_train, X_test, y_train, y_test, vectorizer


def model_evaluate(model):
    X_train, X_test, y_train, y_test, _ = prepare_data()
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))


def trainBNB():
    X_train, X_test, y_train, y_test, _ = prepare_data()
    BNBmodel = BernoulliNB(alpha=2)
    BNBmodel.fit(X_train, y_train)
    model_evaluate(BNBmodel)
    return BNBmodel


def trainSVC():
    X_train, X_test, y_train, y_test, _ = prepare_data()
    SVCmodel = LinearSVC()
    SVCmodel.fit(X_train, y_train)
    model_evaluate(SVCmodel)
    return SVCmodel


def trainLR():
    X_train, X_test, y_train, y_test, _ = prepare_data()
    LRmodel = LogisticRegression(C=2, max_iter=1000, n_jobs=-1)
    LRmodel.fit(X_train, y_train)
    model_evaluate(LRmodel)
    return LRmodel


def pipeline():
    X_train, X_test, y_train, y_test, vectorizer = prepare_data()
    BNBmodel = trainBNB()
    pipe = Pipeline([('vectorizer', vectorizer), ('bnb', BNBmodel)])
    pipe.fit(X_train, y_train)

    model_evaluate(pipe)
    with open('pipeline.pickle', 'wb') as f:
        pickle.dump(pipe, f)

    with open('pipeline.pickle', 'rb') as f:
        loaded_pipe = pickle.load(f)

    model_evaluate(loaded_pipe)
    return pipe


def predict(model, text):
    preprocessed_text = preprocess(text)
    predictions = model.predict(preprocessed_text)
    pred_to_label = {0: 'Negative', 1: 'Positive'}

    data = []
    for t, pred in zip(text, predictions):
        data.append((t, pred, pred_to_label[pred]))

    return data


if __name__ == "__main__":
    pipe = pipeline()
    text = ["I hate twitter",
            "May the Force be with you.",
            "Mr. Stark, I don't feel so good"]

    with open('pipeline.pickle', 'rb') as f:
        loaded_pipe = pickle.load(f)
    predictions = predict(loaded_pipe, text)
    print(predictions)
