import re
import pickle
from nltk.stem import WordNetLemmatizer
from text_classification import preprocess


lemmatizer = WordNetLemmatizer()

with open('pipeline.pickle', 'rb') as f:
    loaded_pipe = pickle.load(f)


def predict_pipeline(text):
    return predict(loaded_pipe, text)


emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad',
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked', ':-$': 'confused', ':\\': 'annoyed',
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink',
          ';-)': 'wink', 'O:-)': 'angel', 'O*-)': 'angel', '(:-D': 'gossip', '=^.^=': 'cat'}

stopwords = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between', 'both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from',
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into', 'is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most', 'my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'own', 're',
             's', 'same', 'she', "shes", 'should', "shouldve", 'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
             'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd", "youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']


def predict(model, text):
    preprocessed_text = preprocess(text)
    predictions = model.predict(preprocessed_text)
    pred_to_label = {0: 'Negative', 1: 'Positive'}

    data = []
    for t, pred in zip(text, predictions):
        data.append({'text': t, 'pred': int(pred), 'label': pred_to_label[pred]})

    return data


if __name__ == "__main__":
    text = ["I hate twitter",
            "May the Force be with you.",
            "Mr. Stark, I don't feel so good"]

    predictions = predict_pipeline(text)
    print(predictions)
