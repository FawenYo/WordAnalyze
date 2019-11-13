#from pycorenlp import StanfordCoreNLP

import os
import pandas as pd
import numpy as np
import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Bidirectional
from keras.preprocessing import sequence
from keras.layers import Dropout
import h5py
import utility_functions as uf
from keras.models import model_from_json
from keras.models import load_model
from flask import Flask, url_for, request
import json
from nltk.tokenize import RegexpTokenizer
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import jieba
import matplotlib.pyplot as plt

# Set up File Path
text_select = 'Trump'
text_path = 'C:/Users/MyStyle/Desktop/WordAnalyze/Text/{}.txt'.format(text_select)
log_path = 'C:/Users/MyStyle/Desktop/WordAnalyze/Result/Log_{}.txt'.format(text_select)

# Initialize
with open(log_path, "a", encoding = 'utf8') as myfile:
        a = "test"
        myfile.write(str(a))
os.remove(log_path)
with open(log_path, "a", encoding = 'utf8') as myfile:
        a = "分析文本 :     {}\n\n".format(text_select)
        myfile.write(str(a))

weight_path = 'C:/Users/MyStyle/Desktop/WordAnalyze/Sent/model/best_model.hdf5'
prd_model = load_model(weight_path)
prd_model.summary()

# Load the best model that is saved
loaded_model = load_model(weight_path)
word_idx = json.load(open("C:/Users/MyStyle/Desktop/WordAnalyze/Sent/Data/word_idx.txt"))

def get_sentiment_DL(prd_model, text_data, word_idx):

    #data = "Pass the salt"

    live_list = []
    batchSize = len(text_data)
    live_list_np = np.zeros((56,batchSize))
    for index, row in text_data.iterrows():
        #print (index)
        text_data_sample = text_data['text'][index]

        # split the sentence into its words and remove any punctuations.
        tokenizer = RegexpTokenizer(r'\w+')
        text_data_list = tokenizer.tokenize(text_data_sample)

        #text_data_list = text_data_sample.split()


        labels = np.array(['1','2','3','4','5','6','7','8','9','10'], dtype = "int")
        #word_idx['I']
        # get index for the live stage
        data_index = np.array([word_idx[word.lower()] if word.lower() in word_idx else 0 for word in text_data_list])
        data_index_np = np.array(data_index)

        # padded with zeros of length 56 i.e maximum length
        padded_array = np.zeros(56)
        padded_array[:data_index_np.shape[0]] = data_index_np
        data_index_np_pad = padded_array.astype(int)


        live_list.append(data_index_np_pad)

    live_list_np = np.asarray(live_list)
    score = prd_model.predict(live_list_np, batch_size=batchSize, verbose=0)
    single_score = np.round(np.dot(score, labels)/10,decimals=2)

    score_all  = []
    for each_score in score:

        top_3_index = np.argsort(each_score)[-3:]
        top_3_scores = each_score[top_3_index]
        top_3_weights = top_3_scores/np.sum(top_3_scores)
        single_score_dot = np.round(np.dot(top_3_index, top_3_weights)/10, decimals = 2)
        score_all.append(single_score_dot)

    text_data['Sentiment_Score'] = pd.DataFrame(score_all)

    return text_data

def live_test(trained_model, data, word_idx):
    live_list = []
    live_list_np = np.zeros((56,1))
    # split the sentence into its words and remove any punctuations.
    tokenizer = RegexpTokenizer(r'\w+')
    data_sample_list = tokenizer.tokenize(data)
    labels = np.array(['1','2','3','4','5','6','7','8','9','10'], dtype = "int")
    # get index for the live stage
    data_index = np.array([word_idx[word.lower()] if word.lower() in word_idx else 0 for word in data_sample_list])
    data_index_np = np.array(data_index)
    # padded with zeros of length 56 i.e maximum length
    padded_array = np.zeros(56)
    if data_index_np.shape[0] <= 56:
        padded_array[:data_index_np.shape[0]] = data_index_np
        data_index_np_pad = padded_array.astype(int)
        live_list.append(data_index_np_pad)
        live_list_np = np.asarray(live_list)
        # get score from the model
        score = loaded_model.predict(live_list_np, batch_size=1, verbose=0)
        single_score = np.round(np.argmax(score)/10, decimals=2) # maximum of the array i.e single band
        # weighted score of top 3 bands
        top_3_index = np.argsort(score)[0][-3:]
        top_3_scores = score[0][top_3_index]
        top_3_weights = top_3_scores/np.sum(top_3_scores)
        single_score_dot = np.round(np.dot(top_3_index, top_3_weights)/10, decimals = 2)
    else :
        single_score = -2
        single_score_dot = -2
    return single_score_dot, single_score

f = open(text_path, encoding = 'utf8')
sample = f.read()
data_sample = sent_tokenize(sample)
pos_nltk = 0
neu_nltk = 0
nag_nltk = 0
pos = 0
neu = 0
nag = 0

sid = SentimentIntensityAnalyzer()
for i in range (0, len(data_sample)-1):
    # NLTK Model
    ss = sid.polarity_scores(data_sample[i])
    if ss['compound']>=0.3:
        sent_analyze = "正面"
        pos_nltk = pos_nltk + 1
    elif -0.3 < ss['compound'] < 0.3:
        sent_analyze = "中立"
        neu_nltk = neu_nltk + 1
    else :
        sent_analyze = "負面"
        nag_nltk = nag_nltk + 1

    # LSTM Model
    result = live_test(loaded_model, data_sample[i], word_idx)
    if result[0] >= 0.6:
        result_sent = "正面"
        pos = pos + 1
    elif 0.3 < result[0] < 0.6 :
        result_sent = "中立"
        neu = neu + 1
    elif result[0] == -2 :
        result_sent = "無法分割"
    else :
        result_sent = "負面"
        nag = nag + 1

    with open(log_path, "a", encoding = 'utf8') as myfile:
        a = "NLTK 情緒模型 : {} , LSTM 情緒模型 : {}    {}\n".format(sent_analyze, result_sent, data_sample[i])
        myfile.write(str(a))

with open(log_path, "a", encoding = 'utf8') as myfile:
        a = "\n\n\n總計 : \n    NLTK _ 正面:{}次，中立:{}次，負面:{}次\n    LSTM _ 正面:{}次，中立:{}次，負面:{}次".format(pos_nltk,neu_nltk,nag_nltk,pos,neu,nag)
        myfile.write(str(a))
print("File Saved !")

# Dispersion Plot
words = ["正面","中立","負面"]
plt_title = "{} Lexical Dispersion Plot".format(text_select)

def dispersion_plot(text, words, ignore_case=False, title=plt_title):
    try:
        from matplotlib import pylab
    except ImportError:
        raise ValueError(
            'The plot function requires matplotlib to be installed.'
            'See http://matplotlib.org/'
        )

    text = list(text)
    words.reverse()

    if ignore_case:
        words_to_comp = list(map(str.lower, words))
        text_to_comp = list(map(str.lower, text))
    else:
        words_to_comp = words
        text_to_comp = text

    points = [
        (x, y)
        for x in range(len(text_to_comp))
        for y in range(len(words_to_comp))
        if text_to_comp[x] == words_to_comp[y]
    ]
    if points:
        x, y = list(zip(*points))
    else:
        x = y = ()
    pylab.plot(x, y, "b|", scalex=0.1)
    pylab.yticks(list(range(len(words))), words, color="b")
    pylab.ylim(-1, len(words))
    pylab.title(title)
    pylab.xlabel("Word Offset")
    pylab.show()

file_1=open(log_path, encoding='utf-8').read()
seg_list=nltk.text.Text(jieba.lcut(file_1))

plt.figure(figsize=(10, 5)) 
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
dispersion_plot(seg_list, words, False, plt_title)
