from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from os import path
from PIL import Image
import urllib
import requests
import numpy as np

# Find where the nltk.file is 
import nltk
print(nltk.__file__)

## sample text
sample = gutenberg.raw("bible-kjv.txt")
tok = word_tokenize(sample)

stopwords = nltk.corpus.stopwords.words('english')
newStopWords = [',','.',':',';','?','And','I']
stopwords.extend(newStopWords)
filtered_sentence = [w for w in tok if not w in stopwords] 
filtered_sentence = [] 
for w in tok: 
    if w not in stopwords: 
        filtered_sentence.append(w) 
mytext = nltk.Text(filtered_sentence)
filter_dist = nltk.FreqDist(filtered_sentence)

## 詞彙多樣性 (相異單詞數量/總單詞數量)
def lexical_diversity(text):
    return len(set(text)) / len(text)
data1 = lexical_diversity(filtered_sentence)
print("詞彙多樣性 (相異單詞數量/總單詞數量) : " , data1)

## 詞彙分布圖
mytext.dispersion_plot(["god","lord","satan","devil"])


## 產生文字雲
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# get data directory (using getcwd() is needed to support running example in generated IPython notebook)
d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
pic_mask = np.array(Image.open(path.join(d, "test1.png")))

string = ''
for w in filtered_sentence:
    string+= w + ' '
    
wordcloud = WordCloud(font_path=r'times.ttf', mask = pic_mask).generate(string)
plt.figure( figsize=(100,80), facecolor='k' ,frameon=False)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

filter_dist.plot(20)

##print("在古騰堡聖經中")
##
##print()
##
##print('god出現次數:', lowered_tok["god"])
##
##print("lord出現次數:")
##print(lowered_tok["lord"])
##
##print("satan出現次數:")
##print(lowered_tok["satan"])
##
##print("devil出現次數:")
##print(lowered_tok["devil"])