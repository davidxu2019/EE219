#!/usr/bin/python

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from nltk import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer    
from nltk.tokenize.regexp import RegexpTokenizer
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import TfidfTransformer

import numpy
import string

ver=1;#there are 3 different kind of stemmer

class tokenizer(object):
	def __init__(self):
		self.tokenize=RegexpTokenizer(r'\b([A-Za-z]+)\b') #remove the punctuations
		if ver==2:
			self.stemmer = SnowballStemmer("english")         #using stemmed version of words
		elif ver==1:
			self.stemmer = LancasterStemmer()	
		else:
			self.stemmer = PorterStemmer()
	def __call__(self, doc):
		return [self.stemmer.stem(token) for token in self.tokenize.tokenize(doc)]
#retrieve the train dataset and test dataset
		
		
categories={'comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey'}

train_data=fetch_20newsgroups(subset='train',categories=categories,shuffle=True,random_state=42)

#set stopwords to exclude the stop words
stop_words=text.ENGLISH_STOP_WORDS

#extract features
vectorizer=text.CountVectorizer(analyzer = 'word',tokenizer=tokenizer(),lowercase=True,min_df=3,stop_words=stop_words)
train_counts=vectorizer.fit_transform(train_data.data)
train_transformer = TfidfTransformer(sublinear_tf=True,use_idf=True).fit(train_counts)
train_tf = train_transformer.transform(train_counts)

print("Final number of terms we extracted when min_df=3 is: %d" %(train_tf.shape[1]))

vectorizer=text.CountVectorizer(analyzer = 'word',tokenizer=tokenizer(),min_df=5,stop_words=stop_words)
train_counts=vectorizer.fit_transform(train_data.data)
train_transformer = TfidfTransformer(sublinear_tf=True,use_idf=True).fit(train_counts)
train_tf = train_transformer.transform(train_counts)
print("Final number of terms we extracted when min_df=5 is: %d" %(train_tf.shape[1]))
