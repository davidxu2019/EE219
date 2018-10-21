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
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF

from numpy import *
from numpy import linalg as la

import string

ver=0;#there are 3 different kind of stemmer

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

vectorizer=text.CountVectorizer(analyzer = 'word',tokenizer=tokenizer(),lowercase=True,min_df=3,stop_words=stop_words)
train_counts=vectorizer.fit_transform(train_data.data)
train_transformer = TfidfTransformer(norm='l2',use_idf=True,sublinear_tf=True)
train_tf = train_transformer.fit_transform(train_counts)

#reduce dimension
k=50
lsi_model=TruncatedSVD(n_components=k)
train_LSI_array=lsi_model.fit_transform(train_tf)
nmf_model=NMF(n_components=k)
train_NMF_array=nmf_model.fit_transform(train_tf)

#show results
print(train_LSI_array.shape)
print(train_NMF_array.shape)