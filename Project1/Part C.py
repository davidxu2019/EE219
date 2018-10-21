# -*- coding: utf-8 -*-

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from nltk import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer    
from nltk.tokenize.regexp import RegexpTokenizer
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer
 
import scipy
import numpy
import string

ver=2;#there are 3 different kind of stemmer

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


categories=['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','comp.windows.x','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey','sci.crypt','sci.electronics','sci.med','sci.space','misc.forsale','talk.politics.misc','talk.politics.guns','talk.politics.mideast','talk.religion.misc','alt.atheism','soc.religion.christian']

num_of_class=20
train_tmp=[]
train_data=[]
num_doc_class=[]
stop_words=text.ENGLISH_STOP_WORDS

#TFxICF and TFxIDF are same because if we see class as the document parameter, so all we need is to make all the documents in same catogory to be a document
for i in range(0,num_of_class):
	tmp_data=fetch_20newsgroups(subset='train',categories=[categories[i]],shuffle=True,random_state=42)
	train_tmp.append(tmp_data.data)
	num_doc_class.append(len(tmp_data.data))
#balance the dataset
min_num_of_class=min(num_doc_class)

#make the all documents in same class to be a document
for i in range(0,num_of_class):
	train_data.append('')
	for j in range(0,min_num_of_class):
		train_data[i] += (' '+train_tmp[i][j])

vectorizer=text.CountVectorizer(analyzer = 'word',
                             decode_error = 'strict',tokenizer=tokenizer(),lowercase=True,min_df=1,stop_words=stop_words)
train_counts=vectorizer.fit_transform(train_data)
train_transformer = TfidfTransformer()
train_tf = train_transformer.fit_transform(train_counts)

train_array=train_tf.toarray()

index=[2,3,13,19]#represent the index of 4 classes

#find top 10 significant terms
for i in range(0,4):
	print('\n'+categories[index[i]])
	sorted_array=numpy.argsort(train_array[index[i]])
	features_names=vectorizer.get_feature_names()
	significant_features=[features_names[j] for j in sorted_array[-10:]]
	significant_features=significant_features[::-1]#reverse it
	print significant_features