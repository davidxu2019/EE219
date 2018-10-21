#!/usr/bin/python
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
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.svm import SVC  
from sklearn.svm import LinearSVC  
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import *
from matplotlib import pyplot
from sklearn.naive_bayes import MultinomialNB
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

categories=['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']

train_data=fetch_20newsgroups(subset='train',categories=categories,shuffle=True,random_state=42)
test_data=fetch_20newsgroups(subset='test',categories=categories,shuffle=True,random_state=42)


#set stopwords to exclude the stop words
stop_words=text.ENGLISH_STOP_WORDS

vectorizer=text.CountVectorizer(analyzer='word',decode_error='ignore',tokenizer=tokenizer(),min_df=2,stop_words=stop_words)
train_counts=vectorizer.fit_transform(train_data.data)
train_transformer = TfidfTransformer()
train_tf = train_transformer.fit_transform(train_counts)
test_counts=vectorizer.transform(test_data.data)
test_counts_array=test_counts.toarray()
test_transformer = TfidfTransformer()
test_tf = test_transformer.fit_transform(test_counts_array)
test_tf_array=test_tf.toarray()



#get reduced matrix
k=50
lsi_model=TruncatedSVD(n_components=k,random_state=42)
nmf_model=NMF(n_components=k)
train_LSI_array=lsi_model.fit_transform(train_tf)
train_NMF_array=nmf_model.fit_transform(train_tf)
test_LSI_array=lsi_model.transform(test_tf_array)
test_NMF_array=nmf_model.transform(test_tf_array)

svm_train_original=SVC(C=100,kernel='linear')

for i in range(0,2):
	print("===========================================================")
	if(i==0):
		svm_train=OneVsOneClassifier(svm_train_original)
		print("svm(One vs One):")
	else:
		svm_train=OneVsRestClassifier(svm_train_original)
		print("svm(One vs Rest):")
	#LSI
	svm_train.fit(train_LSI_array,train_data.target)	
	test_result=svm_train.predict(test_LSI_array)
	LSI_precision = precision_score(test_data.target, test_result,average='weighted')
	LSI_recall = recall_score(test_data.target, test_result,average='weighted')
	LSI_confusionMatrix = confusion_matrix(test_data.target, test_result)
	LSI_accuracy = svm_train.score(test_LSI_array, test_data.target)


	#NMF
	svm_train.fit(train_NMF_array,train_data.target)	
	test_result=svm_train.predict(test_NMF_array)
	NMF_precision = precision_score(test_data.target, test_result,average='weighted')
	NMF_recall = recall_score(test_data.target, test_result,average='weighted')
	NMF_confusionMatrix = confusion_matrix(test_data.target, test_result)
	NMF_accuracy = svm_train.score(test_NMF_array, test_data.target)
	
	print ("accuracy with LSI is ", LSI_accuracy)
	print ("precision with LSI is ", LSI_precision)
	print ("recall with LSI is ", LSI_recall)
	print ("confusion matrix with LSI is ")  
	print (LSI_confusionMatrix) 
	
	print ("accuracy with NMF is ", NMF_accuracy)
	print ("precision with NMF is ", NMF_precision)
	print ("recall with NMF is ", NMF_recall)
	print ("confusion matrix with NMF is ")   
	print (NMF_confusionMatrix)

#train data by using multinomia Naive Bayes
train_max_para=numpy.amax(train_LSI_array)
train_min_para=numpy.amin(train_LSI_array)
test_max_para=numpy.amax(test_LSI_array)
test_min_para=numpy.amin(test_LSI_array)
for i in range(0,len(train_LSI_array)):
	for j in range(0,k):
		train_LSI_array[i,j]=float((train_LSI_array[i,j]-train_min_para)/(train_max_para-train_min_para))
		
for i in range(0,len(test_LSI_array)):
	for j in range(0,k):
		test_LSI_array[i,j]=float((test_LSI_array[i,j]-test_min_para)/(test_max_para-test_min_para))
		
		
#test model One vs One
for i in range(0,2):
	MNB_train_original=MultinomialNB()
	print("===========================================================")
	if(i==0):
		MNB_train=OneVsOneClassifier(MNB_train_original)
		print("MNB(One vs One):")
	else:
		MNB_train=OneVsRestClassifier(MNB_train_original)
		print("MNB(One vs Rest):")
	#LSI
	MNB_train.fit(train_LSI_array,train_data.target)
	test_result=MNB_train.predict(test_LSI_array)
	LSI_precision = precision_score(test_data.target, test_result,average='weighted')
	LSI_recall = recall_score(test_data.target, test_result,average='weighted')
	LSI_confusionMatrix = confusion_matrix(test_data.target, test_result)
	LSI_accuracy = MNB_train.score(test_LSI_array, test_data.target)

	#NMF
	MNB_train.fit(train_NMF_array,train_data.target)
	test_result=MNB_train.predict(test_NMF_array)
	NMF_precision = precision_score(test_data.target, test_result,average='weighted')
	NMF_recall = recall_score(test_data.target, test_result,average='weighted')
	NMF_confusionMatrix = confusion_matrix(test_data.target, test_result)
	NMF_accuracy = MNB_train.score(test_NMF_array, test_data.target)

	#show results
	
	  
	print ("accuracy with LSI is ", LSI_accuracy)
	print ("precision with LSI is ", LSI_precision)
	print ("recall with LSI is ", LSI_recall)
	print ("confusion matrix with LSI is ") 
	print (LSI_confusionMatrix) 

	    
	print ("accuracy with NMF is ", NMF_accuracy)
	print ("precision with NMF is ", NMF_precision)
	print ("recall with NMF is ", NMF_recall)
	print ("confusion matrix with NMF is ")
	print (NMF_confusionMatrix)
	