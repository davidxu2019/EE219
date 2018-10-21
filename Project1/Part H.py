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
from sklearn.svm import SVC  
from sklearn.metrics import *
from matplotlib import pyplot
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import numpy
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
test_data=fetch_20newsgroups(subset='test',categories=categories,shuffle=True,random_state=42)


#set stopwords to exclude the stop words
stop_words=text.ENGLISH_STOP_WORDS

vectorizer=text.CountVectorizer(analyzer = 'word',tokenizer=tokenizer(),lowercase=True,min_df=2,stop_words=stop_words)
train_counts=vectorizer.fit_transform(train_data.data)
train_transformer = TfidfTransformer(sublinear_tf=True,use_idf=True)
train_tf = train_transformer.fit_transform(train_counts)
test_counts=vectorizer.transform(test_data.data)
test_transformer = TfidfTransformer(sublinear_tf=True,use_idf=True)
test_tf = test_transformer.fit_transform(test_counts)


#get reduced matrix
k=50
lsi_model=TruncatedSVD(n_components=k,random_state=42)
nmf_model=NMF(n_components=k)
train_LSI_array=lsi_model.fit_transform(train_tf)
train_NMF_array=nmf_model.fit_transform(train_tf)
test_LSI_array=lsi_model.transform(test_tf)
test_NMF_array=nmf_model.transform(test_tf)

#train data
train_data_catorgory=[]#0 means computer techonology and 1 means recreational activity
test_data_catorgory=[]
for i in train_data.target:
	if(i<4):
		train_data_catorgory.append(0)
	else:
		train_data_catorgory.append(1)
for i in test_data.target:
	if(i<4):
		test_data_catorgory.append(0)
	else:
		test_data_catorgory.append(1)
svm_train=SVC(C=1000,decision_function_shape='ovr',kernel='rbf',random_state=42)
lr_train=LogisticRegression()
MNB_train=MultinomialNB()


#test model	
#svm
#LSI
svm_train.fit(train_LSI_array,train_data_catorgory)	
test_LSI_score = svm_train.decision_function(test_LSI_array)
test_LSI_FPR, test_LSI_TPR, LSI_thresholds = roc_curve(test_data_catorgory, test_LSI_score)
#NMF
svm_train.fit(train_NMF_array,train_data_catorgory)	
test_NMF_score = svm_train.decision_function(test_NMF_array)
test_NMF_FPR, test_NMF_TPR, NMF_thresholds = roc_curve(test_data_catorgory, test_NMF_score)

#MNB
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

#LSI
MNB_train.fit(train_LSI_array,train_data_catorgory)
test_score = MNB_train.predict_proba(test_LSI_array)
MNB_test_LSI_FPR, MNB_test_LSI_TPR, MNB_LSI_thresholds = roc_curve(test_data_catorgory, test_score[:,1])
#NMF
MNB_train.fit(train_NMF_array,train_data_catorgory)
test_score = MNB_train.predict_proba(test_NMF_array)
MNB_test_NMF_FPR, MNB_test_NMF_TPR, MNB_NMF_thresholds = roc_curve(test_data_catorgory, test_score[:,1])

#logistic regression
#LSI
lr_train.fit(train_LSI_array,train_data_catorgory)	
test_result=lr_train.predict(test_LSI_array)
LSI_precision = precision_score(test_data_catorgory, test_result)
LSI_recall = recall_score(test_data_catorgory, test_result)
LSI_confusionMatrix = confusion_matrix(test_data_catorgory, test_result)
LSI_accuracy = lr_train.score(test_LSI_array, test_data_catorgory)
test_LSI_score = lr_train.decision_function(test_LSI_array)
lr_test_LSI_FPR, lr_test_LSI_TPR, lr_LSI_thresholds = roc_curve(test_data_catorgory, test_LSI_score)
#NMF
lr_train.fit(train_NMF_array,train_data_catorgory)	
test_result=lr_train.predict(test_NMF_array)
NMF_precision = precision_score(test_data_catorgory, test_result)
NMF_recall = recall_score(test_data_catorgory, test_result)
NMF_confusionMatrix = confusion_matrix(test_data_catorgory, test_result)
NMF_accuracy = lr_train.score(test_NMF_array, test_data_catorgory)
test_NMF_score = lr_train.decision_function(test_NMF_array)
lr_test_NMF_FPR, lr_test_NMF_TPR, lr_NMF_thresholds = roc_curve(test_data_catorgory, test_NMF_score)

#show results
print("===========================================================")
print ("accuracy in LSI is ", LSI_accuracy)
print ("precision in LSI is ", LSI_precision)
print ("recall in LSI is ", LSI_recall)
print ("confusion matrix in LSI is ")
print (LSI_confusionMatrix)
print("===========================================================")
print ("accuracy in NMF is ", NMF_accuracy)
print ("precision in NMF is ", NMF_precision)
print ("recall in NMF is ", NMF_recall)
print ("confusion matrix in NMF is ")
print (NMF_confusionMatrix)   

pyplot.figure(1)
pyplot.plot(lr_test_LSI_FPR, lr_test_LSI_TPR, color='blue', lw=1, label='Logistic Regression')
pyplot.plot(test_LSI_FPR, test_LSI_TPR, color='red', lw=1, label='SVM')
pyplot.plot(MNB_test_LSI_FPR, MNB_test_LSI_TPR, color='green', lw=1, label='MNB')
pyplot.legend(loc='lower right', fontsize='small')
pyplot.title("ROC curve with LSI")
pyplot.xlabel('FPR')
pyplot.ylabel('TPR')


pyplot.figure(2)
pyplot.plot(lr_test_NMF_FPR, lr_test_NMF_TPR, color='blue', lw=1, label='Logistic Regression')
pyplot.plot(test_NMF_FPR, test_NMF_TPR, color='red', lw=1, label='SVM')
pyplot.plot(MNB_test_NMF_FPR, MNB_test_NMF_TPR, color='green', lw=1, label='MNB')
pyplot.legend(loc='lower right', fontsize='small')
pyplot.title("ROC curve with NMF")
pyplot.xlabel('FPR')
pyplot.ylabel('TPR')
pyplot.show()