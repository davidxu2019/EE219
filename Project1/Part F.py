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
import numpy
from sklearn.cross_validation import KFold
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

categories={'comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey'}

train_data=fetch_20newsgroups(subset='train',categories=categories,shuffle=True,random_state=42)
test_data=fetch_20newsgroups(subset='test',categories=categories,shuffle=True,random_state=42)


#set stopwords to exclude the stop words
stop_words=text.ENGLISH_STOP_WORDS

vectorizer=text.CountVectorizer(analyzer = 'word',tokenizer=tokenizer(),lowercase=True,min_df=2,stop_words=stop_words)
train_counts=vectorizer.fit_transform(train_data.data)
train_transformer = TfidfTransformer(sublinear_tf=True)
train_tf = train_transformer.fit_transform(train_counts)
test_counts=vectorizer.transform(test_data.data)
test_transformer = TfidfTransformer(sublinear_tf=True)
test_tf = test_transformer.fit_transform(test_counts)


#get reduced matrix
k=50
lsi_model=TruncatedSVD(n_components=k,random_state=42)
nmf_model=NMF(n_components=k)
train_LSI_array=lsi_model.fit_transform(train_tf)
train_NMF_array=nmf_model.fit_transform(train_tf)
test_LSI_array=lsi_model.transform(test_tf)
test_NMF_array=nmf_model.transform(test_tf)

#real class array
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
		
#store the training and test dataset	
all_LSI_data=[]
all_NMF_data=[]
all_real_catorgory=[]

all_LSI_data=numpy.append(train_LSI_array,test_LSI_array,axis=0)
all_NMF_data=numpy.append(train_NMF_array,test_NMF_array,axis=0)
all_real_catorgory=numpy.append(train_data_catorgory,test_data_catorgory,axis=0)


num_folds=5
max_LSI_acc=0
max_NMF_acc=0

#find the best parameter of LSI and NMF
for i in range(-3,4):
	k=10**i
	svm_train=SVC(C=k,decision_function_shape='ovr',kernel='rbf',random_state=42)
	kf=KFold(len(all_LSI_data),n_folds=num_folds, shuffle=True)

	test_LSI_score=0
	test_NMF_score=0
	#test model	
	for train_index, test_index in kf:
		#LSI
		svm_train.fit(all_LSI_data[train_index],all_real_catorgory[train_index])
		LSI_accuracy = svm_train.score(all_LSI_data[test_index], all_real_catorgory[test_index])
	

		#NMF
		svm_train.fit(all_NMF_data[train_index],all_real_catorgory[train_index])	
		NMF_accuracy = svm_train.score(all_NMF_data[test_index], all_real_catorgory[test_index])

	test_LSI_score/=num_folds
	test_NMF_score/=num_folds
	if(LSI_accuracy>max_LSI_acc):
		best_LSI_penalty=k	
		max_LSI_acc=LSI_accuracy
	if(NMF_accuracy>max_NMF_acc):
		best_NMF_penalty=k
		max_NMF_acc=NMF_accuracy
		
	print(LSI_accuracy)
	print(NMF_accuracy)
		
		
 
#LSI
svm_train=SVC(C=best_LSI_penalty,decision_function_shape='ovr',kernel='rbf',random_state=42)
svm_train.fit(train_LSI_array,train_data_catorgory)
test_result=svm_train.predict(test_LSI_array)
LSI_precision = precision_score(test_data_catorgory, test_result)
LSI_recall = recall_score(test_data_catorgory, test_result)
LSI_confusionMatrix = confusion_matrix(test_data_catorgory, test_result)
test_LSI_score += svm_train.decision_function(test_LSI_array)
test_LSI_FPR, test_LSI_TPR, LSI_thresholds = roc_curve(test_data_catorgory, test_LSI_score)


#NMF
svm_train=SVC(C=best_NMF_penalty,decision_function_shape='ovr',kernel='rbf',random_state=42)
svm_train.fit(train_NMF_array,train_data_catorgory)
test_result=svm_train.predict(test_NMF_array)
NMF_precision = precision_score(test_data_catorgory, test_result)
NMF_recall = recall_score(test_data_catorgory, test_result)
NMF_confusionMatrix = confusion_matrix(test_data_catorgory, test_result)
test_NMF_score += svm_train.decision_function(test_NMF_array)
test_NMF_FPR, test_NMF_TPR, NMF_thresholds = roc_curve(test_data_catorgory, test_NMF_score)

	#show results
print("===========================================================")
print("best LSI penalty coefficient is ",best_LSI_penalty)
print("accuracy in LSI is ", max_LSI_acc)
print("precision in LSI is ",LSI_precision)
print("recall in LSI is ",LSI_recall)
print("confusion matrix in LSI is ")
print(LSI_confusionMatrix) 

print("===========================================================")
print("best NMF penalty coefficient is ",best_NMF_penalty)
print("accuracy in NMF is ", max_NMF_acc)
print("precision in NMF is ",NMF_precision)
print("recall in NMF is ",NMF_recall)
print("confusion matrix in NMF is ")
print(NMF_confusionMatrix)   





