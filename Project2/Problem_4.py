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
from sklearn import preprocessing
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC  
from sklearn.metrics import *
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans

import numpy 
import string


ver=2
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

def plotscatter(kmean_result, tfidf_matrix, center):
		pyplot.figure()
		for i in range(0,len(kmean_result)):
			pyplot.scatter(tfidf_matrix[i, 0], tfidf_matrix[i, 1], c=color[kmean_result[i]])
		for i in range(0,center.shape[0]):
			pyplot.scatter(center[i,0], center[i,1], marker='^', s=100, linewidths=5, color='k', alpha=0.6)
		
def dimreduce(matrix, k, method):
	if(method=='lsi'):
		lsi_model = TruncatedSVD(n_components=k,random_state=42)
		result = lsi_model.fit_transform(matrix)
		return result
	elif(method=='nmf'):
		nmf_model = NMF(n_components=k)
		result = nmf_model.fit_transform(matrix)
		return result
		
def matrixlog(matrix):
	
	result=FunctionTransformer(numpy.log1p).transform(matrix)			
	return result
	
def matrixnormal(matrix):
	#for i in range(matrix.shape[1]):
	#	matrix[:,i]=preprocessing.scale(matrix[:,i],with_mean=False)
	result=preprocessing.scale(matrix,with_mean=False)
	return result

def kmeanprocess(reduced_matrix, tfidf_matrix, method, n_cluster, actual_tag):
	kmean = KMeans(n_clusters=n_cluster, random_state=42)
	train_kmean = kmean.fit_predict(reduced_matrix)
	kmeans_tag=kmean.labels_
	print("============================================================================")
	print ("\nFull matrix")
	print("Homogeneity: %0.3f" % homogeneity_score(actual_tag, kmeans_tag))
	print("Completeness: %0.3f" % completeness_score(actual_tag, kmeans_tag))
	print("V-measure: %0.3f" % v_measure_score(actual_tag, kmeans_tag))
	print("Adjusted Rand score: %.3f" % adjusted_rand_score(actual_tag, kmeans_tag))
	print("Adjusted Mutual Info Score: %.3f" % adjusted_mutual_info_score(actual_tag, kmeans_tag))
	print("Confusion matrix: ")
	print(confusion_matrix(actual_tag, kmeans_tag))
	train_2d=dimreduce(train_tf, 2, method)
	train_2d_kmean = kmean.fit_predict(train_2d)
	center = kmean.cluster_centers_
	plotscatter(train_kmean, train_2d,center)
	

categories=['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']
color=['red','yellow']


train_data=fetch_20newsgroups(subset='train',categories=categories,shuffle=True,random_state=42)


#set stopwords to exclude the stop words
stop_words=text.ENGLISH_STOP_WORDS
vectorizer=text.CountVectorizer(analyzer = 'word',lowercase=True,tokenizer=tokenizer(),min_df=3,stop_words=stop_words)
train_counts=vectorizer.fit_transform(train_data.data)
train_transformer = TfidfTransformer(sublinear_tf=True,use_idf=True)
train_tf = train_transformer.fit_transform(train_counts)

#get reduced matrix
k_lsi=2
k_nmf=2
train_LSI_array = dimreduce(train_tf, k_lsi, 'lsi')
train_NMF_array = dimreduce(train_tf, k_nmf, 'nmf')

#attach actual label to the document
actual_tag=[]#0 means computer techonology and 1 means recreational activity
actual_tag_inv = []
for i in train_data.target:
	if(i<4):
		actual_tag.append(0)
		actual_tag_inv.append(1)
	else:
		actual_tag.append(1)
		actual_tag_inv.append(0)

#===============================Part a========================================#
kmeanprocess(train_LSI_array, train_tf, 'lsi', 2, actual_tag)
kmeanprocess(train_NMF_array, train_tf, 'nmf', 2, actual_tag)


#===============================LSI Normalize========================================#
#normalize tfidf matrix
LSI_array=train_LSI_array
LSI_normal=matrixnormal(LSI_array)
kmeanprocess(LSI_normal, train_tf, 'lsi', 2, actual_tag)
#===============================NMF Normalize========================================#
NMF_array=train_NMF_array
NMF_normal=matrixnormal(NMF_array)
kmeanprocess(NMF_normal, train_tf, 'nmf', 2, actual_tag)

#===============================NMF Non-linear Transformation========================================#
NMF_array=train_NMF_array
NMF_log=matrixlog(NMF_array)
kmeanprocess(NMF_log, train_tf, 'nmf', 2, actual_tag)

#===============================Two kinds of Transformation========================================#
NMF_array=train_NMF_array
NMF_normal=matrixnormal(NMF_array)
NMF_log=matrixlog(NMF_normal)
kmeanprocess(NMF_log, train_tf, 'nmf', 2, actual_tag)
#different order
NMF_array=train_NMF_array
NMF_log=matrixlog(NMF_array)
NMF_normal=matrixnormal(NMF_log)
kmeanprocess(NMF_normal, train_tf, 'nmf', 2, actual_tag)

pyplot.show()