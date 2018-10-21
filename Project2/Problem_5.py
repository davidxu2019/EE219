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
from sklearn.preprocessing import FunctionTransformer
from sklearn import preprocessing

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
homogeneity=[[] for i in range(7)]
completeness=[[] for i in range(7)]
v_measure=[[] for i in range(7)]
adjusted_rand=[[] for i in range(7)]
adjusted_mutual_info=[[] for i in range(7)]

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
			pyplot.scatter(tfidf_matrix[i, 0], tfidf_matrix[i, 1],s=12, c=color[kmean_result[i]],alpha=0.5)
		for i in range(0,center.shape[0]):
			pyplot.scatter(center[i,0], center[i,1], marker='^', s=100,  color='k', alpha=0.6)
		
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
	result=preprocessing.scale(matrix, with_mean=False)
	return result

def kmeansimple(reduced_matrix,method,n_cluster,actual_tag, index):
	kmean = KMeans(n_clusters=n_cluster, random_state=42)
	train_kmean = kmean.fit_predict(reduced_matrix)
	kmeans_tag=kmean.labels_
	
	homogeneity[index].append(homogeneity_score(actual_tag, kmeans_tag))
	completeness[index].append(completeness_score(actual_tag, kmeans_tag))
	v_measure[index].append(v_measure_score(actual_tag, kmeans_tag))
	adjusted_rand[index].append(adjusted_rand_score(actual_tag, kmeans_tag))
	adjusted_mutual_info[index].append(adjusted_mutual_info_score(actual_tag, kmeans_tag))
	
def kmeanprocess(reduced_matrix, tfidf_matrix, method, n_cluster, actual_tag):
	kmean = KMeans(n_clusters=n_cluster, random_state=42)
	train_kmean = kmean.fit_predict(reduced_matrix)
	kmeans_tag=kmean.labels_
	print("============================================================================")
	print (method)
	print("Homogeneity: %0.3f" % homogeneity_score(actual_tag, kmeans_tag))
	print("Completeness: %0.3f" % completeness_score(actual_tag, kmeans_tag))
	print("V-measure: %0.3f" % v_measure_score(actual_tag, kmeans_tag))
	print("Adjusted Rand score: %.3f" % adjusted_rand_score(actual_tag, kmeans_tag))
	print("Adjusted Mutual Info Score: %.3f" % adjusted_mutual_info_score(actual_tag, kmeans_tag))
	matrix=confusion_matrix(actual_tag, kmeans_tag)
	pyplot.matshow(matrix)
	train_2d=dimreduce(train_tf, 2, method)
	train_2d_kmean = kmean.fit_predict(train_2d)
	center = kmean.cluster_centers_
	plotscatter(train_kmean, train_2d,center)


color = ["grey", "lightcoral", "maroon", "mistyrose", "coral", "peachpuff", "darkorange", "orange", "darkgoldenrod",
             "olive", "yellowgreen", "lawngreen", "lightgreen", "g", "mediumseagreen", "mediumturquoise", "c", "cadetblue",
             "skyblue", "dodgerblue"]
categories=['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','comp.windows.x','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey','sci.crypt','sci.electronics','sci.med','sci.space','misc.forsale','talk.politics.misc','talk.politics.guns','talk.politics.mideast','talk.religion.misc','alt.atheism','soc.religion.christian']

train_data=fetch_20newsgroups(subset='train',categories=categories,shuffle=True,random_state=42)


#set stopwords to exclude the stop words
stop_words=text.ENGLISH_STOP_WORDS
vectorizer=text.CountVectorizer(analyzer = 'word',lowercase=True,tokenizer=tokenizer(),min_df=3,stop_words=stop_words)
train_counts=vectorizer.fit_transform(train_data.data)
train_transformer = TfidfTransformer(sublinear_tf=True,use_idf=True)
train_tf = train_transformer.fit_transform(train_counts)

actual_tag=[]#0 means computer techonology and 1 means recreational activity
actual_tag_inv=[]
for i in train_data.target:
	actual_tag.append(i)
	actual_tag_inv.append(19-i)

r_list=[1,2,3,5,10,20,50,100,300]


for i in range(0,9):
#get reduced matrix

	k_lsi=r_list[i]
	k_nmf=r_list[i]
	train_LSI_array = dimreduce(train_tf, k_lsi, 'lsi')
	train_NMF_array = dimreduce(train_tf, k_nmf, 'nmf')

	#attach actual label to the document
	

	#===============================Part a========================================#
	kmeansimple(train_LSI_array, 'lsi', 20, actual_tag,0)
	kmeansimple(train_NMF_array, 'nmf', 20, actual_tag,1)


	#===============================LSI Normalize========================================#
	#normalize tfidf matrix
	normal=Normalizer(copy=False)
	LSI_array=train_LSI_array
	LSI_normal=matrixnormal(LSI_array)
	kmeansimple(LSI_normal, 'lsi', 20, actual_tag,2)
	#===============================NMF Normalize========================================#
	NMF_array=train_NMF_array
	NMF_normal=matrixnormal(NMF_array)
	kmeansimple(NMF_normal, 'nmf', 20, actual_tag,3)

	#===============================NMF Non-linear Transformation========================================#
	NMF_array=train_NMF_array
	NMF_log=matrixlog(NMF_array)
	kmeansimple(NMF_log, 'nmf', 20, actual_tag,4)

	#===============================Two kinds of Transformation========================================#
	NMF_array=train_NMF_array
	NMF_normal=matrixnormal(NMF_array)
	NMF_log=matrixlog(NMF_normal)
	kmeansimple(NMF_log, 'nmf', 20, actual_tag,5)
	#different order
	NMF_array=train_NMF_array
	NMF_log=matrixlog(NMF_array)
	NMF_normal=matrixnormal(NMF_log)
	kmeansimple(NMF_normal, 'nmf', 20, actual_tag,6)

lw=1
for i in range(7):
	pyplot.figure()
	color = ['b', 'g', 'r','m', 'y']
	pyplot.plot(r_list, homogeneity[i], color=color[0], lw=lw, label='Homogeneity Score')
	pyplot.plot(r_list, completeness[i], color=color[1], lw=lw, label='Completeness Score')
	pyplot.plot(r_list, v_measure[i], color=color[2], lw=lw, label='V_measure Score')
	pyplot.plot(r_list, adjusted_rand[i], color=color[3], lw=lw, label='Adjusted Rand Score')
	pyplot.plot(r_list, adjusted_mutual_info[i], color=color[4], lw=lw, label='Adjusted Mutual Info Score')
	pyplot.xlim([0.0, 350])
	pyplot.ylim([0.0, 1.0])
	pyplot.xlabel('Number of Principle Components')
	pyplot.ylabel('Score')
	pyplot.title('5 metrics Vs Number of Principle Components')
	pyplot.legend(loc='upper right', fontsize='small')

#===============================Part a========================================#
train_LSI_array = dimreduce(train_tf, 100, 'lsi')
train_NMF_array = dimreduce(train_tf, 10, 'nmf')
kmeanprocess(train_LSI_array, train_tf, 'lsi', 20, actual_tag)
kmeanprocess(train_NMF_array, train_tf, 'nmf', 20, actual_tag)


#===============================LSI Normalize========================================#
#normalize tfidf matrix
LSI_array=dimreduce(train_tf, 10, 'lsi')
LSI_normal=matrixnormal(LSI_array)
kmeanprocess(LSI_normal, train_tf, 'lsi', 20, actual_tag)
#===============================NMF Normalize========================================#
NMF_array=dimreduce(train_tf, 50, 'nmf')
NMF_normal=matrixnormal(NMF_array)
kmeanprocess(NMF_normal, train_tf, 'nmf', 20, actual_tag)

#===============================NMF Non-linear Transformation========================================#
NMF_array=dimreduce(train_tf, 10, 'nmf')
NMF_log=matrixlog(NMF_array)
kmeanprocess(NMF_log, train_tf, 'nmf', 20, actual_tag)

#===============================Two kinds of Transformation========================================#
NMF_array=dimreduce(train_tf, 100, 'nmf')
NMF_normal=matrixnormal(NMF_array)
NMF_log=matrixlog(NMF_normal)
kmeanprocess(NMF_log, train_tf, 'nmf', 20, actual_tag)
#different order
NMF_array=dimreduce(train_tf, 50, 'nmf')
NMF_log=matrixlog(NMF_array)
NMF_normal=matrixnormal(NMF_log)
kmeanprocess(NMF_normal, train_tf, 'nmf', 20, actual_tag)

pyplot.show()