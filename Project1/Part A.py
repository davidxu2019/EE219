from sklearn.datasets import fetch_20newsgroups
import numpy
from matplotlib import pyplot

#retrieve the train dataset and test dataset

categories=['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']

train_data=fetch_20newsgroups(subset='train',categories=categories,shuffle=True,random_state=42)
train_test=fetch_20newsgroups(subset='test',categories=categories,shuffle=True,random_state=42)

#Show the histogram of dataset
pyplot.figure(1)
bins=[0,1,2,3,4,5,6,7,8]
pyplot.subplot(311)
pyplot.axis([0,8,0,1000])
pyplot.xlabel("Class Index")
pyplot.ylabel("Number of Documents")
pyplot.title("Train Dataset")
count_data,index,patches=pyplot.hist(train_data.target,bins,edgecolor='black',facecolor='green',alpha=0.6)

#Show the histogram of test dataset
pyplot.subplot(313)
pyplot.axis([0,8,0,600])
pyplot.xlabel("Class Index")
pyplot.ylabel("Number of Documents")
pyplot.title("Test Dataset")
count_test,index,patches=pyplot.hist(train_test.target,bins,edgecolor='black',facecolor='yellow',alpha=0.6)

pyplot.show()

for i in range(0,8):
	print ('Number of documents in',categories[i],'in train dataset is',count_data[i])
	print ('Number of documents in',categories[i],'in test dataset is',count_test[i])
	
print('Number of documents in Computer Techonology in train dataset is ',numpy.sum(count_data[0:4]))
print('Number of documents in Recreational Activity in train dataset is ',numpy.sum(count_data[4:8]))
print('Number of documents in Computer Techonology in test dataset is ',numpy.sum(count_test[0:4]))
print('Number of documents in Recreational Activity in test dataset is ',numpy.sum(count_test[4:8]))