# -*- coding: UTF-8 -*-

import math
import numpy as np
import time
start=time.clock()
import csv
from sklearn.svm import SVC
import numpy as np
from sklearn import preprocessing,neighbors
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


#向量夹角余弦
def cosine_distance(u, v):
    return np.dot(u, v) / (math.sqrt(np.dot(u, u)) * math.sqrt(np.dot(v, v)))

#皮尔逊相关系数：
'''
This is the formula for the pearson correlation coefficient:
len(x) * sumOfAllXY -(sumAllX*sumAllY)
/
sqrt[(n*sumAll(x2)-(sumAllX)^2) * (n * sumAll(y^2) - (sumAllY)^2 )]
'''

'''
create two vectors to test the correlation between them
'''

def dot_product(x_vector, y_vector):
    assert(len(x_vector) == len(y_vector))
    return x_vector.dot(y_vector)

def sum_all_Items_pow_2(input_vector):
    return sum([x**2 for x in input_vector])

def numerator(x_vector, y_vector):
    assert(len(x_vector) == len(y_vector))
    return len(x_vector) * dot_product(x_vector, y_vector) - sum(x_vector) * sum(y_vector)

def denominator(x_vector, y_vector):
    assert(len(x_vector) == len(y_vector))
    return np.sqrt((len(x_vector) * sum_all_Items_pow_2(x_vector) - sum(x_vector)**2)
                        * (len(y_vector) * sum_all_Items_pow_2(y_vector) - sum(y_vector)**2))

def pearson(x_vector, y_vector):
    return numerator(x_vector, y_vector) / denominator(x_vector, y_vector)




x=[1,2,3,4,5,6,7,8,9,10,11]
x=np.array(x)
y=[3,3,4,5,6,7,8,9,10,11,12]
y=np.array(y)
nn=[1,2,3,4,5]

# cosi=cosine_distance(x,y)
# print("Cosine Distance: {0:.4f}" .format(cosi))

# pea=pearson(x, y)
# print("Pearson Correlation: {0:.4f}" .format(pea))

####################################################################################
# 特征值读取
# featureList
allElectronicsData=open(r'speed.csv','rb')
reader=csv.reader(allElectronicsData)
headers=reader.next()
# print(headers)
temp1=[]

for row in reader:
    # print(row[-1])
    b=float(row[-1])
    temp1.append(b)
# print(temp1)

n=len(temp1)

featureList=[]
#
for i in range(0,n-4):
    k1=[]
    k1=temp1[i]
    k2=temp1[i+1]
    k3=temp1[i+2]
    k4=temp1[i+3]
    k5=temp1[i+4]

    k=[k1,k2,k3,k4,k5]
    # print(k)
    featureList.append(k)


# print(kk)


incident_feature=[76.3,69.0,70.8,68.0,60.0]
incident_feature=np.array(incident_feature)

peaList=[]
for i in range(0,n-4):
    kk=featureList[i]
    kk=np.array(kk)
    pea=pearson(incident_feature, kk)
    # print(pea)
    peaList.append([pea])
# print(peaList)

y_train=[0,0,0,0,1,1,0,1,1,0,0,1,0]
clf=SVC(kernel='rbf',C=1e3,gamma=0.001)
clf.fit(peaList,y_train)


pre_speed=[76.5,69.1,71,67.0,62.0]
pre_pea=pearson(incident_feature,pre_speed)
result=clf.predict(pre_pea)
print(result)

