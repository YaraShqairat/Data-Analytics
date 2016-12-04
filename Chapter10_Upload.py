# -*- coding: utf-8 -*-
"""
Created on Sun Dec 04 20:43:28 2016

@author: Yara
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from scipy.stats.stats import pearsonr  
from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans,vq
from scipy.spatial.distance import cdist

#Enter the correct path where the data file is stored on your machine

df = pd.read_csv ('./Data/tshirt_sizes.csv')


print df[:10]




d_color = {
    "S": "b",
    "M": "r",
    "L": "g",
}
fig, ax = plt.subplots()
for size in ["S", "M", "L"]:
    color = d_color[size]
    df[df.Size == size].plot(kind='scatter', x='Height', y='Weight', label=size, ax=ax, color=color)
handles, labels = ax.get_legend_handles_labels()
_ = ax.legend(handles, labels, loc="upper left")



# In[58]:

km = KMeans(3,init='k-means++', random_state=3425) # initialize
km.fit(df[['Height','Weight']])
df['SizePredict'] = km.predict(df[['Height','Weight']])
print pd.crosstab(df.Size
                  ,df.SizePredict
                  ,rownames = ['Size']
                  ,colnames = ['SizePredict'])


# In[55]:

c_map = {
    2: "M",
    1: "S",
    0: "L",
}

df['SizePredict'] = df['SizePredict'].map(c_map)
df['SizePredict'][:10]


# In[56]:

fig, ax = plt.subplots()
for size in ["S", "M", "L"]:
    color = d_color[size]
    df[df.SizePredict == size].plot(kind='scatter', x='Height', y='Weight', label=size, ax=ax, color=color)
handles, labels = ax.get_legend_handles_labels()
_ = ax.legend(handles, labels, loc="upper left")


# In[96]:

#Enter the correct path where the data file is stored on your machine

df = pd.read_csv('./Data/UN.csv')

print('----')

print('Individual columns - Python data types')
[(col, type(df[col][0])) for col in df.columns] 


# In[97]:

print('Percentage of the values complete in the columns')
df.count(0)/df.shape[0] * 100


# In[98]:

df = df[['lifeMale', 'lifeFemale', 'infantMortality', 'GDPperCapita']]

df = df.dropna(how='any')


# In[67]:

K = range(1,10)

# scipy.cluster.vq.kmeans
k_clusters = [kmeans(df.values,k) for k in K] # apply kmeans 1 to 10
k_clusters[:3]


# In[68]:

euclidean_centroid = [cdist(df.values, centroid, 'euclidean') for (centroid,var) in k_clusters]
print '-----with 1 cluster------'
print euclidean_centroid[0][:5]

print '-----with 2 cluster------'
print euclidean_centroid[1][:5]


# In[69]:

distance = [np.min(D,axis=1) for D in euclidean_centroid]
print '-----with 1st cluster------'
print distance[0][:5]
print '-----with 2nd cluster------'
print distance[1][:5]


# In[70]:

avgWithinSumSquare = [sum(d)/df.values.shape[0] for d in distance]
avgWithinSumSquare


# In[71]:

point_id = 2
# plot elbow curve
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, avgWithinSumSquare, 'b*-')
ax.plot(K[point_id], avgWithinSumSquare[point_id], marker='o', markersize=12, 
      markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
plt.grid(True)
plt.xlabel('No. of clusters')
plt.ylabel('Avg sum of squares')
tt = plt.title('Elbow Curve')


# We'll now apply the K-Means algorithm to cluster the countries together

# In[99]:

km = KMeans(3, init='k-means++', random_state = 3425) # initialize
km.fit(df.values)
df['countrySegment'] = km.predict(df.values)
df[:5]


# Let's find the average GDP per capita for each country segment

# In[100]:

df.groupby('countrySegment').GDPperCapita.mean()



# In[101]:

clust_map = {
    0:'Developing',
    1:'Under Developed',
    2:'Developed'
}

df.countrySegment = df.countrySegment.map(clust_map)
df[:10]


# Let's see the GDP vs infant mortality rate of the countries based on the cluster

# In[110]:

d_color = {
    'Developing':'y',
    'Under Developed':'r',
    'Developed':'g'
}

fig, ax = plt.subplots()
for clust in clust_map.values():
    color = d_color[clust]
    df[df.countrySegment == clust].plot(kind='scatter', x='GDPperCapita', y='infantMortality', label=clust, ax=ax, color=color)
handles, labels = ax.get_legend_handles_labels()
_ = ax.legend(handles, labels, loc="upper right")


# In[111]:

fig, ax = plt.subplots()
for clust in clust_map.values():
    color = d_color[clust]
    df[df.countrySegment == clust].plot(kind='scatter', x='GDPperCapita', y='lifeMale', label=clust, ax=ax, color=color)
handles, labels = ax.get_legend_handles_labels()
_ = ax.legend(handles, labels, loc="lower right")



# In[112]:

fig, ax = plt.subplots()
for clust in clust_map.values():
    color = d_color[clust]
    df[df.countrySegment == clust].plot(kind='scatter', x='GDPperCapita', y='lifeFemale', label=clust, ax=ax, color=color)
handles, labels = ax.get_legend_handles_labels()
_ = ax.legend(handles, labels, loc="lower right")

