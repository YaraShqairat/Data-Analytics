# -*- coding: utf-8 -*-
"""
Created on Sun Dec 04 14:44:32 2016

@author: Yara
"""

get_ipython().magic(u'matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn import tree
from sklearn.externals.six import StringIO  
from sklearn import linear_model,cross_validation

#Enter the correct path where the data file is stored on your machine
data = pd.read_csv('Data\census.csv')
data

data.count(0)/data.shape[0] * 100

data = data.dropna(how='any')
del data['education_num']



# Hypothesis 1: People who are older, earn more.


hist_above_50 = plt.hist(data[data.greater_than_50k == 1].age.values, 10, facecolor='green', alpha=0.5)
plt.title('Age distribution of Above 50K earners')
plt.xlabel('Age')
plt.ylabel('Frequency')


hist_below_50 = plt.hist(data[data.greater_than_50k == 0].age.values, 10, facecolor='green', alpha=0.5)
plt.title('Age distribution of below 50K earners')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Hypothesis 2: Earning Bias based on working class 


dist_data = pd.concat([data[data.greater_than_50k == 1].groupby('workclass').workclass.count()
                          , data[data.greater_than_50k == 0].groupby('workclass').workclass.count()], axis=1)

dist_data.columns = ['wk_class_gt50','wk_class_lt50']

dist_data_final = dist_data.wk_class_gt50 / (dist_data.wk_class_lt50 + dist_data.wk_class_gt50 )

dist_data_final.sort_values(ascending=False)
ax = dist_data_final.plot(kind = 'bar', color = 'r', y='Percentage')
ax.set_xticklabels(dist_data_final.index, rotation=30, fontsize=8, ha='right')
ax.set_xlabel('Working Class')
ax.set_ylabel('Percentage of People')
plt.show()

# Hypothesis 3: People with more education, earn more


dist_data = pd.concat([data[data.greater_than_50k == 1].groupby('education').education.count()
                          , data[data.greater_than_50k == 0].groupby('education').education.count()], axis=1)

dist_data.columns = ['education_gt50','education_lt50']

dist_data_final = dist_data.education_gt50 / (dist_data.education_gt50 + dist_data.education_lt50 )

dist_data_final.sort_values(ascending = False)
ax =dist_data_final.plot(kind = 'bar', color = 'r')
ax.set_xticklabels(dist_data_final.index, rotation=30, fontsize=8, ha='right')
ax.set_xlabel('Education Level')
ax.set_ylabel('Percentage of People')
plt.show()

# Hypothesis 4: Married People tend to earn more</strong>


dist_data = pd.concat([data[data.greater_than_50k == 1].groupby('marital_status').marital_status.count()
                          , data[data.greater_than_50k == 0].groupby('marital_status').marital_status.count()], axis=1)

dist_data.columns = ['marital_status_gt50','marital_status_lt50']

dist_data_final = dist_data.marital_status_gt50 / (dist_data.marital_status_gt50 + dist_data.marital_status_lt50 )

dist_data_final.sort_values(ascending = False)

ax = dist_data_final.plot(kind = 'bar', color = 'r')
ax.set_xticklabels(dist_data_final.index, rotation=30, fontsize=8, ha='right')
ax.set_xlabel('Marital Status')
ax.set_ylabel('Percentage of People')
plt.show()

# Hypothesis 5: There is bias in earning based on occupation 


dist_data = pd.concat([data[data.greater_than_50k == 1].groupby('occupation').occupation.count()
                          , data[data.greater_than_50k == 0].groupby('occupation').occupation.count()], axis=1)

dist_data.columns = ['occupation_gt50','occupation_lt50']

dist_data_final = dist_data.occupation_gt50 / (dist_data.occupation_gt50 + dist_data.occupation_lt50 )

dist_data_final.sort_values(ascending = False)

ax = dist_data_final.plot(kind = 'bar', color = 'r')

ax.set_xticklabels(dist_data_final.index, rotation=30, fontsize=8, ha='right')
ax.set_xlabel('Occupation')
ax.set_ylabel('Percentage of People')
plt.show()

# <strong> Hypothesis 6:  There is bias in earning based on race </strong>



dist_data = pd.concat([data[data.greater_than_50k == 1].groupby('race').race.count()
                          , data[data.greater_than_50k == 0].groupby('race').race.count()], axis=1)

dist_data.columns = ['race_gt50','race_lt50']

dist_data_final = dist_data.race_gt50 / (dist_data.race_gt50 + dist_data.race_lt50 )

dist_data_final.sort_values(ascending = False)

ax = dist_data_final.plot(kind = 'bar', color = 'r')

ax.set_xticklabels(dist_data_final.index, rotation=30, fontsize=8, ha='right')
ax.set_xlabel('Race')
ax.set_ylabel('Percentage of People')

plt.show()


# <strong>Hypothesis 7: Men earn more</strong>


dist_data = pd.concat([data[data.greater_than_50k == 1].groupby('gender').gender.count()
                          , data[data.greater_than_50k == 0].groupby('gender').gender.count()], axis=1)

dist_data.columns = ['gender_gt50','gender_lt50']

dist_data_final = dist_data.gender_gt50 / (dist_data.gender_gt50 + dist_data.gender_lt50 )

dist_data_final.sort_values(ascending = False)

ax = dist_data_final.plot(kind = 'bar', color = 'r')

ax.set_xticklabels(dist_data_final.index, rotation=30, fontsize=8, ha='right')
ax.set_xlabel('Gender')
ax.set_ylabel('Percentage of People')

plt.show()


#  Hypothesis 8: People who clock in more hours, earn more



hist_above_50 = plt.hist(data[data.greater_than_50k == 1]
.hours_per_week.values, 10, facecolor='green',
alpha=0.5)

plt.title('Hours per week distribution of Above 50K earners')





hist_below_50 = plt.hist(data[data.greater_than_50k ==
0].hours_per_week.values, 10, facecolor='green', alpha=0.5)

plt.title('Hours per week distribution of Below 50K earners')


plt.show()

# Hypothesis 9: There is a bias in earning based on the country of origin

plt.figure(figsize=(10,5))
dist_data = pd.concat([data[data.greater_than_50k == 1].groupby('native_country').native_country.count()
                          , data[data.greater_than_50k == 0].groupby('native_country').native_country.count()], axis=1)

dist_data.columns = ['native_country_gt50','native_country_lt50']

dist_data_final = dist_data.native_country_gt50 / (dist_data.native_country_gt50 + dist_data.native_country_lt50 )

dist_data_final.sort(ascending = False)

ax = dist_data_final.plot(kind = 'bar', color = 'r')

ax.set_xticklabels(dist_data_final.index, rotation=40, fontsize=8, ha='right')
ax.set_xlabel('Country')
ax.set_ylabel('Percentage of People')



# ## Decision Trees


#Enter the correct path where the data file is stored on your machine

data_test = pd.read_csv('Data\census_test.csv')
data_test = data_test.dropna(how='any')
formula = 'greater_than_50k ~  age + workclass + education + marital_status + occupation + race + gender + hours_per_week + native_country ' 

y_train,x_train = dmatrices(formula, data=data, return_type='dataframe')
y_test,x_test = dmatrices(formula, data=data_test, return_type='dataframe')

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)

from sklearn.metrics import classification_report

y_pred = clf.predict(x_test)

print pd.crosstab(y_test.greater_than_50k
                  ,y_pred
                  ,rownames = ['Actual']
                  ,colnames = ['Predicted'])

print '\n \n'

print classification_report(y_test.greater_than_50k,y_pred)

import sklearn.ensemble as sk

clf = sk.RandomForestClassifier(n_estimators=100)
clf = clf.fit(x_train, y_train.greater_than_50k)

y_pred = clf.predict(x_test)

print pd.crosstab(y_test.greater_than_50k
                  ,y_pred
                  ,rownames = ['Actual']
                  ,colnames = ['Predicted'])

print '\n \n'

print classification_report(y_test.greater_than_50k,y_pred)


clf = sk.RandomForestClassifier(n_estimators=100, oob_score=True,min_samples_split=5)
clf = clf.fit(x_train, y_train.greater_than_50k)

y_pred = clf.predict(x_test)

print pd.crosstab(y_test.greater_than_50k
                  ,y_pred
                  ,rownames = ['Actual']
                  ,colnames = ['Predicted'])

print '\n \n'

print classification_report(y_test.greater_than_50k,y_pred)

clf = sk.RandomForestClassifier(n_estimators=100, oob_score=True,min_samples_split=5, min_samples_leaf= 2)
clf = clf.fit(x_train, y_train.greater_than_50k)

y_pred = clf.predict(x_test)

print pd.crosstab(y_test.greater_than_50k
                  ,y_pred
                  ,rownames = ['Actual']
                  ,colnames = ['Predicted'])

print '\n \n'

print classification_report(y_test.greater_than_50k,y_pred)


model_ranks = pd.Series(clf.feature_importances_, index=x_train.columns, name='Importance').sort_values(ascending=False, inplace=False)
model_ranks.index.name = 'Features'
top_features = model_ranks.iloc[:31].sort_values(ascending=True, inplace=False)
plt.figure(figsize=(15,7))
ax = top_features.plot(kind='barh')
_ = ax.set_title("Variable Ranking")
_ = ax.set_xlabel('Performance')
_ = ax.set_yticklabels(top_features.index, fontsize=8)



