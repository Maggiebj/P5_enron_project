#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot



### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file) #{name1:{...},name2:{...}...}
#print data_dict
	
	   
###DATA EXPLORATORY
##getting  the total number of people and the number if poi
poicount=0 
namecount=len(data_dict.keys())
for name,detail in data_dict.iteritems():
    for key,value in detail.iteritems():
        if key=='poi' and value==1:
            poicount +=1	
print "the total number of people is ",namecount
print "the number of POI is ",poicount
print "The POI classes are unbalanced!"

##getting the total number of features
allfeatures=data_dict[data_dict.keys()[0]].keys()
print "all featues are {}".format(len(data_dict[data_dict.keys()[0]].keys()))
print allfeatures

##checking missing values
#convert dictionay data_dict to dataframe  
#df = pd.DataFrame.from_records(list(data_dict.values()))#data_dict.values() is a list of {{salary:..},{salary:..}..}
#employees = pd.Series(list(data_dict.keys()))
#df.set_index(employees, inplace=True)
#df_dict = df.to_dict('index')
#if df_dict == data_dict :
#    print "data_dict is succefully converted to df!"
df = pd.DataFrame.from_dict(data_dict, orient='index', dtype=np.float)
df.replace(to_replace='NaN', value=np.nan, inplace=True)#replaced "NaN"str by np.nan
print "checking missing values in features..."
print df.isnull().any(axis=0)
def nullpercentage(key):
   nullvalue=df[key]
   nullvalue=nullvalue[df.isnull()[key]]
   percentage=round(float(len(nullvalue))/len(df[key]),3)
   print 'the feature {} has null value percentage is {}'.format(key,percentage) 
   
   return None
nullpercentage('salary') 
nullpercentage('deferral_payments')
nullpercentage('total_payments')
nullpercentage('loan_advances')
nullpercentage('bonus')
nullpercentage('restricted_stock_deferred')
nullpercentage('deferred_income')
nullpercentage('total_stock_value')
nullpercentage('expenses')
nullpercentage('exercised_stock_options')
nullpercentage('other')
nullpercentage('long_term_incentive')
nullpercentage('restricted_stock')
nullpercentage('director_fees')
nullpercentage('to_messages')
nullpercentage('from_poi_to_this_person')
nullpercentage('from_messages')
nullpercentage('from_this_person_to_poi')
nullpercentage('shared_receipt_with_poi')



### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list=["poi","salary","total_payments","bonus","total_stock_value","expenses","exercised_stock_options","other","restricted_stock","shared_receipt_with_poi","rate of poi/from","rate of poi/to"]

### Task 2: Remove outliers
features = ["salary","bonus"]
data = featureFormat(data_dict, features)
maxoutlier=0
for point in data:
    salary = point[0]
    bonus = point[1]
    #matplotlib.pyplot.scatter( salary, bonus )
    if maxoutlier<point[0]:  ##find the maxoutlier of salary and second outlier of salary
        secondoutlier=maxoutlier
        maxoutlier=point[0]
for i in range(len(data_dict.keys())):
        if data_dict[data_dict.keys()[i]]['salary']==maxoutlier :
            print "maxoutlier is: ",data_dict.keys()[i]
        elif data_dict[data_dict.keys()[i]]['salary']==secondoutlier:
            print "secondoutlier is: ",data_dict.keys()[i]

data_dict.pop('TOTAL') #remove 'TOTAL'key and return it,145 person left
print "We removed the maxoutlier!!!"
print "after removed maxoutlier the number of row is",len(data_dict)

### Task 3: Create new feature(s)
#Adding new features of "rate of poi/from" and "rate of poi/to", and ignore 'to_messages','from_poi_to_this_person','from_messages','from_this_person_to_poi'
df_1=df.drop("TOTAL")
df_1["rate of poi/from"]=df_1["from_this_person_to_poi"]/df_1["from_messages"]
df_1["rate of poi/to"]=df_1["from_poi_to_this_person"]/df_1["to_messages"]
#for featureFormat() using str"NaN", so replace np.nan to str"NaN"
df_1.replace(to_replace=np.nan, value="NaN", inplace=True)
#convert df_1 to dictionary df_dict_1
data_dict_1 = df_1.to_dict('index')


### Store to my_dataset for easy export below.
my_dataset = data_dict_1

### Extract features and labels from dataset for local testing	
print "CHECKING FOR SELF IMPLEMENTED FEATURES......"
features_list_1=["poi","rate of poi/to","rate of poi/from"]
data = featureFormat(my_dataset, features_list_1, remove_NaN=True, remove_all_zeroes=True, sort_keys = True)
labels, features = targetFeatureSplit(data) #len(labels),len(features):143,143

print "after extract features and labels,with remove_all_zeroes=True, the number of row is:",len(labels)
print "the number of poi is",sum(labels)
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
##parameter tuned for decisiontree when testing 2 self-implemented features
from sklearn.cross_validation import StratifiedShuffleSplit
cv=StratifiedShuffleSplit(labels,100,test_size=0.1,random_state=42)
clf_dt=DecisionTreeClassifier()
parameters={"min_samples_leaf":[2,4,6],"min_samples_split":[2,4,6],"random_state":[24,46,60]}
from sklearn.grid_search import GridSearchCV
a_grid_search=GridSearchCV(estimator=clf_dt,param_grid=parameters,cv=cv,scoring='recall')
a_grid_search.fit(features,labels)
best_clf=a_grid_search.best_estimator_
#print "best parameter for decisiontree with 2 self-impletented features",best_clf
test_classifier(best_clf,my_dataset,features_list_1)






#MYCODE for standarized manually
#df_features.replace(to_replace=0, value=np.nan, inplace=True)#before standardize ,0 should be converted to np.nan
#print "df_features:",df_features.iloc[0]
#print "df_features mean",df_features.mean(axis=0)
#print "df_features std",df_features.std(axis=0)

#for column in features_list[1:]:
#    df_features[column]=(df_features[column]-df_features[column].mean(axis=0))/df_features[column].std(axis=0)
#df_features.replace(to_replace=np.nan, value=0, inplace=True)#"All null values has to be set 0 before using sklearn"	
#standard_features=[]
#for i in range(len(df_features)):  
#	standard_features.append(list(df_features.iloc[i][features_list[1:]])) #save numerical features[1:10] to standard_features list,customer features[10:]		
#print "standardized features data",standard_features[0]


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
##feature selection 
print "USING DECISIONTREE TRAINING......"
features_list_2=["poi","salary","total_payments","bonus","total_stock_value","expenses","exercised_stock_options","other","restricted_stock",'to_messages','from_messages']
data = featureFormat(my_dataset, features_list_2, remove_NaN=True, remove_all_zeroes=True, sort_keys = True)
labels, features = targetFeatureSplit(data) 
cv=StratifiedShuffleSplit(labels,1,test_size=0.1,random_state=42)
for train_idx, test_idx in cv:
	features_train = []
	features_test  = []
	labels_train   = []
	labels_test    = []
	for ii in train_idx:
		features_train.append( features[ii] )
		labels_train.append( labels[ii] )
	for jj in test_idx:
		features_test.append( features[jj] )
		labels_test.append( labels[jj] )
from sklearn.tree import DecisionTreeClassifier
clf_dt=DecisionTreeClassifier(random_state=42)
clf_dt=clf_dt.fit(features_train,labels_train)


importances=clf_dt.feature_importances_
import numpy as np
indices = np.argsort(importances)[::-1]
features_list_3=features_list_2[1:]
print 'Feature Ranking: '
for i in range(len(features_list_3)): #exclude the first  "poi"
    print "feature no. {}: {} ({})".format(i+1,features_list_3[indices[i]],importances[indices[i]])



testfeatureswithpoi=[]
testfeatureswithpoi.append("poi")
testfeatureswithpoi.append(features_list_3[indices[0]])
testfeatureswithpoi.append(features_list_3[indices[1]])
testfeatureswithpoi.append(features_list_3[indices[2]])
#testfeatureswithpoi.append(features_list_3[indices[3]])
#testfeatureswithpoi.append(features_list_3[indices[4]])


print "decisiontree with all features"
test_classifier(clf_dt,my_dataset,features_list_2)
print "decisiontree with selected features"
print "features list selected:",testfeatureswithpoi
test_classifier(clf_dt,my_dataset,testfeatureswithpoi)




##second algorithm SVC trial, using scaling before training
print "USING SVC TRAINING "

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC



#svc with all features after scaled and default parameters 
pipe=Pipeline(steps=[('scaling', MinMaxScaler()),('classifier',SVC())])
print "svc with all features after scaled:"
test_classifier(pipe,my_dataset,features_list_2)


#svc features selection with RFECV
#scaler=MinMaxScaler()
#features_scaled=scaler.fit_transform(features)
#svc_clf=SVC()
#svc_fs = RFECV(svc_clf,cv=5,scoring="recall")
#svc_fs=svc_fs.fit(features_scaled,labels)
#print "RFECV features selection:{}{}",format(features_list[1:],svc_fs.support_)
#print "RFECV features selection ranking:{}{}",format(features_list[1:],svc_fs.ranking_)
#X_new=pipeline.named_steps['kbest']
#feature_scores=['%.2f' % elem for elem in X_new.scores_]
#feature_selected_tuple=[(features_list[i+1],feature_scores[i]) for i in X_new.get_support(indices=True)]
#feature_selected_tuple = sorted(feature_selected_tuple, key=lambda feature: float(feature[1]), reverse=True)
#print "feature_selected_tuple by selectkbest",feature_selected_tuple

#features_list2=["poi","bonus","salary","rate of poi/from","total_stock_value","shared_receipt_with_poi","exercised_stock_options","total_payments","restricted_stock"]

#from sklearn.decomposition import PCA
#pca=PCA(n_components=2)
#pca.fit(features_train)
#print "pca.explained_variance_ratio_",pca.explained_variance_ratio_
#X_train_pca = pca.transform(features_train)
#X_test_pca = pca.transform(features_test)

#cv = StratifiedShuffleSplit(labels,50, random_state = 42)
#clf=SVC()
#scv_params={'clf__C': [1e-5, 1e-2, 1e-1, 1, 10, 1e2, 1e5],
#                       'clf__gamma': [0.0],
#                       'clf__kernel': ['rbf'],
#                       'clf__tol': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5] 
#                       }
#pipe=Pipeline(steps=[("minmaxer",MinMaxScaler()),("clf",SVC())])
#from sklearn.grid_search import GridSearchCV
#a_grid_search=GridSearchCV(pipe,param_grid=scv_params,cv=cv,scoring='recall')
#a_grid_search.fit(features,labels)
#best_clf=a_grid_search.best_estimator_
#print "svc  best estimator",best_clf
#clf=SVC(C=100000.0, cache_size=200, class_weight=None, coef0=0.0,decision_function_shape=None, degree=3, gamma="auto", kernel='rbf',max_iter=-1, probability=False, random_state=None, shrinking=True,tol=0.1, verbose=False)





##third algorithm adaboost trial
print "USING ADABOOST TRAINING..."

from sklearn.ensemble import AdaBoostClassifier
clf_ada_all=AdaBoostClassifier(n_estimators=50,random_state=42)
print "adaboost with all features"
#test_classifier(clf_ada_all,my_dataset,features_list_2)

clf_ada=AdaBoostClassifier(n_estimators=50)
from sklearn.grid_search import GridSearchCV
parameters={"random_state":[24,42,60],"learning_rate":[1.0,2.0,3.0]}
cv=StratifiedShuffleSplit(labels,10,test_size=0.1,random_state=42)
a_grid_search=GridSearchCV(estimator=clf_ada,param_grid=parameters,cv=cv,scoring='recall')
grid_search_result=a_grid_search.fit(features,labels)
clf=grid_search_result.best_estimator_

print "adaboost with selected features"
print "features list selected:",testfeatureswithpoi
test_classifier(clf,my_dataset,testfeatureswithpoi)
#test_classifier(clf_ada_all,my_dataset,testfeatureswithpoi)
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#from sklearn.metrics import recall_score,precision_score,accuracy_score
#accuracy=accuracy_score(labels_test,pred)
#recall = recall_score(labels_test,pred)
#precision=precision_score(labels_test,pred)
#print "accuracy score is ",accuracy
#print "recall score is ",recall
#print "precision score is ",precision


# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, testfeatureswithpoi)


##MYCODE :convert my_dataset dictionary to list
#import csv
#dictlist=[]
#temp=[]
#fieldnames=['name']


#for name,detail in my_dataset.iteritems():
#   temp.append(name)
   
#   for key,value in detail.iteritems():
       
#	   temp.append(value)
#	   if len(fieldnames)<=len(my_dataset[my_dataset.keys()[0]].keys()):
#	       fieldnames.append(key)
            
#   dictlist.append(temp) 
#   temp=[]   
#   #print dictlist
#   #print fieldname   
   
##write dictlist to csv
#f = open('enron.csv','wb')
#spamwriter = csv.writer(f, delimiter=',',)
                            
#spamwriter.writerow(fieldnames)
#for i in range(len(my_dataset)):
 #   spamwriter.writerow(dictlist[i])
#f.close()


