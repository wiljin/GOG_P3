import numpy as np
import csv
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error as mae
import pandas as pd
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

# Predict via the user-specific median.
# If the user has no data, use the global median.

profile = "profiles.csv"
train_file = 'train.csv'
test_file  = 'test.csv'
small_test_file = 'small_test.csv'
soln_file  = 'submit_1.csv'
artists = "artists.csv"

# Load the profile data-- next is to skip headers
#profile_data = {}
#with open(profile, 'r') as profile_fh:
#    profile_csv = csv.reader(profile_fh, delimiter=',', quotechar='"')
#    next(profile_csv, None)
#    for row in profile_csv:
#        user   = row[0]
#        sex = row[1]
#        age  = row[2]
#        country = row[3]
#    
#        if not user in profile_data:
#            profile_data[user] = {}
#        if age> 10 & age<70:
#            profile_data[user][age] = age
#        if sex !="":
#            profile_data[user][sex] = sex
#        if country !="":
#            profile_data[user][country] = country

# Load the training data.
train_data = {}
with open(train_file, 'r') as train_fh:
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    next(train_csv, None)
    for row in train_csv:
        user   = row[0]
        artist = row[1]
        plays  = row[2]
    
        if not user in train_data:
            train_data[user] = {}
        train_data[user][artist] = int(plays)
        

#%% Test if this works
#for user,user_data in train_data.iteritems():
#        for plays in user_data.iteritems:
#            plays -= user_medians[user]
    

#%%
# Compute the global median, per-user median
plays_array  = []
user_medians = {}
for user, user_data in train_data.iteritems():
    user_plays = []
    for artist, plays in user_data.iteritems():
        plays_array.append(plays)
        user_plays.append(plays)

    user_medians[user] = np.median(np.array(user_plays))
global_median = np.median(np.array(plays_array))

#%%
X = np.load('X.npy')
Y= np.load('Y.npy')
#%%
together=[]
for i in xrange(len(X)):
    together.append(X[i,1],together[i,1]=int(Y[i]))

#%%
together_edit = np.reshape(together,(-1,2))
grouping = pd.DataFrame({'artist_ID':together_edit[:,0], 'plays': together_edit[:,1]})
pd.to_numeric(grouping['plays'])
grouped = grouping.groupby('artist_ID').apply(np.median)
#for i in 
#grouping = matplotlib.mlab.rec_groupby(grouping,X,)
#%%
artist_median={}
for i in grouping.artist_ID.unique():
    count = 0
    a = np.median(grouping[grouping['artist_ID']==i]['plays'])
    artist_median[i] = a
#Remove a bunch of stuff
del(plays_array, train_data)
del(together)
del(together_edit)
del(grouping)
#%% Append appropriate X values to the Y
correct_X=[]
for i in range(len(X)):
    a = user_medians[X[i,0]]
    b = artist_median[X[i,1]]
    correct_X.append([a,b,float(Y[i])])
#%%
correct_X = np.array(correct_X)
#%% run regression-- order is user mean, artist mean, play number
X_train,X_val,Y_train,Y_val = train_test_split(correct_X[:,:-1], correct_X[:,-1],test_size=0.2)
test = linear_model.LinearRegression(n_jobs=-1)
test.fit(X_train,Y_train)
print test.score(X_val,Y_val)
print mae(Y_train,test.predict(X_train))
print test.coef_
print test.intercept_
#%% Testing another regression
X_train2,X_val2,Y_train2,Y_val2 = train_test_split(correct_X[:,:-1], np.log(correct_X[:,-1]),test_size=0.2)
test2 = linear_model.LinearRegression(n_jobs=-1)
test2.fit(X_train2,Y_train2)
test2.score(X_val2,Y_val2)
mae(np.exp(Y_train2),np.exp(test2.predict(X_train2)))
test2.coef_
test2.intercept_
#a = test.predict(X_train)

#%%

#%%
# Write out test solutions.
c=0
with open(test_file, 'r') as test_fh:
    test_csv = csv.reader(test_fh, delimiter=',', quotechar='"')
    next(test_csv, None)

    with open(soln_file, 'wb') as soln_fh:
        soln_csv = csv.writer(soln_fh,
                              delimiter=',',
                              quotechar='"',
                              quoting=csv.QUOTE_MINIMAL)
        soln_csv.writerow(['Id', 'plays'])

        for row in test_csv:
            id     = row[0]
            user   = row[1]
            artist = row[2]
            if user in user_medians and artist in artist_median:
                answer = test.predict(np.reshape([user_medians[user],artist_median[artist]],(-1,2)))
                if answer[0]<= 0:
                    soln_csv.writerow([id, global_median])
                else:
                    soln_csv.writerow([id, answer[0]])
            else:
                print "User", id, "not in training data."
                soln_csv.writerow([id, global_median])
                
#%%
post = np.genfromtxt('submit_1.csv', delimiter=',')[1:, -1]
