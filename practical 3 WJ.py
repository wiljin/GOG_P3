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
soln_file  = 'submit_2.csv'
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
    together.append([X[i,1],int(Y[i])])

#%%
together_edit = np.reshape(together,(-1,2))
grouping = pd.DataFrame({'artist_ID':together_edit[:,0], 'plays': together_edit[:,1]})
grouping['plays']=pd.to_numeric(grouping['plays'])
#grouped = grouping.groupby('artist_ID').apply(np.median)

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
print mae(Y_val,test.predict(X_val))
print test.coef_
print test.intercept_


#%% Replace the zeros -- did not help that much
X_train,X_val,Y_train,Y_val = train_test_split(correct_X[:,:-1], correct_X[:,-1],test_size=0.2)
test = linear_model.LinearRegression(n_jobs=-1)
test.fit(X_train,Y_train)
pred = test.predict(X_val)
pred[pred[:]<=0 ] = 0
print test.score(X_val,Y_val)
print mae(Y_val,pred)
print test.coef_
print test.intercept_
#%% Try out elastic net
Elastic = linear_model.ElasticNet(warm_start=True)
Elastic.fit(X_train,Y_train)
pred = Elastic.predict(X_val)
pred[pred[:]<=0 ] = 0
print Elastic.score(X_val,Y_val)
print mae(Y_val,pred)
print Elastic.coef_
print Elastic.intercept_
#%% RANSAC Regressor
X_train,X_val,Y_train,Y_val = train_test_split(correct_X[:,:-1], correct_X[:,-1],test_size=0.2)
ran = linear_model.RANSACRegressor()
ran.fit(X_train,Y_train)
#No 0 values in predict
#pred = ran.predict(X_val)
print ran.score(X_val,Y_val)
print mae(Y_val, ran.predict(X_val))
print mae(Y_val,X_val[:,0])

#%% Try random weightings
predY_val=[]
smallest_MAE=9999999999999
best_weight = 2
for i in np.arange(0,1.1,0.05):
    for j in range(len(X_val)):
        predY_val.append(i*X_val[j,0]+(1-i)*X_val[j,1])
    a = mae(Y_val,predY_val)
    if a<smallest_MAE:
        smallest_MAE = a
        best_weight = i
    predY_val=[]
    print ("For weight : %f the MAE was %f"%(i,round(a,2)))
print best_weight
print smallest_MAE

#%%
# Write out test solutions.
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
                answer = ran.predict(np.reshape([user_medians[user],artist_median[artist]],(-1,2)))
                if answer[0]<= 0:
                    soln_csv.writerow([id, 0])
                else:
                    soln_csv.writerow([id, answer[0]])
            else:
                print "User", id, "not in training data."
                soln_csv.writerow([id, global_median])
                
#%%
post = np.genfromtxt('submit_1.csv', delimiter=',')[1:, -1]
