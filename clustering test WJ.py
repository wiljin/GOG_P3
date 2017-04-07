import numpy as np
import csv
from sklearn.cluster import KMeans
from kmodes import kprototypes
from sklearn.cross_validation import train_test_split

#pip install kmodes
# Predict via the user-specific median.
# If the user has no data, use the global median.

profile = "profiles.csv"
train_file = 'train.csv'
test_file  = 'test.csv'
soln_file  = 'user_median.csv'
artists = "artists.csv"

# Load the profile data-- next is to skip headers
profile_data = {}
with open(profile, 'r') as profile_fh:
    profile_csv = csv.reader(profile_fh, delimiter=',', quotechar='"')
    next(profile_csv, None)
    for row in profile_csv:
        user   = row[0]
        sex = row[1]
        age  = row[2]
        country = row[3]
        if age == '':
            age = 'z'
        elif int(age)< 10 & int(age)>70:
            age = 'z'
        if sex =='':
            sex == 'z'
        if country == '':
            country == 'z'
        profile_data[user] = [sex, age, country]
        #if age == 'z':
            #print profile_data[user]

with open(artists, 'r') as artist_fh:
    artists_csv = csv.reader(artist_fh, delimiter=',', quotechar='"')
    next(artists_csv, None)
    for row in artists_csv:
        artist = row[0]
        name = row[1]

        profile_data[artist] = [name]
#%%
X = np.genfromtxt(train_file, dtype=object, delimiter=',')[1:, :-1]
Y = np.genfromtxt(train_file, dtype=object, delimiter=',')[1:, -1]
#%%
#Right now X is of the format User,artist 
X = np.append(X,np.full((len(X),4),np.nan),axis=1)
#%%
for i in xrange(len(X)):
    for j in range(2,5):   
     #This should add gender, age, country
     X[i,j] = profile_data[X[i,0]][j-2]
     X[i,5] = profile_data[X[i,1]][0]
#%%
#Fix ages
a= np.mean(X[(X[:,3]!='z')&(X[:,3]!='')][:,3].astype(int))
#%%
for i in np.where( X[:,3] == 'z')[0]:
    X[i,3] = 24
#%%
#np.save('X',X)
#np.save('Y',Y)
#%%
# Make training matrix.
#training_matrix = []
#for user in train_data:
#    for artist in train_data[user]:
#        training_matrix.append([user, artist] + train_data[user][artist])

#reg = KMeans(n_clusters = 10, n_init = 3, n_jobs = -1)
#reg.fit(training_matrix[:-1],training_matrix[-1])
#reg.fit(training_matrix)
#reg.fit(training_matrix[:-1],training_matrix[-1])
#%%
X_train,X_val,Y_train,Y_val = train_test_split(X, Y,test_size=0.9)
reg = kprototypes.KPrototypes(n_clusters = 8, init='Cao')
reg.fit(X_train,y=Y_train,categorical =[0,1,2,4,5] )
#%% Test out vs user mean
for i in X_val
# Write out test solutions.
with open(test_file, 'r') as test_fh:
    test_csv = csv.reader(test_fh, delimiter=',', quotechar='"')
    next(test_csv, None)

    with open(soln_file, 'w') as soln_fh:
        soln_csv = csv.writer(soln_fh,
                              delimiter=',',
                              quotechar='"',
                              quoting=csv.QUOTE_MINIMAL)
        soln_csv.writerow(['Id', 'plays'])

        for row in test_csv:
            id     = row[0]
            user   = row[1]
            artist = row[2]

            if user in user_medians:
                soln_csv.writerow([id, user_medians[user]])
            else:
                print "User", id, "not in training data."
                soln_csv.writerow([id, global_median])
                
