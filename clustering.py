import numpy as np
import csv
from sklearn.cluster import KMeans
from kmodes import kprototypes
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
        
        profile_data[user] = [sex, age, country]

with open(artists, 'r') as artist_fh:
    artists_csv = csv.reader(artist_fh, delimiter=',', quotechar='"')
    next(artists_csv, None)
    for row in artists_csv:
        artist = row[0]
        name = row[1]

        profile_data[artist] = [name]

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
        
        # Get other user/artist data.
        if user in profile_data:
            user_data = profile_data[user]
        else:
            user_data = ["", "", ""]
        if artist in profile_data:
            artist_data = profile_data[artist]
        else:
            artist_data = [""]
                
        train_data[user][artist] =  user_data + artist_data + [int(plays)]
#%%
# Make training matrix.
training_matrix = []
for user in train_data:
    for artist in train_data[user]:
        training_matrix.append([user, artist] + train_data[user][artist])

#reg = KMeans(n_clusters = 10, n_init = 3, n_jobs = -1)
#reg.fit(training_matrix[:-1],training_matrix[-1])
#reg.fit(training_matrix)
#reg.fit(training_matrix[:-1],training_matrix[-1])
#%%
train_num = np.asarray(training_matrix,dtype=object)
reg = kprototypes.KPrototypes(n_clusters = 8, init='Cao')
reg.fit(train_num[:,:-1],y=train_num[:,-1],categorical =[0,1,2,4,5] )
#%%
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
                
