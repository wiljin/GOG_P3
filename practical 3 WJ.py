import numpy as np
import csv

# Predict via the user-specific median.
# If the user has no data, use the global median.

profile = "profiles.csv"
train_file = 'train.csv'
test_file  = 'test.csv'
soln_file  = 'user_median.csv'
artists = "artists.csv"
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
    
        if not user in profile_data:
            profile_data[user] = {}
        
        profile_data[user][sex] = sex
        profile_data[user][age] = age
        profile_data[user][country] = country

artists_data = {}
with open(artists, 'r') as artist_fh:
    artists_csv = csv.reader(artist_fh, delimiter=',', quotechar='"')
    next(artists_csv, None)
    for row in artists_csv:
        artist   = row[0]
        name = row[1]

        
        profile_data[artist] = name

# Compute the global median and per-user median.
plays_array  = []
user_medians = {}
for user, user_data in train_data.iteritems():
    user_plays = []
    for artist, plays in user_data.iteritems():
        plays_array.append(plays)
        user_plays.append(plays)

    user_medians[user] = np.median(np.array(user_plays))
global_median = np.median(np.array(plays_array))

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
                
