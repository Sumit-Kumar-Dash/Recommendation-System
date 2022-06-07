import pandas as pd
import numpy as np

# Commented out IPython magic to ensure Python compatibility.
# import dependent libraries
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
import warnings

from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k
from lightfm import LightFM
#from skopt import forest_minimize

movies_df = pd.read_csv('movies.csv',usecols=['movieId','title'],dtype={'movieId': 'int32', 'title': 'str'})
rating_df = pd.read_csv('ratings.csv',usecols=['userId', 'movieId', 'rating'],dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

df = pd.merge(rating_df,movies_df,on='movieId')

df.replace('', np.nan, inplace=True)


combine_movie_rating = df.dropna(axis = 0, subset = ['title'])
movie_ratingCount = (combine_movie_rating.
     groupby(by = ['title'])['rating'].
     count().
     reset_index().
     rename(columns = {'rating': 'totalRatingCount'})
     [['title', 'totalRatingCount']]
    )

rating_with_totalRatingCount = combine_movie_rating.merge(movie_ratingCount, left_on = 'title', right_on = 'title', how = 'left')

popularity_threshold = 50
rating_popular_movie= rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')


movie_features_df=rating_popular_movie.pivot_table(index='userId',columns='title',values='rating').fillna(0)

from scipy.sparse import csr_matrix
from scipy import sparse
movie_features_df_matrix = csr_matrix(movie_features_df.values)

model = LightFM(loss='warp')
'''random_state=2016,
                learning_rate=0.90,
                no_components=150,
                user_alpha=0.000005)'''

model = model.fit(movie_features_df_matrix)
'''epochs=100,
                  num_threads=16, verbose=False)'''

def sample_recommendation(model, data, user_ids):
  n_users ,n_items  = movie_features_df_matrix.shape
  for user_id in user_ids:
    known_positives = movie_features_df.columns[movie_features_df_matrix.tocsr()[user_id].indices]
    scores = model.predict(user_id, np.arange(n_items))
    top_items = movie_features_df.columns[np.argsort(-scores)]
    #print results
    print("\nUser %s" % user_id)
    print("Most Liked:")

    for x in known_positives[:3]:
        print("%s" % x)
    
    print("Recommend:")

    for x in top_items[:3]:
        print("%s" % x)

res = sample_recommendation(model, movie_features_df_matrix, [0, 4, 25, 9])
print(res)

