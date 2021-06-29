import pandas as pd
import csv
import conf
import numpy as np

def transform_sparse_to_dense_data(sparse_df):
    transformed_data = []
    print(sparse_df)
    for uid in range(len(sparse_df)):
        for item_id in sparse_df.columns:
            if (item_id != 'user'): 
                rating = sparse_df.iloc[uid][item_id]
                #print(f"user: {uid} - item: {item_id} - rating: {rating}")
                if (rating != 0) & (item_id != 'user'):
                    user_id = int(uid)+1
                    #print(f'rating: {rating}')
                    sample = [user_id, int(item_id), int(rating)]
                    transformed_data.append(sample)
                
    df = pd.DataFrame(transformed_data, columns = ['user', 'item', 'rating'])
    return df

def transform_dense_to_sparse_data(ratings):
    user_item_matrix = ratings.pivot(*ratings.columns)
    user_item_matrix = user_item_matrix.fillna(0)
    #user_item_matrix.columns = user_item_matrix.columns.astype(str)
    print(user_item_matrix)
    return user_item_matrix

def transform_dense_to_sparse_data_amazon(ratings):
    users = set(ratings['user'])
    items = set(ratings['item'])
    df = pd.DataFrame(np.nan, index=users, columns=items)
    print(df)
    pass
    

def dense_to_csv(df, cd, epochs, bs):
    df.to_csv(f'{conf.SYN_DATA_DIR}syn_dense_{cd}_{epochs}eps_{bs}bs.csv')

