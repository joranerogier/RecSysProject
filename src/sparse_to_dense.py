import pandas as pd
import csv
import conf

def transform_sparse_to_dense_data(sparse_df):
    transformed_data = []
    for uid in range(len(sparse_df)):
        for item_id in sparse_df.columns:
            rating = sparse_df.iloc[uid][item_id]
            #print(f"user: {uid} - item: {item_id} - rating: {rating}")
            if rating != "":
                user_id = int(uid)+1
                sample = [user_id, int(item_id), int(rating)]
                transformed_data.append(sample)
            
    df = pd.DataFrame(transformed_data, columns =['user', 'item', 'rating'])
    return df

def dense_to_csv(df, cd, epochs, bs):
    df.to_csv(f'{conf.SYN_DATA_DIR}syn_dense_{cd}_{epochs}eps_{bs}bs.csv')

