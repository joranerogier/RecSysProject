
def transform_dense_to_sparse_data(ratings):
    user_item_matrix = ratings.pivot(*ratings.columns)
    user_item_matrix = user_item_matrix.fillna("")
    user_item_matrix.columns = user_item_matrix.columns.astype(str)
    return user_item_matrix 

