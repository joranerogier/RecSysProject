import pandas as pd

# import own scripts
import conf

explicit_data = pd.read_csv(f'{conf.DATA_DIR}orig_sparse_beyond_mainstream_tau_0.18_playcounts_impl60_lastfm_items_removed_implicit.csv')

implicit_data = explicit_data.copy(deep=True)

print(explicit_data)
for uid in range(len(explicit_data)):
    for item_id in explicit_data.columns:
        rating = explicit_data.iloc[uid][item_id]
        if (item_id != 'user') & (rating > 0):
            implicit_data[item_id][uid] = 1
            #print(f"user: {uid} - item: {item_id} - rating: {rating}")

print(implicit_data)
implicit_data.to_csv(f'{conf.DATA_DIR}orig_sparse_beyond_mainstream_tau_0.18_playcounts_impl60_lastfm_items_removed_implicit_true.csv', index=False)
