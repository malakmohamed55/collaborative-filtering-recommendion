import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class UserBasedCF:
    def __init__(self, data):
        self.data = data
        self.user_item_matrix = data.pivot_table(index='user_id', columns='item_id', values='rating')
        self.user_item_filled = self.user_item_matrix.fillna(0)
        self.similarity = cosine_similarity(self.user_item_filled)
        self.sim_df = pd.DataFrame(self.similarity, index=self.user_item_matrix.index, columns=self.user_item_matrix.index)

    def predict(self, user_id, item_id, k=5):
        if item_id not in self.user_item_matrix.columns:
            return np.nan

        sims = self.sim_df[user_id].drop(user_id)
        sims = sims[sims > 0]
        rated_users = self.user_item_matrix[item_id].dropna()

        common_users = sims.index.intersection(rated_users.index)
        if len(common_users) == 0:
            return np.nan

        sims = sims.loc[common_users]
        ratings = rated_users.loc[common_users]

        top_k = sims.sort_values(ascending=False).head(k)
        top_ratings = ratings.loc[top_k.index]

        if top_k.sum() == 0:
            return np.nan

        return np.dot(top_k, top_ratings) / top_k.sum()

    def recommend(self, user_id, n=10):
        unrated_items = self.user_item_matrix.columns[self.user_item_matrix.loc[user_id].isna()]
        predictions = [(item, self.predict(user_id, item)) for item in unrated_items]
        predictions = [p for p in predictions if not np.isnan(p[1])]
        top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
        return top_n
