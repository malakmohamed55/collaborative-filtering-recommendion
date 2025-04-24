from sklearn.metrics import mean_squared_error
import numpy as np

def compute_rmse(true_ratings, predicted_ratings):
    mse = mean_squared_error(true_ratings, predicted_ratings)
    return np.sqrt(mse)

def precision_at_k(recommendations, actual_ratings, k=10, threshold=3.5):
    hits = 0
    for item_id, predicted_rating in recommendations[:k]:
        actual = actual_ratings.get(item_id, 0)
        if actual >= threshold:
            hits += 1
    return hits / k if k else 0
