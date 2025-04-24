import pandas as pd

def load_data(path='data/'):
    ratings = pd.read_csv(f"{path}u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
    movies = pd.read_csv(f"{path}u.item", sep="|", encoding="latin-1", names=[
        "item_id", "title", "release_date", "video_release_date", "IMDb_URL",
        "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
        "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ])
    users = pd.read_csv(f"{path}u.user", sep="|", names=["user_id", "age", "gender", "occupation", "zip_code"])
    
    ratings = ratings.merge(movies[["item_id", "title"]], on="item_id")
    data = ratings.merge(users, on="user_id")
    
    return data
