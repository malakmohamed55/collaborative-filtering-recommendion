# Collaborative Filtering Recommendation System

A project implementing User-Based Collaborative Filtering using MovieLens 100K dataset.

## Structure

- `data/` - MovieLens dataset files
- `notebooks/` - Step-by-step Jupyter Notebooks
- `src/` - Source code (data loader, recommender, evaluation)
- `results/` - Evaluation metrics and recommendations

## Usage

```python
from src.data_loader import load_data
from src.recommender import UserBasedCF

data = load_data("data/")
model = UserBasedCF(data)
model.recommend(user_id=1)
