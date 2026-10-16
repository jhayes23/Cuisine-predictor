# Cuisine Predictor

Predicts a recipe's cuisine (Italian, Mexican, Indian, and 17 others) from its list of ingredients, using Word2Vec ingredient embeddings and a tuned logistic regression classifier.

**Result:** 74.8% accuracy on a held-out validation set, nearly 4x the 19.7% you'd get by always guessing the most common cuisine.

## Dataset

The data is from Kaggle's [What's Cooking?](https://www.kaggle.com/competitions/whats-cooking) competition.

| File | Recipes | Contents |
| --- | --- | --- |
| `train.json` | 39,774 | Recipe ID, cuisine label, ingredient list |
| `test.json` | 9,944 | Recipe ID, ingredient list (no labels) |

There are 20 cuisines, and the classes are imbalanced. Italian is the largest at 19.7% of recipes.

## Approach

1. **Exploratory analysis.** Plotted ingredient counts per recipe, the most common ingredients, and average recipe size by cuisine. Common ingredients like salt appear everywhere, so they carry little signal about cuisine.
2. **Ingredient embeddings.** Trained a Gensim Word2Vec model on the ingredient lists, treating each ingredient as a word (350 dimensions, window 10, min count 2, 20 epochs). Similar ingredients end up close together: the nearest neighbors of "meat" are beef, pork, and ground beef.
3. **Recipe features.** Represented each recipe as the average of its ingredient vectors.
4. **Model.** Logistic regression inside a scikit-learn pipeline with `StandardScaler`, so the features are scaled within each training fold.
5. **Tuning.** Held out a stratified 20% of the labeled data for validation. On the remaining 80%, chose the regularization strength `C` with 3-fold cross-validation, then scored the best model once on the validation set.

## Results

| C | Mean CV accuracy |
| --- | --- |
| 0.01 | 72.8% |
| 0.1 | 73.1% |
| 1 | 73.7% |
| **10** | **74.0%** |

| Metric | Accuracy |
| --- | --- |
| Validation accuracy (C = 10) | **74.8%** |
| Majority-class baseline | 19.7% |

The tuned model writes its predictions for the unlabeled test set to `clf_output.csv`.

## What I learned

- **Check convergence.** The first version of this model never finished training. A convergence warning was hidden by a blanket `warnings.filterwarnings('ignore')`. Scaling the features and raising `max_iter` fixed it.
- **Keep the evaluation honest.** `C` was chosen with cross-validation on training data only, and the validation set was scored once, so the 74.8% is not tuned to the validation set.
- **Known limitation.** The Word2Vec model was trained on all labeled recipes, including the validation split. It never sees cuisine labels, so the effect should be small, but a stricter setup would train the embeddings on the training split only.

## Next steps

- Compare against TF-IDF features over the ingredient lists, a strong baseline for this kind of data.
- Look at per-cuisine accuracy and the confusion matrix to see which cuisines get mixed up.

## Running it

```bash
pip install gensim numpy pandas matplotlib scikit-learn jupyter
jupyter notebook cuisine_predictor.ipynb
```

Run all cells top to bottom. The grid search takes several minutes on a laptop.

## Tools

Python, pandas, NumPy, Gensim (Word2Vec), scikit-learn, Matplotlib
