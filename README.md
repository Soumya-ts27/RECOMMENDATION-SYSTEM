# RECOMMENDATION-SYSTEM

**COMPANY**: CODETECH IT SOLUTIONS

**NAME**: TARIMELA SRINIVASA SOUMYA

**INTERN ID**:CT12WJVV

**DOMAIN**: MACHINE LEARNING

**BATCH DURATION**: JANUARY 5th,2025 to APRIL 5th,2025

**MENTOR NAME**: NEELA SANTHOSH

This Python program implements a movie recommendation system using the **Surprise** library. It employs **Singular Value Decomposition (SVD)** for matrix factorization, which is a popular technique for collaborative filtering. The goal of the program is to predict user ratings for movies (or items) based on past interactions.

---

## Step 1: Importing Libraries

The following libraries are imported:
- **Pandas and NumPy**: For data manipulation and numerical operations.
- **Scikit-Learn**: For splitting data using `train_test_split`.
- **Surprise**: Specifically designed for building and analyzing recommendation systems. It provides tools for loading data, training models, performing cross-validation, and making predictions.

---

## Step 2: Loading the Dataset

A sample dataset is created using a Python dictionary with three columns:
- **user_id**: Represents individual users (1 to 5).
- **item_id**: Represents items (e.g., movies) that users have rated.
- **rating**: Represents users’ ratings for items on a scale of 1 to 5.

The dataset is converted into a Pandas DataFrame using:
```python
df = pd.DataFrame(data)
```

This is a small simulated dataset, but in real-world scenarios, larger datasets such as **MovieLens** or custom datasets can be used.

---

## Step 3: Defining the Reader

The **Surprise** library requires a specific format to load datasets.  
The `Reader` class defines the expected rating scale, which in this case ranges from **1 to 5**:
```python
reader = Reader(rating_scale=(1, 5))
```

The data is then loaded using:
```python
data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)
```
This prepares the dataset for model training and evaluation.

---

## Step 4: Splitting the Dataset

The program splits the data into training and testing sets using `train_test_split` from **Scikit-Learn**:
```python
trainset, testset = train_test_split(df, test_size=0.2, random_state=42)
```
- **Training Set**: Used to train the recommendation model.
- **Testing Set**: Used to evaluate the model’s accuracy by comparing predicted ratings to actual ratings.

---

## Step 5: Model Building using SVD

The program uses **Singular Value Decomposition (SVD)**, which is a matrix factorization technique. SVD decomposes the user-item rating matrix into three matrices to extract latent factors representing users and items. It is effective for discovering patterns in sparse datasets.

The model is initialized using:
```python
model = SVD()
```

---

## Step 6: Cross-Validation

To evaluate the model’s performance, **5-fold cross-validation** is applied using `cross_validate()`:
```python
cross_validate(model, data, cv=5, verbose=True)
```
- **cv=5**: Splits the data into five subsets, using four for training and one for testing, iteratively.
- **verbose=True**: Prints evaluation metrics like RMSE (Root Mean Square Error) and MAE (Mean Absolute Error).

---

## Step 7: Model Training on Full Data

After cross-validation, the model is trained on the entire dataset using:
```python
trainset = data.build_full_trainset()
model.fit(trainset)
```
This allows the model to learn from all available data, maximizing its predictive accuracy.

---

## Step 8: Making Predictions

The program predicts a rating for a specific user and item using:
```python
user_id = 1
item_id = 104
pred = model.predict(user_id, item_id)
```
Here:
- **User 1** is requesting a prediction for **Item 104** (a movie they haven’t rated before).
- `model.predict()` uses the learned SVD model to estimate a rating.

The predicted rating is printed using:
```python
print(f"Predicted rating for User {user_id} on Item {item_id}: {pred.est:.2f}")
```
The output provides the estimated rating with two decimal places.

---

## Conclusion

This program demonstrates how to build a simple and effective recommendation system using collaborative filtering with SVD. By analyzing historical user-item interactions, the model can predict future ratings, helping recommend items users are likely to enjoy.

### Advantages:
- SVD is efficient for sparse datasets.
- Cross-validation ensures robust evaluation.
- Predictions are made using a scalable and interpretable approach.

### Limitations:
- The model assumes user preferences are linear, which may not always be the case.
- Cold start problems may occur for new users or items with no ratings.

To improve the model, hyperparameter tuning or hybrid recommendation approaches (combining collaborative filtering with content-based filtering) can be applied.

**OUTPUT**:![Image](https://github.com/user-attachments/assets/00c4b2e3-a534-4879-85d8-7fee7ba98209)
