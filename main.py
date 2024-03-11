# %% [markdown]
# # Spotify Song Prediction

# %% [markdown]
# ##### Dependencies

# %%
# Ucitavanje biblioteka i funkcija
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# %% [markdown]
# ##### Loading the dataset

# %%
df = pd.read_csv('spotify_songs.csv')
df.head()

# %% [markdown]
# 

# %% [markdown]
# ##### Dropping invalid and missing values

# %%
columns_to_drop = ['track_id', 'track_artist', 'track_name', 'track_album_id', 'track_album_name', 'playlist_name', 'playlist_id']
df.drop(columns=columns_to_drop, inplace=True)
df.dropna(axis=0, inplace=True)

# df = df[df['track_popularity'] < 15]

rows_to_drop = df.loc[df['duration_ms'] < 20000].index
df.drop(rows_to_drop, inplace=True)

new_df = df.copy()

# Assuming 'date' is a string column in the format 'YYYY-MM-DD'
new_df[['year', 'month', 'day']] = new_df['track_album_release_date'].str.split('-', expand=True)

# Convert the columns to numeric (to handle missing values during median calculation)
new_df['year'] = pd.to_numeric(new_df['year'], errors='coerce', downcast='integer')
# new_df['month'] = pd.to_numeric(new_df['month'], errors='coerce', downcast='integer')
# new_df['day'] = pd.to_numeric(new_df['day'], errors='coerce', downcast='integer')

# Calculate median values for year, month, and day
median_year = int(new_df['year'].median())
# median_month = int(new_df['month'].median())
# median_day = int(new_df['day'].median())

# Fill missing values with median values
new_df['year'] = new_df['year'].fillna(median_year).astype(int)
# new_df['month'] = new_df['month'].fillna(median_month).astype(int)
# new_df['day'] = new_df['day'].fillna(median_day).astype(int)

X = new_df.drop(['track_popularity', 'track_album_release_date', 'year', 'month', 'day'], axis=1)
# X.columns

X = pd.get_dummies(X, dtype=int)
X['year'] = new_df['year']
y = new_df['track_popularity']


# %%
# X.tail()
X.columns

# %%
df.iloc[3446]
X.iloc[3446]

# %% [markdown]
# ##### Splitting the dataset

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# %%
plt.hist(y, bins=25)

# %%
X.describe()

# %%
print(X.columns)

numeric_feats = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
dummy_feats = [feat for feat in X.columns if feat not in numeric_feats]

print(X[numeric_feats])

# %%
numeric_values = df[numeric_feats]
correlation_matrix = numeric_values.corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

# %%
def model_evaluation(y_test, y_predicted, N, d):
    mse = mean_squared_error(y_test, y_predicted)
    mae = mean_absolute_error(y_test, y_predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_predicted)
    r2_adj = 1 - ((1 - r2) * (N - 1)) / (N - d - 1)

    # printing values
    print('Mean squared error: ', mse)
    print('Mean absolute error: ', mae)
    print('Root mean squared error: ', rmse)
    print('R2 score: ', r2)
    print('R2 adjusted score: ', r2_adj)

    # Uporedni prikaz nekoliko pravih i predvidjenih vrednosti
    res = pd.concat([pd.DataFrame(y_test.values),
                    pd.DataFrame(y_predicted)], axis=1)
    res.columns = ['y', 'y_pred']
    print(res.head(20))
    return mse, mae, rmse, r2, r2_adj

# %% [markdown]
# ##### Scaling the data

# %%
s = StandardScaler()
s.fit(X_train[numeric_feats])
X_train_std = s.transform(X_train[numeric_feats])
X_test_std = s.transform(X_test[numeric_feats])
X_train_std = pd.DataFrame(X_train_std)
X_test_std = pd.DataFrame(X_test_std)

X_train_std = pd.concat([X_train_std, X_train[dummy_feats].reset_index(drop=True)], axis=1)
X_test_std = pd.concat([X_test_std, X_test[dummy_feats].reset_index(drop=True)], axis=1)

X_train_std.columns = list(X.columns)
X_test_std.columns = list(X.columns)
X_train_std.head()

# %%
# folds = KFold(n_splits=5, shuffle=True, random_state=42)
# hyper_params = [{'n_features_to_select': list(range(1, 16))}]

# lm = LinearRegression()
# lm.fit(X_train_std, y_train)
# rfe = RFE(lm)

# model_cv = GridSearchCV(
#     estimator=rfe,
#     param_grid=hyper_params,
#     scoring='r2',
#     cv=folds,
#     verbose=1,
#     return_train_score=True
# )

# model_cv.fit(X_train_std, y_train)
# cv_results = pd.DataFrame(model_cv.cv_results_)
# cv_results

# %%
# # plotting cv results
# plt.figure(figsize=(16,6))

# plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])
# plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])
# plt.xlabel('number of features')
# plt.ylabel('r-squared')
# plt.title("Optimal Number of Features")
# plt.legend(['test score', 'train score'], loc='upper left')

# %% [markdown]
# Optimal number of features is 13

# %%
# final model
n_features_optimal = 13

lrm = LinearRegression()
lrm.fit(X_train, y_train)

rfe = RFE(lrm, n_features_to_select=n_features_optimal)             
rfe = rfe.fit(X_train, y_train)

# predict prices of X_test
y_pred = lrm.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("r2:", r2)
print("mse", mse)

# %%
# def standardizacija(x_train, x_test, numeric_feats, dummy_feats):
#     # Save the original order of columns
#     cols = x_train.columns
    
#     # Standardize numeric features
#     s = StandardScaler()
#     s.fit(x_train[numeric_feats])
    
#     x_train_std_numeric = s.transform(x_train[numeric_feats])
#     x_test_std_numeric = s.transform(x_test[numeric_feats])
    
#     x_train_std_numeric = pd.DataFrame(x_train_std_numeric, columns=numeric_feats)
#     x_test_std_numeric = pd.DataFrame(x_test_std_numeric, columns=numeric_feats)

#     # Concatenate standardized numeric features with dummy features
#     x_train_std = pd.concat([x_train_std_numeric, x_train[dummy_feats]], axis=1)
#     x_test_std = pd.concat([x_test_std_numeric, x_test[dummy_feats]], axis=1)

#     return x_train_std_numeric, x_test_std_numeric

# %% [markdown]
# ##### Lasso with poly features

# %%
poly = PolynomialFeatures(interaction_only=False, degree=2, include_bias=True)
X_inter_train = poly.fit_transform(X_train_std)
X_inter_test = poly.transform(X_test_std)
out_feat_names = PolynomialFeatures.get_feature_names_out(
    poly, input_features=None)

# %%
# X_train_std, X_test_std = standardizacija(X_train, X_test, numeric_feats, dummy_feats)
# poly = PolynomialFeatures(interaction_only=False, degree=3, include_bias=False)
# X_inter_train = poly.fit_transform(X_train_std)
# X_inter_test = poly.transform(X_test_std)

# %%
# Inicijalizacija modela
lm = Lasso(alpha=0.01)

#Obuka
lm.fit(X_inter_train, y_train)

# Testiranje
y_predicted = lm.predict(X_inter_test)

# Evaluacija (racunanje mera uspesnosti)
model_evaluation(y_test, y_predicted, X_inter_train.shape[0], X_inter_train.shape[1])

# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(lm.coef_)),lm.coef_)
print("koeficijenti: ", lm.coef_)

# %% [markdown]
# ##### Ridge Poly

# %%
# Inicijalizacija modela
rm = Ridge(alpha=5)

# Obuka
rm.fit(X_inter_train, y_train)

# Testiranje
y_predicted = rm.predict(X_inter_test)

# Evaluacija (racunanje mera uspesnosti)
model_evaluation(y_test, y_predicted, X_inter_train.shape[0], X_inter_train.shape[1])

# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(rm.coef_)),rm.coef_)
print("koeficijenti: ", rm.coef_)


