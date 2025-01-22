import numpy as np
import os
import pandas as pd
import joblib
from scipy.stats import randint
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from xgboost import XGBRegressor
import matplotlib
matplotlib.use('svg')
import matplotlib.pyplot as plt

RANDOM_STATE = 42

def target_encoding(target_df: pd.DataFrame, full_df: pd.DataFrame, target_feature: str, target_value: str):
    target_medians = target_df.groupby(target_feature).median()
    target_counts = target_df[target_feature].value_counts()
    global_median = target_df[target_value].median()
    full_df[f"{target_feature}_target"] = full_df[target_feature].apply(lambda x: global_median if x in target_counts[target_counts<=5].index or x =="–" or x not in target_counts.index else target_medians.loc[x][0])
    full_df.drop(labels=[target_feature], axis=1, inplace=True)

    return full_df

##Load Data and select features
features = pd.read_csv("train_dataset.csv",index_col="id")
features['fuel_type'] = features['fuel_type'].apply(lambda x: x if x !="–" else "not supported")
features['model_year'] = 2024 - features['model_year']
transmission = features['transmission'].value_counts()
features['transmission'] = features['transmission'].apply(lambda x: "other" if x in transmission[transmission<=5].index or x =="–" else x)
color = features['ext_col'].value_counts()
features['ext_col'] = features['ext_col'].apply(lambda x: "other" if x in color[color<=5].index or x =="–" else x)
color = features['int_col'].value_counts()
features['int_col'] = features['int_col'].apply(lambda x: "other" if x in color[color<=5].index or x =="–" else x)
features['accident'] = features['accident'].apply(lambda x: 0 if x == "None reported" else 1)
features.drop(labels=['clean_title'],axis=1, inplace=True)
features = target_encoding(features[['model','price']], features, 'model', 'price')
features = target_encoding(features[['engine','price']], features, 'engine', 'price')

#New features
features['model_year2'] = features['model_year'].apply(lambda x: 1 if x == 0 else x)
features['milage_per_year'] = features['milage']/features['model_year2']
features.drop(labels=['model_year2'], inplace=True, axis=1)


#One hot encoding
print(features.shape)
cat_variables = ['brand','fuel_type','transmission','ext_col', 'int_col']
features = pd.get_dummies(data = features,
                         prefix = cat_variables,
                         columns = cat_variables)

#explore features
numeric_features = ['model_year','milage','model_target','engine_target']
fig, axes  = plt.subplots(nrows=2,ncols=2)
index = 0
for row in range(axes.shape[0]):
    for col in range(axes.shape[1]):
        axes[row][col].title.set_text(numeric_features[index])
        axes[row][col].hist(features[numeric_features[index]])
        index+=1

plt.tight_layout()
fig.set_size_inches(10, 10)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.savefig("feature_distributions.svg")

fig, axes  = plt.subplots(nrows=2,ncols=2)
index = 0
for row in range(axes.shape[0]):
    for col in range(axes.shape[1]):
        axes[row][col].title.set_text(numeric_features[index])
        axes[row][col].scatter(features[numeric_features[index]],features['price'])
        index+=1

plt.tight_layout()
fig.set_size_inches(10, 10)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.savefig("feature_v_price.svg")


#Scale features
features.reset_index(inplace=True)
categorical_features = [x for x in features.columns if x not in numeric_features]#this would include the price which is not categorical but we don't want to scale this so its fine
scaler = preprocessing.RobustScaler()
scaler.fit(features[numeric_features])
scaled_features = scaler.transform(features[numeric_features])
scaled_features = pd.DataFrame(scaled_features, columns = numeric_features)
scaled_features = pd.concat([scaled_features,features[categorical_features]],axis=1)
features.set_index('id',inplace=True)
scaled_features.set_index('id',inplace=True)
scaled_features.drop(labels=['price'],axis=1, inplace=True)
print(scaled_features.head())


#save scaling params
joblib.dump(scaler, "training_set_scaling_params.pkl")

#plot scaled features
fig, axes  = plt.subplots(nrows=2,ncols=2)
index = 0
for row in range(axes.shape[0]):
    for col in range(axes.shape[1]):
        axes[row][col].title.set_text(numeric_features[index])
        axes[row][col].hist(scaled_features[numeric_features[index]])
        index+=1

plt.tight_layout()
fig.set_size_inches(10, 10)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.savefig("scaled_feature_distributions.svg")


#Hyper parameter tuning
##Subsetting features
features_to_use = pd.read_table("target_features.txt",header=None)
scaled_features = scaled_features[features_to_use[0]]

#'''
#Create training and validation sets
X_train, X_val, y_train, y_val = train_test_split(scaled_features, features['price'], train_size = 0.8, random_state = RANDOM_STATE)

param_dist = {
    'learning_rate': [0.1, 0.05, 0.01, 0.001],
    'max_depth': randint(1, 15),
    'n_estimators': randint(50, 500),
    'min_child_weight': randint(1, 15),
    'subsample' : [0.7, 0.8, 0.9, 1]
}

print("Starting random grid search...")
xgb = XGBRegressor(objective = 'reg:squarederror',random_state=RANDOM_STATE)
#random_search = RandomizedSearchCV(xgb, param_dist, cv=5, n_iter=50, n_jobs=8, scoring=make_scorer(mean_squared_error, squared=False))
random_search = RandomizedSearchCV(xgb, param_dist, cv=5, n_iter=50, n_jobs=8)
random_search.fit(X_train, y_train, eval_set = [(X_val,y_val)], verbose=False)

best_params = random_search.best_params_
print("Best parameters:", best_params)

X_train, X_val, y_train, y_val = train_test_split(scaled_features, features['price'], train_size = 0.9, random_state = RANDOM_STATE)

n = int(len(X_train)*0.9)
X_train_fit, X_train_eval, y_train_fit, y_train_eval = X_train[:n], X_train[n:], y_train[:n], y_train[n:]

xgb_model = XGBRegressor(objective = 'reg:squarederror', random_state = RANDOM_STATE, **best_params)
#xgb_model = XGBRegressor(objective = 'reg:squarederror', random_state = RANDOM_STATE,
#                        learning_rate = 0.1, max_depth = 4, min_child_weight = 14, n_estimators = 72, subsample = 0.8)
xgb_model.fit(X_train_fit,y_train_fit, eval_set = [(X_train_fit,y_train_fit),(X_train_eval,y_train_eval)], verbose=False)

results = xgb_model.evals_result()
epochs = len(results['validation_0']['rmse'])
print(f"Number of epochs {epochs}")


# plot log loss
x_axis = range(0, epochs)
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
ax.legend()
plt.ylabel('RMSE')
plt.title('XGBoost RMSE')
plt.savefig("Training_rmse_50_iter.svg")


# Plot normalized gain
gain = dict(zip(X_train_fit.columns, xgb_model.feature_importances_))
gain = dict(sorted(gain.items(), key=lambda item: item[1], reverse=True))
fig, ax = plt.subplots()
plt.bar(gain.keys(), gain.values())
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Gain')
plt.title('Feature Importance')
fig.set_size_inches(30, 10)
plt.tight_layout()
plt.savefig("Importance_normalized_gain_50_iter.svg")
pd.DataFrame.from_dict(gain,orient='index').to_csv("Gain.csv")

# Plot weight
weight = xgb_model.get_booster().get_score(importance_type= "weight")
weight = dict(sorted(weight.items(), key=lambda item: item[1], reverse=True))
fig, ax = plt.subplots()
plt.bar(weight.keys(), weight.values())
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Weight')
plt.title('Feature Importance')
fig.set_size_inches(30, 10)
plt.tight_layout()
plt.savefig("Importance_weight_50_iter.svg")
pd.DataFrame.from_dict(weight,orient='index').to_csv("weight.csv")

print(f"Metrics validation:\n\tRMSE: {mean_squared_error(y_val,xgb_model.predict(X_val), squared=False):.4f}")
print(f"Metrics train:\n\tRMSE: {mean_squared_error(y_train,xgb_model.predict(X_train), squared=False):.4f}")
print(f"Metrics validation:\n\tR2: {r2_score(y_val,xgb_model.predict(X_val)):.4f}")
print(f"Metrics train:\n\tR2: {r2_score(y_train,xgb_model.predict(X_train)):.4f}")


#Save model
joblib.dump(xgb_model, "50_iter_model_Final.pkl")
xgb_model.save_model("50_iter_model_Final.json")


#Cross validation
#xgb_model = XGBRegressor(objective = 'reg:squarederror', random_state = RANDOM_STATE,
#                        learning_rate = 0.1, max_depth = 4, min_child_weight = 14, n_estimators = 72, subsample = 0.8)
xgb_model = XGBRegressor(objective = 'reg:squarederror', random_state = RANDOM_STATE, **best_params)

#xgb_model = XGBRegressor()
scores = cross_val_score(xgb_model, scaled_features, features['price'], cv=5, scoring=make_scorer(mean_squared_error, squared=False))
print(f"Cross validate RMSE {', '.join([str(s) for s in scores])}")
print("%0.2f RMSE with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

scores = cross_val_score(xgb_model, scaled_features, features['price'], cv=5)
print(f"Cross validate R2 {', '.join([str(s) for s in scores])}")
print("%0.2f R2 with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
#'''