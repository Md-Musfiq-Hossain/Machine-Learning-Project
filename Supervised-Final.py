#!/usr/bin/env python
# coding: utf-8

# ## Install Dependencies

# ### Importing all the dependencies

# In[ ]:


get_ipython().system('pip install pandas seaborn matplotlib numpy scipy scikit-learn xgboost scikit-optimize bayesian-optimization')


# In[40]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import uniform, randint, loguniform
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, root_mean_squared_error, mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from bayes_opt import BayesianOptimization
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import gc


# # Regression for Thermal Stability Dataset

# ### Loading dataset from Github [It may take some time]

# In[3]:


#https://raw.githubusercontent.com/Shown246/CSE445_Datasets/refs/heads/main/thermal_stability_cleaned.csv
df = pd.read_csv('https://raw.githubusercontent.com/Shown246/CSE445_Datasets/refs/heads/main/thermal_stability_cleaned.csv')
df.head()


# #### Splitting Data in Train and Test

# In[4]:


# Separate features (X) and target (y)
X = df.drop(columns=["thermal_tuning_efficiency"])  # Replace "target_column" with the actual column name
y = df["thermal_tuning_efficiency"]

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)


# ## Linear Regression

# ### Train Linear Regration (Baseline Model)

# In[ ]:


# Initialize the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)

# Evaluation
print("\nLinear Regression Baseline Performance:")
print("MAE:", mean_absolute_error(y_test, linear_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, linear_pred)))
print("R² Score:", r2_score(y_test, linear_pred))


# ### Grid Search CV

# In[ ]:


ridge = Ridge()
# Grid Search
ridge_grid_params = {'alpha': [0.1, 1.0, 10.0, 100.0]}
ridge_grid = GridSearchCV(ridge, ridge_grid_params, cv=5, scoring='neg_mean_squared_error')
ridge_grid.fit(X_train, y_train)
ridge_grid_best = ridge_grid.best_estimator_
ridge_grid_mse = mean_squared_error(y_test, ridge_grid_best.predict(X_test))
ridge_grid_r2 = r2_score(y_test, ridge_grid_best.predict(X_test))

# Print Grid Search hyperparameters and scores
print("\nRidge Grid Search Hyperparameters and Scores:")
ridge_grid_results = pd.DataFrame(ridge_grid.cv_results_)[['param_alpha', 'mean_test_score', 'std_test_score']]
ridge_grid_results['mean_test_score'] = -ridge_grid_results['mean_test_score']  # Convert to positive MSE
print(ridge_grid_results)


# ### Random Search CV

# In[ ]:


# Random Search
ridge_random_params = {'alpha': np.logspace(-2, 2, 20)}
ridge_random = RandomizedSearchCV(ridge, ridge_random_params, n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
ridge_random.fit(X_train, y_train)
ridge_random_best = ridge_random.best_estimator_
ridge_random_mse = mean_squared_error(y_test, ridge_random_best.predict(X_test))
ridge_random_r2 = r2_score(y_test, ridge_random_best.predict(X_test))

# Print Random Search hyperparameters and scores
print("\nRidge Random Search Hyperparameters and Scores:")
ridge_random_results = pd.DataFrame(ridge_random.cv_results_)[['param_alpha', 'mean_test_score', 'std_test_score']]
ridge_random_results['mean_test_score'] = -ridge_random_results['mean_test_score']  # Convert to positive MSE
print(ridge_random_results)


# ### Bayesian Optimization

# In[ ]:


# Define the cross-validation objective function
def ridge_cv(alpha, tol, max_iter):
    model = Ridge(
        alpha=alpha,
        tol=tol,
        max_iter=int(max_iter),
        solver='saga',  # Using saga to support max_iter and tol
        random_state=42
    )
    # Use negative MSE since BayesianOptimization maximizes the score
    scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
    return scores.mean()

# Define the search space for hyperparameters
pbounds = {
    'alpha': (0.01, 100),
    'tol': (1e-6, 1e-2),
    'max_iter': (100, 1000)
}

# Initialize Bayesian Optimization
optimizer = BayesianOptimization(
    f=ridge_cv,
    pbounds=pbounds,
    random_state=42,
)

# Run optimization: 2 initial random points, 8 iterations = 10 total
optimizer.maximize(init_points=2, n_iter=8)

# Get best hyperparameters
best_params = optimizer.max['params']
best_alpha = best_params['alpha']
best_tol = best_params['tol']
best_max_iter = int(best_params['max_iter'])

# Train final Ridge model with best params
ridge_bayes_best = Ridge(
    alpha=best_alpha,
    tol=best_tol,
    max_iter=best_max_iter,
    solver='saga',
    random_state=42
)
ridge_bayes_best.fit(X_train, y_train)

# Evaluate on test data
ridge_bayes_mse = mean_squared_error(y_test, ridge_bayes_best.predict(X_test))
ridge_bayes_r2 = r2_score(y_test, ridge_bayes_best.predict(X_test))

# Print results
print("\n🔍 Ridge Bayesian Optimization Hyperparameters and Scores:")
print(f"Best parameters: {best_params}")
print(f"Best tol: {best_tol:.6f}")
print(f"Best max_iter: {best_max_iter}")
print(f"MSE on test set: {ridge_bayes_mse:.4f}")
print(f"R² score on test set: {ridge_bayes_r2:.4f}")

# Optional: Show all search results
results_table = pd.DataFrame(optimizer.res)
results_table['MSE (CV)'] = -results_table['target']  # Convert neg MSE back to positive
print("\n📋 Bayesian Optimization Search History:")
print(results_table[['params', 'MSE (CV)']])


# ### Results Comparison

# In[ ]:


# Initialize the Linear Regression model as baseline
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
baseline_preds = linear_model.predict(X_test)
baseline_mse = mean_squared_error(y_test, baseline_preds)
baseline_r2 = r2_score(y_test, baseline_preds)


# Create the results DataFrame
results = pd.DataFrame({
    'Model': ['Ridge (Grid)', 'Ridge (Random)', 'Ridge (Bayes)', 'Baseline (Linear Regression)'],
    'Best Params': [ridge_grid.best_params_, ridge_random.best_params_, optimizer.max['params'], 'N/A'], # Use optimizer.max['params'] instead of ridge_bayes_best.best_params_
    'MSE': [ridge_grid_mse, ridge_random_mse, ridge_bayes_mse, baseline_mse],
    'R2': [ridge_grid_r2, ridge_random_r2, ridge_bayes_r2, baseline_r2]
})

# Style the table
print("\nPerformance Metrics:")
styled_results = results.style.format({
    'MSE': '{:.4f}',  # Format MSE to 4 decimal places
    'R2': '{:.4f}'    # Format R2 to 4 decimal places
}).set_properties(**{
    'text-align': 'center',
    'border': '1px solid black',
    'padding': '5px'
}).set_table_styles([
    {'selector': 'th', 'props': [('font-weight', 'bold'), ('text-align', 'center'), ('background-color', '#f2f2f2')]}
])

display(styled_results)


# In[ ]:


# Plotting the results
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='MSE', data=results, palette='viridis')
plt.title('Model Performance Comparison (MSE)')
plt.xticks(rotation=45)
plt.ylabel('Mean Squared Error (MSE)')
plt.xlabel('Model')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
# Plotting the results
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='R2', data=results, palette='viridis')
plt.title('Model Performance Comparison (R2 Score)')
plt.xticks(rotation=45)
plt.ylabel('R2 Score')
plt.xlabel('Model')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# ## Random Forrest

# ### Train Random Forest (Baseline Model)

# In[ ]:


default_model = RandomForestRegressor(random_state=42)
default_model.fit(X_train, y_train)
default_pred = default_model.predict(X_test)
# Evaluation
print("\nDefault Model Performance:")
print("MAE:", mean_absolute_error(y_test, default_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, default_pred)))
print("R² Score:", r2_score(y_test, default_pred))


# ### Grid Search CV

# In[ ]:


param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10.0, 20.0],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
print("Performing Grid Search...")
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, n_jobs=1)
grid_search.fit(X_train, y_train)

# Get best model
best_grid_model = grid_search.best_estimator_
grid_pred = best_grid_model.predict(X_test)

# Evaluate model performance
print("\nGrid Search Best Model Performance:")
print("Best Parameters:", grid_search.best_params_)
print("MAE:", mean_absolute_error(y_test, grid_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, grid_pred)))
print("R² Score:", r2_score(y_test, grid_pred))


# ### Random Search CV

# In[ ]:


param_dist = {
    'n_estimators': np.arange(50, 300, 50),
    'max_depth': [10, 20, 30],
    'min_samples_split': np.arange(2, 11, 2),
    'min_samples_leaf': np.arange(1, 5)
}
print("Performing Random Search...")
random_search = RandomizedSearchCV(RandomForestRegressor(random_state=42), param_dist, n_iter=10, cv=3, n_jobs=1)
random_search.fit(X_train, y_train)

# Get best model
best_random_model = random_search.best_estimator_
random_pred = best_random_model.predict(X_test)

# Evaluate model performance
print("\nRandom Search Best Model Performance:")
print("Best Parameters:", random_search.best_params_)
print("MAE:", mean_absolute_error(y_test, random_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, random_pred)))
print("R² Score:", r2_score(y_test, random_pred))


# ### Bayesian Optimization

# In[ ]:


def rf_bo(n_estimators, max_depth):
    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        random_state=42
    )
    return cross_val_score(model, X_train, y_train, cv=3, n_jobs=1, scoring='r2').mean()

print("Performing Bayesian Optimization...")
bo = BayesianOptimization(rf_bo, {'n_estimators': (50, 200), 'max_depth': (10, 30)})
bo.maximize(n_iter=5)

# Get best parameters
best_bo_params = bo.max['params']
best_bo_params['n_estimators'] = int(best_bo_params['n_estimators'])
best_bo_params['max_depth'] = int(best_bo_params['max_depth'])

# Train model with best parameters
best_bo_model = RandomForestRegressor(**best_bo_params, random_state=42)
best_bo_model.fit(X_train, y_train)
bo_pred = best_bo_model.predict(X_test)

# Evaluate model performance
print("\nBayesian Optimization Best Model Performance:")
print("Best Parameters:", best_bo_params)
print("MAE:", mean_absolute_error(y_test, bo_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, bo_pred)))
print("R² Score:", r2_score(y_test, bo_pred))


# ### Compare Hyperparameter Tuning Results

# In[ ]:


results = pd.DataFrame({
    "Method": ["Base Line","Grid Search", "Random Search", "Bayesian Optimization"],
    "Best R² Score": [r2_score(y_test, default_pred),r2_score(y_test, grid_pred), r2_score(y_test, random_pred), r2_score(y_test, bo_pred)]
})
print(results)


# In[ ]:


# Plot results
plt.figure(figsize=(8,6))
sns.barplot(x=results["Method"], y=results["Best R² Score"], palette='viridis')
plt.title("Comparison of Hyperparameter Tuning Methods")
plt.ylabel("R² Score")
plt.show()


# ## XGBoost

# In[ ]:


def evaluate_model(model, X_test, y_test):
    """
    Evaluates a regression model using RMSE, MAE, and R² score.

    Parameters:
    model: Trained regression model
    X_test: Test feature set
    y_test: True target values

    Returns:
    Dictionary with RMSE, MAE, and R² score.
    """
    y_pred = model.predict(X_test)
    return {
        "RMSE": root_mean_squared_error(y_test, y_pred),  # Updated function
        "MAE": mean_absolute_error(y_test, y_pred),
        "R²": r2_score(y_test, y_pred)
    }


# In[ ]:


# Create a scorer function for RMSE
rmse_scorer = make_scorer(root_mean_squared_error, greater_is_better=False)


# ### Training XGBoost Model

# In[ ]:


model = XGBRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Root Mean Squared Error:", np.sqrt(mse))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))


# ### Hyperparameter Tuning

# ### Grid Search CV

# In[ ]:


xgb = XGBRegressor(objective="reg:squarederror", random_state=42)

param_grid = {
    "n_estimators": [50,100, 200],
    "learning_rate": [0.01, 0.1],
    "max_depth": [1, 5, 7],
    'subsample': [0.3,0.5,0.7]
}

grid_search = GridSearchCV(
    estimator=xgb,                  # The model to tune
    param_grid=param_grid,           # The parameter grid we defined
    cv=5,                            # 5-fold cross-validation
    scoring= rmse_scorer ,         
    n_jobs=1,                       
    return_train_score=True        
)

grid_search.fit(X_train, y_train)

grid_results = evaluate_model(grid_search.best_estimator_, X_test, y_test)
print(grid_results)
print("Best Parameters (GridSearchCV):", grid_search.best_params_)


# ### Random Search CV

# In[ ]:


param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.7, 0.3)
}

random_search = RandomizedSearchCV(
    estimator=xgb, # The model to tune
    param_distributions=param_dist, # The parameter distribution to sample from
    n_iter=100, # The number of parameter settings that are sampled
    cv=5, # 5-fold cross-validation
    scoring=rmse_scorer, # Lower MSE is better
    n_jobs=1, # Uses all CPU cores
    random_state=42,
    return_train_score=True # Include training scores
)
random_search.fit(X_train, y_train)

random_results = evaluate_model(random_search.best_estimator_, X_test, y_test)
print(random_results)
print("Best Parameters (RandomizedSearchCV):", random_search.best_params_)


# ### Bayesian Optimization

# In[ ]:


def xgb_cv(n_estimators, learning_rate, max_depth):
    # Cast integer params
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)

    xgb = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )

    # Use negative RMSE (because bayes_opt maximizes)
    rmse_scores = cross_val_score(
        xgb, X_train, y_train,
        scoring=make_scorer(mean_squared_error, greater_is_better=False),
        cv=2
    )

    # Return mean negative RMSE
    return rmse_scores.mean()

# Define bounds
pbounds = {
    'n_estimators': (50, 300),
    'learning_rate': (0.01, 0.2),
    'max_depth': (3, 10)
}

# Run Bayesian Optimization
optimizer = BayesianOptimization(
    f=xgb_cv,
    pbounds=pbounds,
    random_state=42
)

optimizer.maximize(init_points=3, n_iter=17)  # Total 20 like your BayesSearchCV

# Get best params
best_params = optimizer.max['params']
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['max_depth'] = int(best_params['max_depth'])

# Train final model with best hyperparameters
xgb_best = XGBRegressor(
    **best_params,
    objective='reg:squarederror',
    random_state=42,
    n_jobs=1
)

xgb_best.fit(X_train, y_train)

# Final evaluation
y_pred = xgb_best.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = xgb_best.score(X_test, y_test)

# Results
print("\n📌 Best Parameters (Bayesian Optimization):", best_params)
print(f"✅ Final RMSE: {rmse:.4f}")
print(f"✅ R² Score: {r2:.4f}")

# Optional: Show all trials
results_table = pd.DataFrame(optimizer.res)
results_table['RMSE (CV)'] = (-results_table['target'])**0.5
print("\n📋 Bayesian Optimization Search History:")
print(results_table[['params', 'RMSE (CV)']])


# ### Performance Comparison Table

# In[ ]:


# 1️⃣ GridSearchCV Best Model
best_grid = grid_search.best_estimator_
y_pred_grid = best_grid.predict(X_test)
mse_grid = mean_squared_error(y_test, y_pred_grid)
rmse_grid = np.sqrt(mse_grid)
r2_grid = r2_score(y_test, y_pred_grid)

# 2️⃣ RandomizedSearchCV Best Model
best_random = random_search.best_estimator_
y_pred_random = best_random.predict(X_test)
mse_random = mean_squared_error(y_test, y_pred_random)
rmse_random = np.sqrt(mse_random)
r2_random = r2_score(y_test, y_pred_random)

# 3️⃣ Bayesian Optimization Best Model
y_pred_optuna = xgb_best.predict(X_test)
mse_optuna = mean_squared_error(y_test, y_pred_optuna)
rmse_optuna = np.sqrt(mse_optuna)
r2_optuna = r2_score(y_test, y_pred_optuna)

#  Display results
print("\n--- Model Performance Comparison ---")
print(f"GridSearchCV   -> MSE: {mse_grid:.4f}, RMSE: {rmse_grid:.4f}, R2: {r2_grid:.4f}")
print(f"RandomSearchCV -> MSE: {mse_random:.4f}, RMSE: {rmse_random:.4f}, R2: {r2_random:.4f}")
print(f"Optuna (Bayesian) -> MSE: {mse_optuna:.4f}, RMSE: {rmse_optuna:.4f}, R2: {r2_optuna:.4f}")


# In[ ]:


results = pd.DataFrame({
    'Model': ['GridSearchCV', 'RandomizedSearchCV', 'Optuna (Bayesian)'],
    'MSE': [mse_grid, mse_random, mse_optuna],
    'RMSE': [rmse_grid, rmse_random, rmse_optuna],
    'R2 Score': [r2_grid, r2_random, r2_optuna]
})

print("\n--- Comparison Table ---")
print(results)


# In[ ]:


# Plotting
fig, axs = plt.subplots(1, 3, figsize=(18,5))

# MSE Plot
axs[0].bar(results['Model'], results['MSE'], color=['skyblue', 'lightgreen', 'salmon'])
axs[0].set_title('MSE Comparison')
axs[0].set_ylabel('MSE')

# RMSE Plot
axs[1].bar(results['Model'], results['RMSE'], color=['skyblue', 'lightgreen', 'salmon'])
axs[1].set_title('RMSE Comparison')
axs[1].set_ylabel('RMSE')

# R2 Score Plot
axs[2].bar(results['Model'], results['R2 Score'], color=['skyblue', 'lightgreen', 'salmon'])
axs[2].set_title('R² Score Comparison')
axs[2].set_ylabel('R² Score')

plt.tight_layout()
plt.show()


# ## Multi Layer Perception

# In[4]:


cv_results = {
    'Method': [],
    'RMSE': [],
    'MAE': [],
    'R2': []
}


# In[5]:


mlp = MLPRegressor(random_state=42)

mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("MLP Regressor Performance:")
print("RMSE:", rmse)
print("MAE:", mae)
print(f"R^2 Score: {r2}")
cv_results['Method'].append('MLP Regressor')
cv_results['RMSE'].append(rmse)
cv_results['MAE'].append(mae)
cv_results['R2'].append(r2)


# In[6]:


param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu'],
    'solver': ['sgd'],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'learning_rate': ['adaptive'],
    'max_iter': [500]
}

param_dist = {
    'hidden_layer_sizes': [(np.random.randint(50, 200),), (np.random.randint(50, 200), np.random.randint(50, 200))],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': loguniform(1e-4, 1e-1),
    'learning_rate': ['adaptive'],
    'max_iter': [500]
}


# ### Grid Search CV

# In[7]:


grid_search = GridSearchCV(
  MLPRegressor(random_state=42),
  param_grid,
  scoring='neg_mean_squared_error',
  cv=3,
  n_jobs=-1,
  return_train_score=True
)
grid_search.fit(X_train, y_train)

print("Best GridSearchCV Parameters:", grid_search.best_params_)
print("Best GridSearchCV Score:", -grid_search.best_score_)
best_model_grid = grid_search.best_estimator_
y_pred_grid = best_model_grid.predict(X_test)

rmse_grid = np.sqrt(mean_squared_error(y_test, y_pred_grid))
r2_grid = r2_score(y_test, y_pred_grid)
mae_grid = mean_absolute_error(y_test, y_pred_grid)
print("GridSearchCV RMSE:", rmse_grid)
print("GridSearchCV MAE:", mae_grid)
print(f"GridSearchCV R^2 Score: {r2_grid}")
cv_results['Method'].append('MLP Regressor (Grid Search)')
cv_results['RMSE'].append(rmse_grid)
cv_results['MAE'].append(mae_grid)
cv_results['R2'].append(r2_grid)


# ### Random Search CV

# In[8]:


random_search = RandomizedSearchCV(
    MLPRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    scoring='neg_mean_squared_error',
    cv=2,
    n_jobs=-1,
    random_state=42,
    return_train_score=True
)
random_search.fit(X_train, y_train)
print("Best RandomizedSearchCV Parameters:", random_search.best_params_)
print("Best RandomizedSearchCV Score:", -random_search.best_score_)
best_model_random = random_search.best_estimator_
y_pred_random = best_model_random.predict(X_test)
rmse_random = np.sqrt(mean_squared_error(y_test, y_pred_random))
r2_random = r2_score(y_test, y_pred_random)
mae_random = mean_absolute_error(y_test, y_pred_random)
print("RandomizedSearchCV RMSE:", rmse_random)
print("RandomizedSearchCV MAE:", mae_random)
print(f"RandomizedSearchCV R^2 Score: {r2_random}")
cv_results['Method'].append('MLP Regressor (Random Search)')
cv_results['RMSE'].append(rmse_random)
cv_results['MAE'].append(mae_random)
cv_results['R2'].append(r2_random)


# ### Bayesian Optimization

# In[22]:


def mlp_cv(hidden_layer_sizes, alpha):
    hidden_layer_sizes = int(hidden_layer_sizes)
    alpha = float(alpha)

    model = MLPRegressor(
        hidden_layer_sizes=(hidden_layer_sizes,),
        alpha=alpha,
        max_iter=500,
        random_state=42
    )

    score = cross_val_score(model, X_train, y_train,
                            scoring='neg_mean_squared_error',
                            cv=2, n_jobs=-1)
    
    return score.mean()

# Parameter bounds
pbounds = {
    'hidden_layer_sizes': (50, 200),
    'alpha': (1e-4, 1e-1)
}

# Bayesian Optimization
optimizer = BayesianOptimization(
    f=mlp_cv,
    pbounds=pbounds,
    random_state=42
)

optimizer.maximize(init_points=5, n_iter=15)

# Extract best parameters
best_params = optimizer.max['params']
best_hidden = int(best_params['hidden_layer_sizes'])
best_alpha = float(best_params['alpha'])

# Train final model
final_model = MLPRegressor(
    hidden_layer_sizes=(best_hidden,),
    alpha=best_alpha,
    max_iter=500,
    activation='relu',
    solver='adam',
    learning_rate='adaptive',
    random_state=42
)

final_model.fit(X_train, y_train)
y_pred_bayes = final_model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 Final Evaluation Metrics:")
print("Best Parameters (Bayesian Optimization):", best_params)
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R^2  : {r2:.4f}")

cv_results['Method'].append('MLP Regressor (Bayesian Optimization)')
cv_results['RMSE'].append(rmse)
cv_results['MAE'].append(mae)
cv_results['R2'].append(r2)


# ### Comparison of Hyperparameter Tuning Techniques

# In[24]:


cv_df = pd.DataFrame(cv_results)

# Set style
sns.set_theme(style="whitegrid")

# Plot RMSE
plt.figure(figsize=(10, 6))
sns.barplot(x='Method', y='RMSE', data=cv_df, palette='muted')
plt.title('Comparison of RMSE Across Tuning Methods (Lower is better)')
plt.ylabel('RMSE')
plt.show()

# Plot MAE
plt.figure(figsize=(10, 6))
sns.barplot(x='Method', y='MAE', data=cv_df, palette='muted')
plt.title('Comparison of MAE Across Tuning Methods (Lower is better)')
plt.ylabel('MAE')
plt.show()

# Plot R²
plt.figure(figsize=(10, 6))
sns.barplot(x='Method', y='R2', data=cv_df, palette='muted')
plt.title('Comparison of R² Score Across Tuning Methods (Higher is better)')
plt.ylabel('R²')
plt.show()


# In[ ]:


del grid_search
del random_search
del optimizer
del best_model_grid
del best_model_random
del final_model
del y_pred_grid
del y_pred_random
del y_pred_bayes
gc.collect()


# ## k-Nearest Neighbors

# ### Grid Search CV

# In[5]:


knn = KNeighborsRegressor()
knn_grid_params = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
knn_grid = GridSearchCV(knn, knn_grid_params, cv=3, scoring='neg_mean_squared_error')
knn_grid.fit(X_train, y_train)
knn_grid_best = knn_grid.best_estimator_
knn_grid_mse = mean_squared_error(y_test, knn_grid_best.predict(X_test))
knn_grid_r2 = r2_score(y_test, knn_grid_best.predict(X_test))

# Print Grid Search hyperparameters and scores
print("\nKNN Grid Search Hyperparameters and Scores:")
knn_grid_results = pd.DataFrame(knn_grid.cv_results_)[['param_n_neighbors', 'param_weights', 'mean_test_score', 'std_test_score']]
knn_grid_results['mean_test_score'] = -knn_grid_results['mean_test_score']  # Convert to positive MSE
print(knn_grid_results)


# ### Random Search CV

# In[6]:


# Random Search
knn_random_params = {'n_neighbors': np.arange(3, 15), 'weights': ['uniform', 'distance']}
knn_random = RandomizedSearchCV(knn, knn_random_params, n_iter=10, cv=3, scoring='neg_mean_squared_error', random_state=42)
knn_random.fit(X_train, y_train)
knn_random_best = knn_random.best_estimator_
knn_random_mse = mean_squared_error(y_test, knn_random_best.predict(X_test))
knn_random_r2 = r2_score(y_test, knn_random_best.predict(X_test))

# Print Random Search hyperparameters and scores
print("\nKNN Random Search Hyperparameters and Scores:")
knn_random_results = pd.DataFrame(knn_random.cv_results_)[['param_n_neighbors', 'param_weights', 'mean_test_score', 'std_test_score']]
knn_random_results['mean_test_score'] = -knn_random_results['mean_test_score']  # Convert to positive MSE
print(knn_random_results)


# ### Bayesian Optimization

# In[7]:


# Bayesian Optimization
knn_bayes_params = {'n_neighbors': (3, 15), 'weights': ['uniform', 'distance']}
knn_bayes = BayesSearchCV(knn, knn_bayes_params, n_iter=10, cv=3, scoring='neg_mean_squared_error', random_state=42)
knn_bayes.fit(X_train, y_train)
knn_bayes_best = knn_bayes.best_estimator_
knn_bayes_mse = mean_squared_error(y_test, knn_bayes_best.predict(X_test))
knn_bayes_r2 = r2_score(y_test, knn_bayes_best.predict(X_test))

# Print Bayesian Optimization hyperparameters and scores
print("\nKNN Bayesian Optimization Hyperparameters and Scores:")
knn_bayes_results = pd.DataFrame({
    'param_n_neighbors': knn_bayes.cv_results_['param_n_neighbors'],
    'param_weights': knn_bayes.cv_results_['param_weights'],
    'mean_test_score': -knn_bayes.cv_results_['mean_test_score'],  # Convert to positive MSE
    'std_test_score': knn_bayes.cv_results_['std_test_score']
})
print(knn_bayes_results)


# ### Results Comparison

# In[8]:


results = pd.DataFrame({
    'Model': ['KNN (Grid)', 'KNN (Random)', 'KNN (Bayes)'],
    'Best Params': [knn_grid.best_params_, knn_random.best_params_, knn_bayes.best_params_],
    'MSE': [ knn_grid_mse, knn_random_mse, knn_bayes_mse],
    'R2': [knn_grid_r2, knn_random_r2, knn_bayes_r2]
})

# Style the table
print("\nPerformance Metrics:")
styled_results = results.style.format({
    'MSE': '{:.4f}',  # Format MSE to 4 decimal places
    'R2': '{:.4f}'    # Format R2 to 4 decimal places
}).set_properties(**{
    'text-align': 'center',  # Center-align text
    'border': '1px solid black',  # Add borders
    'padding': '5px'  # Add padding
}).set_table_styles([
    {'selector': 'th', 'props': [('font-weight', 'bold'), ('text-align', 'center'), ('background-color', '#f2f2f2')]}
])  # Style headers

display(styled_results)


# In[9]:


plt.figure(figsize=(12, 7))  # Adjusted figure size for better spacing

# Sort models by R² score descending
results_sorted = results.sort_values('R2', ascending=False)

# Create bar plot with sorted data
ax = sns.barplot(x='Model', y='R2', data=results_sorted, palette='viridis')

# Add value labels on top of bars
for p in ax.patches:
    ax.annotate(f'{p.get_height():.3f}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 5),
                textcoords='offset points')

plt.title('R² Score Comparison Across Models and Tuning Methods', fontsize=14, pad=20)
plt.xlabel('Model', fontsize=12)
plt.ylabel('R² Score', fontsize=12)
plt.xticks(rotation=45, ha='right')  # Better alignment for rotated labels
plt.ylim(top=min(1.0, results['R2'].max() + 0.1))  # Adjust y-axis limit if needed

# Improve layout spacing
plt.tight_layout()
plt.show()


# # Classification Credit Card Customer Churn Dataset

# In[10]:


# https://raw.githubusercontent.com/Shown246/CSE445_Datasets/refs/heads/main/Customer_Churn_Classification_cleaned.csv
df = pd.read_csv('https://raw.githubusercontent.com/Shown246/CSE445_Datasets/refs/heads/main/Customer_Churn_Classification_cleaned.csv')
df.head()


# ## Loading data  to split

# In[11]:


# Define features and target variable
X = df.drop(columns=["exited"])  # Replace "Churn" with actual target column name
y = df["exited"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## Logistic Regression

# In[12]:


print("Training baseline Logistic Regression...")
baseline_model = LogisticRegression(max_iter=1000, random_state=42)
baseline_model.fit(X_train, y_train)
baseline_pred = baseline_model.predict(X_test)
baseline_acc = accuracy_score(y_test, baseline_pred)
print(f"Baseline Accuracy: {baseline_acc:.4f}\n")


# ### Grid Search CV

# In[14]:


param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear'],
    'penalty': ['l1', 'l2']
}

print("Performing Grid Search...")
grid_search = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), param_grid, scoring='f1', cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)
grid_acc = accuracy_score(y_test, grid_search.best_estimator_.predict(X_test))
print(f"Grid Search Accuracy: {grid_acc:.4f}")


# ### Random Search CV

# In[16]:


param_dist = {
    'C': np.logspace(-3, 2, 100),
    'solver': ['liblinear'],
    'penalty': ['l1', 'l2']
}

print("Performing Random Search...")
random_search = RandomizedSearchCV(LogisticRegression(max_iter=1000, random_state=42), param_distributions=param_dist, n_iter=20, cv=3, random_state=42, n_jobs=-1, scoring='f1')
random_search.fit(X_train, y_train)
random_acc = accuracy_score(y_test, random_search.best_estimator_.predict(X_test))
print(f"Random Search Accuracy: {random_acc:.4f}")


# ### Bayesian Optimization

# In[17]:


# Define the function to optimize
def bo_logistic_regression(C, max_iter, l1_ratio):
    model = LogisticRegression(
        C=C,
        max_iter=int(max_iter),
        solver='saga',               # saga supports 'elasticnet'
        penalty='elasticnet',
        l1_ratio=l1_ratio,
        random_state=42
    )
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='f1', n_jobs=-1).mean()
    return score

# Define parameter bounds
param_bounds = {
    'C': (0.001, 10.0),
    'max_iter': (100, 1000),
    'l1_ratio': (0.0, 1.0)
}

# Run Bayesian Optimization
bo_optimizer = BayesianOptimization(
    f=bo_logistic_regression,
    pbounds=param_bounds,
    random_state=42,
    verbose=2
)

print("Running Bayesian Optimization...")
bo_optimizer.maximize(init_points=10, n_iter=30)

# Extract best parameters
best_params = bo_optimizer.max['params']
best_params['max_iter'] = int(best_params['max_iter'])

# Train final model using best parameters
best_lr = LogisticRegression(
    C=best_params['C'],
    max_iter=best_params['max_iter'],
    solver='saga',
    penalty='elasticnet',
    l1_ratio=best_params['l1_ratio'],
    random_state=42
)

best_lr.fit(X_train, y_train)
bo_predictions = best_lr.predict(X_test)

# Evaluate
bo_accuracy = accuracy_score(y_test, bo_predictions)
print(f"Bayesian Optimization Accuracy: {bo_accuracy:.4f}")


# ### Comparing Models

# In[18]:


# Results Comparison Table
results_df = pd.DataFrame({
    "Tuning Method": ["Baseline", "Grid Search", "Random Search", "Bayesian Optimization"],
    "Accuracy": [baseline_acc, grid_acc, random_acc, bo_accuracy]
})
print("\nFinal Accuracy Comparison:")
print(results_df)

# Bar Plot
plt.figure(figsize=(8, 5))
sns.barplot(x="Tuning Method", y="Accuracy", data=results_df, palette="viridis")
plt.ylim(0.5, 1.0)
plt.title("Model Accuracy by Tuning Method")
plt.ylabel("Accuracy")
plt.xlabel("Method")
plt.show()


# ## Random forrest

# In[19]:


print("Training default Random Forest model...")
default_model = RandomForestClassifier(random_state=42)
default_model.fit(X_train, y_train)
default_pred = default_model.predict(X_test)
default_accuracy = accuracy_score(y_test, default_pred)
print(f"Default Model Accuracy: {default_accuracy:.4f}\n")


# ### Grid Search CV

# In[20]:


param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}
print("Performing Grid Search...")
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, scoring='f1', cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)
grid_accuracy = accuracy_score(y_test, grid_search.best_estimator_.predict(X_test))
print(f"Grid Search Best Accuracy: {grid_accuracy:.4f}\n")


# ### Random Search CV

# In[21]:


param_dist = {
    "n_estimators": np.arange(50, 300, 50),
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": np.arange(2, 11, 2),
    "min_samples_leaf": np.arange(1, 5)
}
print("Performing Random Search...")
random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_distributions=param_dist,
                                   n_iter=20, cv=3, random_state=42, n_jobs=-1, scoring='f1')
random_search.fit(X_train, y_train)
random_accuracy = accuracy_score(y_test, random_search.best_estimator_.predict(X_test))
print(f"Random Search Best Accuracy: {random_accuracy:.4f}\n")


# ### Bayesian Optimization

# In[22]:


def rf_bo(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    rf = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        random_state=42
    )
    scores = cross_val_score(rf, X_train, y_train, cv=3, n_jobs=-1)
    return scores.mean()
param_bounds = {
    'n_estimators': (50, 300),
    'max_depth': (5, 50),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 4),
}
bo_optimizer = BayesianOptimization(f=rf_bo, pbounds=param_bounds, random_state=42)
bo_optimizer.maximize(init_points=5, n_iter=15)

best_params_bo = bo_optimizer.max['params']
best_params_bo['n_estimators'] = int(best_params_bo['n_estimators'])
best_params_bo['max_depth'] = int(best_params_bo['max_depth'])
best_params_bo['min_samples_split'] = int(best_params_bo['min_samples_split'])
best_params_bo['min_samples_leaf'] = int(best_params_bo['min_samples_leaf'])

final_rf_bo = RandomForestClassifier(**best_params_bo, random_state=42)
final_rf_bo.fit(X_train, y_train)
bayesian_accuracy = accuracy_score(y_test, final_rf_bo.predict(X_test))

print(f"Bayesian Optimization Best Accuracy: {bayesian_accuracy:.4f}\n")


# ### Comparing Models

# In[23]:


results_df = pd.DataFrame({
    "Tuning Method": ["Default Model", "Grid Search", "Random Search", "Bayesian Optimization"],
    "Accuracy": [default_accuracy, grid_accuracy, random_accuracy, bayesian_accuracy]
})
print("\nFinal Comparison Table:")
print(results_df)


# In[24]:


plt.figure(figsize=(8, 5))
sns.barplot(x="Tuning Method", y="Accuracy", data=results_df, palette="viridis")
plt.ylim(0.5, 1.0)
plt.xlabel("Hyperparameter Tuning Method")
plt.ylabel("Accuracy Score")
plt.title("Model Accuracy Comparison")
plt.show()


# In[25]:


results_df = pd.DataFrame({
    "Tuning Method": ["Default Model", "Grid Search", "Random Search", "Bayesian Optimization"],
    "Accuracy": [default_accuracy, grid_accuracy, random_accuracy, bayesian_accuracy]
})
print("\nFinal Comparison Table:")
print(results_df)


# ## XGBoost Model

# In[30]:


# Train the model
model = XGBClassifier(objective="binary:logistic", eval_metric="logloss")
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# ### Grid Search

# In[33]:


param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
}

grid_search = GridSearchCV(XGBClassifier(eval_metric="logloss"), param_grid, cv=3, verbose=2, scoring='f1', n_jobs=-1)

grid_search.fit(X_train, y_train)
print("Best Params (Grid Search):", grid_search.best_params_)
print("Best Score (Grid Search):", grid_search.best_score_)


# ### Random Search

# In[35]:


param_dist = {
    'n_estimators': np.arange(50, 300),
    'max_depth': np.arange(3, 10, 2),
    'learning_rate': np.linspace(0.01, 0.3, 10),
}

random_search = RandomizedSearchCV(XGBClassifier(eval_metric="logloss"), param_dist, n_iter=10, verbose=2, cv=3, scoring='f1', n_jobs=-1)

random_search.fit(X_train, y_train)
print("Best Params (Random Search):", random_search.best_params_)
print("Best Score (Random Search):", random_search.best_score_)


# ### Bayesian Optimization

# In[37]:


def xgb_eval(n_estimators, max_depth, learning_rate):
    model = XGBClassifier(n_estimators=int(n_estimators), max_depth=int(max_depth), learning_rate=learning_rate, eval_metric="logloss")

    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

optimizer = BayesianOptimization(
    f=xgb_eval,
    pbounds={'n_estimators': (50, 300), 'max_depth': (3, 10), 'learning_rate': (0.01, 0.3)},
    random_state=42,
    verbose=2  
)

optimizer.maximize(n_iter=10)
print("Best Params (Bayesian Optimization):", optimizer.max)
print("Best Score (Bayesian Optimization):", optimizer.max['target'])


# ### Comparing Result

# In[38]:


results_df = pd.DataFrame({
    "Tuning Method": ["Grid Search", "Random Search", "Bayesian Optimization"],
    "Accuracy": [grid_search.best_score_, random_search.best_score_, optimizer.max['target']]
})
print("\nFinal Comparison Table:")
print(results_df)


# In[39]:


# Bar Plot
plt.figure(figsize=(10, 5))
sns.barplot(x="Tuning Method", y="Accuracy", data=results_df, palette="viridis")
plt.ylim(0.5, 1.0)
plt.title("Model Accuracy by Tuning Method")
plt.ylabel("Accuracy")
plt.xlabel("Method")
plt.show()


# ### Final Model with Best Hyperparameters

# ## Multi Layer Perception

# In[47]:


param_grid = {
    'hidden_layer_sizes': [(50,), (100,100), (100, 50)],
    'alpha': [0.001, 0.01, 0.1],
    'learning_rate': ['adaptive'],
    'max_iter': [500]
}

param_dist = {
    'hidden_layer_sizes': [(np.random.randint(50, 200), np.random.randint(50, 200))],
    'alpha': loguniform(1e-4, 1e-1),
    'learning_rate': ['adaptive'],
    'max_iter': [500]
}

param_bayes = {
    'hidden_layer_1': Integer(50, 200),
    'hidden_layer_2': Integer(0, 200),
    'alpha': Real(1e-4, 1e-1, prior='log-uniform'),
    'learning_rate': Categorical(['adaptive'])
}


# ### Grid Search CV

# In[44]:


grid = GridSearchCV(
    MLPClassifier(random_state=42),
    param_grid,
    scoring='accuracy',
    cv=3,
    n_jobs=-1,
    verbose=2
)
grid.fit(X_train, y_train)
grid_score = grid.best_score_
print(f"GridSearchCV best score: {grid_score}")
grid_best_params = grid.best_params_
print(f"GridSearchCV best params: {grid_best_params}")


# ### Random Search CV

# In[48]:


random_search = RandomizedSearchCV(
    MLPClassifier(random_state=42),
    param_dist,
    n_iter=20,
    scoring='accuracy',
    cv=3,
    n_jobs=-1,
    verbose=2,
    random_state=42
)
random_search.fit(X_train, y_train)
random_score = random_search.best_score_
print(f"RandomizedSearchCV best score: {random_score}")
random_best_params = random_search.best_params_
print(f"RandomizedSearchCV best params: {random_best_params}")


# ### Bayesian Optimization

# In[49]:


class CustomMLPClassifier(MLPClassifier):
    def set_params(self, **params):
        hl1 = params.pop('hidden_layer_1', 100)
        hl2 = params.pop('hidden_layer_2', 0)
        if hl2 > 0:
            params['hidden_layer_sizes'] = (hl1, hl2)
        else:
            params['hidden_layer_sizes'] = (hl1,)
        return super().set_params(**params)

bayes_search = BayesSearchCV(
    estimator=CustomMLPClassifier(random_state=42),
    search_spaces=param_bayes,
    n_iter=20,
    scoring='accuracy',
    cv=3,
    n_jobs=-1,
    verbose=2
)
bayes_search.fit(X_train, y_train)
bayes_score = bayes_search.best_score_
print(f"BayesSearchCV best score: {bayes_score}")
bayes_best_params = bayes_search.best_params_
print(f"BayesSearchCV best params: {bayes_best_params}")


# ### Result Comparison

# In[50]:


results_df = pd.DataFrame({
    "Tuning Method": ['GridSearchCV', 'RandomizedSearchCV', 'BayesSearchCV'],
    "Accuracy": [grid_score, random_score, bayes_score]
})
print("\nFinal Comparison Table:")
print(results_df)


# In[51]:


# Bar Chart
methods = ['GridSearchCV', 'RandomizedSearchCV', 'BayesSearchCV']
scores = [grid_score, random_score, bayes_score]

plt.figure(figsize=(8, 5))
plt.bar(methods, scores, color=['skyblue', 'orange', 'green'])
plt.title("MLPClassifier Accuracy Comparison")
plt.ylabel("Cross-Validated Accuracy")
plt.ylim(0.5, 1.0)
for i, v in enumerate(scores):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# ## k-Nearest Neighbors

# In[52]:


print("Training default KNN model...")
default_model = KNeighborsClassifier()
default_model.fit(X_train, y_train)
default_pred = default_model.predict(X_test)
default_accuracy = accuracy_score(y_test, default_pred)
print(f"Default Model Accuracy: {default_accuracy:.4f}\n")


# ### Grid Search CV

# In[53]:


print("Performing Grid Search...")
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'p': [1, 2] 
}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=3, verbose=3, n_jobs=-1)
grid_search.fit(X_train, y_train)
grid_accuracy = accuracy_score(y_test, grid_search.best_estimator_.predict(X_test))
print(f"🔹 Grid Search Best Accuracy: {grid_accuracy:.4f}")
print(f"Best Parameters: {grid_search.best_params_}\n")


# ### Random Search CV

# In[54]:


print("Performing Random Search...")
param_dist = {
    'n_neighbors': np.arange(3, 21, 2),
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}
random_search = RandomizedSearchCV(KNeighborsClassifier(), param_distributions=param_dist,
                                   n_iter=20, cv=3, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)
random_accuracy = accuracy_score(y_test, random_search.best_estimator_.predict(X_test))
print(f"🔹 Random Search Best Accuracy: {random_accuracy:.4f}")
print(f"Best Parameters: {random_search.best_params_}\n")


# ### Bayesian Optimization

# In[55]:


def knn_bo(n_neighbors, p):
    # p must be 1 or 2 (Manhattan or Euclidean), weights as categorical
    params = {
        'n_neighbors': int(n_neighbors),
        'weights': 'distance' if n_neighbors > 10 else 'uniform',  # Simplified choice
        'p': int(p)
    }
    knn = KNeighborsClassifier(**params)
    scores = cross_val_score(knn, X_train, y_train, cv=3, n_jobs=-1)
    return scores.mean()

param_bounds = {
    'n_neighbors': (3, 20),
    'p': (1, 2)
}
bo_optimizer = BayesianOptimization(f=knn_bo, pbounds=param_bounds, random_state=42)
bo_optimizer.maximize(init_points=5, n_iter=15)

# Get best parameters
best_params_bo = bo_optimizer.max['params']
best_params_bo['n_neighbors'] = int(best_params_bo['n_neighbors'])
best_params_bo['p'] = int(best_params_bo['p'])
best_params_bo['weights'] = 'distance' if best_params_bo['n_neighbors'] > 10 else 'uniform'

# Train final model
final_knn_bo = KNeighborsClassifier(**best_params_bo)
final_knn_bo.fit(X_train, y_train)
bo_accuracy = accuracy_score(y_test, final_knn_bo.predict(X_test))
print(f"🔹 Bayesian Optimization Best Accuracy: {bo_accuracy:.4f}")
print(f"Best Parameters: {best_params_bo}\n")


# ### Performance Comparison

# In[56]:


# Comparison table
results_df = pd.DataFrame({
    "Tuning Method": ["Default Model", "Grid Search", "Random Search", "Bayesian Optimization"],
    "Accuracy": [default_accuracy, grid_accuracy, random_accuracy, bayesian_accuracy]
})
print("\nFinal Comparison Table:")
print(results_df)

# Plot: Accuracy Comparison
plt.figure(figsize=(8, 5))
sns.barplot(x="Tuning Method", y="Accuracy", data=results_df, palette="viridis")
plt.ylim(0.5, 1.0)
plt.xlabel("Hyperparameter Tuning Method")
plt.ylabel("Accuracy Score")
plt.title("KNN Model Accuracy Comparison")
plt.show()

