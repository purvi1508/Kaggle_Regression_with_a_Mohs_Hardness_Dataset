import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import median_absolute_error
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('/kaggle/input/playground-series-s3e25/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s3e25/test.csv')
print(train_df.head())

X = train_df.drop('Hardness', axis=1)
y = train_df['Hardness']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

X = train_df.drop('Hardness', axis=1)  # Features
y = train_df['Hardness']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a more complex neural network model
model = Sequential()
model.add(Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dropout(0.5))  # Dropout for regularization
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))  # Dropout for regularization
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')

# Train the model
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Calculate Median Absolute Error
medae = median_absolute_error(y_test, y_pred)
print(f'Median Absolute Error: {medae}')

import lightgbm as lgb
from sklearn.metrics import median_absolute_error
from sklearn.model_selection import train_test_split

# Assuming your DataFrame is named df
# Split the data into features (X) and target variable (y)
X = train_df.drop(['id', 'Hardness'], axis=1)  
y = train_df['Hardness']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the objective function for optimization
def objective(trial):
    params = {
        'metric': 'mae',
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 15),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.05),
        'n_estimators': trial.suggest_int('n_estimators', 300, 700),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.1, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0),
        'seed': trial.suggest_categorical('seed', [42]),
        'device': trial.suggest_categorical('device', ['cpu']),  # Force CPU usage
    }

    model_lgb = lgb.LGBMRegressor(**params)
    model_lgb.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model_lgb.predict(X_test)

    # Calculate Median Absolute Error (MedAE)
    medae = median_absolute_error(y_test, y_pred)

    return medae

# Optimize hyperparameters using Optuna
import optuna

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Get the best parameters
best_params = study.best_params
print(f"Best Parameters: {best_params}")

# Train the final model with the best parameters
final_model = lgb.LGBMRegressor(**best_params)
final_model.fit(X_train, y_train)

# Predict on the test set
y_pred_final = final_model.predict(X_test)

# Calculate Median Absolute Error (MedAE) for the final model
medae_final = median_absolute_error(y_test, y_pred_final)
print(f"Final Model MedAE: {medae_final}")

X = test_df.drop(['id'], axis=1)
X_test_scaled = scaler.transform(X)
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
y_pred = model.predict(X_test_reshaped)
print(y_pred)

submission_df = pd.DataFrame({'id': test_df['id'], 'Hardness': y_pred.flatten()})
submission_df.to_csv('submission.csv', index=False)
