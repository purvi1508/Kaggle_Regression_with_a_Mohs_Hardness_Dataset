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

# Assuming 'train_df' is your DataFrame with the data
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

X = test_df.drop(['id'], axis=1)
X_test_scaled = scaler.transform(X)
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
y_pred = model.predict(X_test_reshaped)
print(y_pred)

submission_df = pd.DataFrame({'id': test_df['id'], 'Hardness': y_pred.flatten()})
submission_df.to_csv('submission.csv', index=False)