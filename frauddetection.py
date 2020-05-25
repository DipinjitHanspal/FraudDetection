# Step 1: Use a SOM to identify outliers in the dataset
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, Dense
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)

# Training the SOM
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8, 1)], mappings[(6, 8)]), axis=0)
frauds = sc.inverse_transform(frauds)

# Step 2: Predict the probability of a fraudulent application using a supervised deep learning model
""" 
    Frauds contains the suspected fraudulent accounts. We will categorize the suspected accounts 
    as fraudulent or not along with the percentage likelihood
"""

# Don't need customerID
customers = dataset.iloc[:, 1:].values

# Problem : devising a dependant variable from an unsupervised model
# Solution : Augment frauds outcome from SOM to generate dependant variable
"""
    Frauds contains customerIDs of suspected frauds. This can be used to find the index of the customer 
    customers matrix, thus allowing us to map a 1 in the dependant variable vector at the location of the
    suspected fraud customers
"""
is_fraud = np.zeros(len(dataset))
# Update suspected frauds
for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1

# Feature scaling
sc = StandardScaler()
customers = sc.fit_transform(customers)

classifer = Sequential()
classifer.add(Dense(units=2, kernel_initializer='uniform',
                    activation='relu', input_dim=15))
classifer.add(
    Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
classifer.compile(optimizer='adam',
                  loss='binary_crossentropy', metrics=['accuracy'])
classifer.fit(customers, is_fraud, batch_size=1, epochs=2)
# Predicting probability of frauds
y_pred = classifer.predict(customers)
y_pred = np.concatenate((dataset.iloc[:,0:1].values, y_pred), axis=1)
# Sort y_pred by index 1
y_pred = y_pred[y_pred[:, 1].argsort()]
