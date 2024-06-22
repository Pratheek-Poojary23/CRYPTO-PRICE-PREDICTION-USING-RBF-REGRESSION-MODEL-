# CRYPTO-PRICE-PREDICTION-USING-RBF-REGRESSION-MODEL-
ETHEREUM-PRICE-PREDICTION-USING-RBF-REGRESSION-MODEL

### Problem Statement

This study aims to forecast Ethereum (ETH) prices in Indian Rupees (INR) using Radial Basis Function (RBF). The primary challenge is to develop accurate predictive models that anticipate Ethereum price movements, including changes in price signs, to support informed investment decisions.

### Data

A single dataset for Ethereum (ETH) prices in Indian Rupees (INR), 
collected from Kaggle. The dataset spans from January 2018 to July 2021, total 1681 data points and contains essential attributes such as date, open price, high price, low price, close price, adjusted close price, and trading volume.

### Feature engineering

- The selected features used in our model include 'Year', 'Month', 'Day', 'Open', 'High', and 'Low'. These features represent temporal aspects (year, month, day) as well as daily price fluctuations (open, high, low) of Ethereum.
- The 'Adj Close' column, representing adjusted closing prices, is excluded as a feature in our analysis. Adjusted closing prices are typically adjusted for factors such as dividends, stock splits or other corporate actions, which may not directly reflects ETH's market demand.

```python
data = data.drop(['Adj Close'], axis=1)
X = data[['Year', 'Month', 'Day', 'Open', 'High', 'Low']]
y = data['Close']
```

The 'Date' column in the dataset contains timestamps representing the date and time when each observation was recorded. Extracting the year, month, and day from the 'Date' column allows the model to capture potential seasonal or yearly patterns in Ethereum price movements. 

```python
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

```

### Model Implementation: 
- The dataset is split into training and testing sets, with 80% allocated for training and 20% for testing.
```python
train_size = int(0.8 * len(data))
X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

```
Notably, the split was done sequentially to preserve the temporal sequence of the data. Random splitting may bias the evaluation of model performance, especially if there are systematic changes in the data over time. 

- To ensure model stability and convergence, features are standardized using the StandardScaler to achieve zero mean and unit variance.
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1))
```
- To enhance the model's performance, hyperparameter tuning is done using GridSearchCV.
```python
param_grid = {
    'kernel': [1.0 * RBF(length_scale=length_scale) for length_scale in np.logspace(-1, 2, 5)],
    'alpha': np.logspace(-3, 2, 5)
}
```
- Gaussian Process Regression, a non-parametric and flexible regression technique, with an RBF kernel to model the relationship between the input features and the target variable.
```python
gp = GaussianProcessRegressor()
grid_search = GridSearchCV(gp, param_grid, cv=5, scoring='neg_mean_squared_error')
```

- The RBF regressor is trained on the scaled training data to learn the relationship between the input features and the target variable. Predictions are subsequently made on the scaled test set.
```python
grid_search.fit(X_train_scaled, y_train_scaled)

# Access the best model and its parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

y_pred = best_model.predict(X_test_scaled)
y_pred_train = best_model.predict(X_train_scaled)
```
