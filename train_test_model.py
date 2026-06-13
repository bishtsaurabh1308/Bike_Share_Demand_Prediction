
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import joblib

print('all done')

pd.set_option('display.max_columns',None)
# load dataset
df = pd.read_csv('Bike_Share_Demand.csv')
# print(df.head())

df = df.drop(columns='date')
# print(df.head(2))

# encoding
df = pd.get_dummies(df,columns=['season','weather','city_zone'],drop_first=True,dtype=int)
# print(df.head(2))

# featurs and target
X = df.drop(columns='bike_demand')
y = df['bike_demand']
# print(X)
# print(y)

# split data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# fit model
lr = LinearRegression()
lr.fit(X_train_scaled,y_train)

# prediction
y_pred = lr.predict(X_test_scaled)

# evaluate
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,y_pred)

print('MAE', mae)
print('MSE', mse)
print('RMSE', rmse)
print('r2_score', r2)

# overfitting
print(lr.score(X_train_scaled, y_train))
print(lr.score(X_test_scaled, y_test))

# save model
joblib.dump(lr,'model.pkl')

joblib.dump(scaler, "scaler.pkl")

joblib.dump(X.columns, "feature_columns.pkl")

print('trained and saved sucessfully')