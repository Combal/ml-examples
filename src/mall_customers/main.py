import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('data/Mall_Customers.csv', names=['CustomerID', 'Gender', 'Age', 'AnnualIncome', 'SpendingScore'],
                 skiprows=1, index_col='CustomerID')

df = pd.get_dummies(df, columns=['Gender'], drop_first=True)
y = df.SpendingScore
df = df.drop('SpendingScore', axis='columns')

# print(df.head())
# print(y.head())

threshold = 180

X, X_test = df[:threshold], df[threshold:]
y, y_test = y[:threshold], y[threshold:]

model = LinearRegression()

model.fit(X, y)

print('score: {}'.format(model.score(X_test, y_test)))

