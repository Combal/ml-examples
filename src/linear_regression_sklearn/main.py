from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

df = pd.read_csv('./data/linear_regression.csv', header=None, names=['X1', 'X2', 'Y'])

df.insert(loc=0, column='X0', value=1)

print(df.head())
X = df.values[:400, 0:3]
y = df.values[:400, 3]

X_test = df.values[400:, 0:3]
y_test = df.values[400:, 3]

reg = LinearRegression().fit(X, y)
print('theta: {}'.format(reg.coef_))
print('validation score: {}'.format(reg.score(X, y)))
print('test score: {}'.format(reg.score(X_test, y_test)))

print('predict: {}'.format(reg.predict(np.array([
    [1, -88.49784519830919294, -98.54773331130996894],  # -866.1290774336297318
    [1, -5.281762866637990328, 11.40201130968224419]  # -2.524902938366139438
]))))
