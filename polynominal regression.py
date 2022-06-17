
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

veriler = pd.read_csv('maaslar.csv')

x = veriler.iloc[:,1:2]
y = veriler.iloc [:,2:]
xnum = x.values
ynum = y.values

poly_reg = PolynomialFeatures (degree=4)
x_poly = poly_reg.fit_transform (xnum)
lin_reg = LinearRegression()
lin_reg.fit(x_poly,y)


plt.scatter(xnum,ynum,color = 'green')
plt.plot(xnum,lin_reg.predict(x_poly), color ='red')

plt.show()

#some predictions

print(lin_reg.predict(poly_reg.fit_transform([[8]])))
print(lin_reg.predict(poly_reg.fit_transform([[12]])))
print(lin_reg.predict(poly_reg.fit_transform([[5.6]])))

