
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

veriler = pd.read_csv('maaslar.csv')

x = veriler.iloc[:,1:2]
y = veriler.iloc [:,2:]
xnum = x.values
ynum = y.values

'''
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

'''

sc1 = StandardScaler()
xsc = sc1.fit_transform(xnum)
sc2 = StandardScaler()
ysc = sc2.fit_transform(ynum)

svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(xsc,ysc)

plt.scatter (xsc,ysc, color = 'orange')
plt.plot (xsc,svr_reg.predict(xsc), color = 'green')
plt.show()

#some predictions
sc3 = StandardScaler()
pred1 = svr_reg.predict([[9.8]])
sc3.fit([[9.8]])
pred1 = sc3.inverse_transform([[9.8]])
print(pred1)
#print (svr_reg.predict([[9.8]]))
#print (svr_reg.predict([[10]]))


