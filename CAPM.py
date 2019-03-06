#2019-02-10 Peilin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
stock_data=pd.read_csv('D:\\python_data\\Task1\\TRD_Index_20.csv')
xin_an_data=pd.read_csv('D:\\python_data\\Task1\\xin_an.csv').iloc[:,1:]
#Risk-free interest rate
R_f=(1+0.036)**(1/360)-1
#remove the volume is zero
xin_an_data=xin_an_data[xin_an_data['Volume']!=0]
#Move forward one day

#The series of xinan return 
xin_an_return=(xin_an_data['Close']-xin_an_data['Close'].shift(1))/xin_an_data['Close'].shift(1)
xin_an_return.index=pd.to_datetime(xin_an_data['Date'])

stock_data=stock_data[stock_data['Indexcd']==902]
index_return=pd.Series(stock_data['Retindex'].values,index=pd.to_datetime(stock_data['Trddt'].values))
index_return.name='Return'
#to_Datatime can be comparable

xin_an_return=xin_an_return.dropna()
xin_an_return.name='Return'
xin_an_return=xin_an_return-R_f
#costruct a list
index_return=index_return[[index for index in xin_an_return.index]]
index_return=index_return-R_f

Ret=pd.merge(pd.DataFrame(xin_an_return),pd.DataFrame(index_return),how='inner',left_index=True,right_index=True)

plt.scatter(xin_an_return,index_return)
#Y and X(dependent var and independent var)

ols_model=sm.OLS(Ret['Return_x'],sm.add_constant(Ret['Return_y']))
ols_model_result=ols_model.fit()
print(ols_model_result.summary())
X=np.linspace(-0.1,0.1,10)
plt.plot(X,1.1293*X-0.0017)
