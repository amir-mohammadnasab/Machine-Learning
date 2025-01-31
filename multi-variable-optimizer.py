import numpy as np
import pandas as pd

def error(x,y,w):
    y_hat = w*x
    y_hat_final = y_hat.sum(axis=1)[:,None]
    error_hat = np.sum((y_hat_final - y)**2)/y.size
    return error_hat

def gradient(x,y,w,eta):
    y_hat = w*x
    y_hat_final = y_hat.sum(axis=1)[:,None]
    for j in range(w.shape[0]):
        w[j] -= eta*(np.sum(2*x[:,j][:,None]*(y_hat_final - y))/y.size)
    shib = np.sum(2*x[:,3][:,None]*(y_hat_final - y))/y.size
    return w , shib

def give_eta(x,y):
    x = np.hstack((np.ones((x.shape[0],1)),x))
    eta_list = []
    for eta in np.linspace(0,1,1000):
        w = np.zeros(x.shape[1])
        for j in range(1500):
            w , shib = gradient(x,y,w,eta)
        eta_list.append([eta,shib])
    eta_list = np.array(eta_list)
    eta_list = np.nan_to_num(eta_list,nan=1)
    return round(eta_list[np.argmin(np.abs(eta_list)[:,1])][0],4)

def Optimizer(x,y):
    eta = 0.001
    # eta = give_eta(x,y) / in (small or some) datas , you can unmark this to set a automatic optimized eta for you
    x = np.hstack((np.ones((x.shape[0],1)),x))
    w = np.zeros(x.shape[1])
    
    shib = 5
    while (shib < -0.0001 or shib > 0.0001): 
        # print(error(x,y,w)) / if you wanna see errors , please unmark this
        w , shib = gradient(x,y,w,eta)
    return w    
    
def test(x,w):
    x = np.hstack((np.ones((x.shape[0],1)),x))
    y = x*w
    y_hat_final = y.sum()
    return y_hat_final

#Example Dataset
df = pd.read_csv("./income.csv")
train = np.array(df)[:,1:]
x = train[:,0:5]
y = train[:,5][:,None] #be aware , Your Targets must be in columns


# Optimized Weight
w = Optimizer(x,y)
print(w)

# Now you can test and evaluate your data on model (you should enter a matrix format with numpy)
# k = test(array,w)
# print(k)


