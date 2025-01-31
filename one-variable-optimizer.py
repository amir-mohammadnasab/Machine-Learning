import numpy as np

def error(x,y,w1,w0):
    error_hat = sum((w1*x + w0 - y)**2)/y.size
    return error_hat

def gradient(x, y, w1, w0, eta):
    grad_w1 = sum(2*x*(w1*x + w0 - y))/y.size
    grad_w0 = sum(2*(w1*x + w0 - y))/y.size
    w1 -= eta*grad_w1
    w0 -= eta*grad_w0
    return w1, w0 , grad_w1 , grad_w0

def give_eta(x,y):
    eta_list = []
    for eta in np.linspace(0,1,10000):
        w1 = np.random.randn()
        w0 = np.random.randn()
        for j in range(20):
            w1 , w0 , grad_w1 , grad_w0 = gradient(x,y,w1,w0,eta)
        eta_list.append([eta,grad_w1])
    eta_list = np.array(eta_list)
    eta_list = np.nan_to_num(eta_list,nan=1)
    return round(eta_list[np.argmin(np.abs(eta_list)[:,1])][0],4)

def Optimizer(x,y):
    eta = 0.001
    # eta = give_eta(x,y) / in (small or some) datas , you can unmark this to set a automatic optimized eta for you
    w1 = np.random.randn()
    w0 = np.random.randn()
    grad1 , grad0 = 5 , 5
    while (grad1 < -0.001 or grad1 > 0.001) or (grad0 < -0.001 or grad0 > 0.001):
        error_hat = error(x,y,w1,w0)
        # print(error_hat) / if you wanna see errors , please unmark this
        w1 , w0 , grad1 , grad0 = gradient(x,y,w1,w0,eta)
        
    print(f"w1={w1} , w0={w0} , error={error_hat}")
    print(f"y={w1}x + {w0}")



# Example
x = np.array([2,6,10,220,67,12,90,78,20,400,23,1,0])
y = np.array([200,71,100,21,67,12,90,78,80,7,8,61,200])


# print(y)
Optimizer(x,y)









