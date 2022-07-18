# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 14:18:56 2022

@author: Feng
"""

import numpy as np
import config
import mod

def first_order_system(env, Ts, init_value):
    datalength = len(env)
    l = datalength + 2
    env = np.append(env, [env[-2], env[-1]])
    sens = np.zeros(l)
    N = l

    for k in range(N):
        if k == 0:
            env[k-1] = init_value
            sens[k-1] = init_value
        sens[k] = (Ts*env[k] + Ts*env[k-1] - (Ts-16)*sens[k-1])/(16 + Ts)

    env = env[0:-2]
    sens = sens[0:-2]

    return sens


def produce_whitenoise(x, snr):
    len_x = len(x)
    Ps = np.sum(np.power(x, 2)) / len_x
    Pn = Ps / (np.power(10, snr/10))
    noise = np.random.randn(len_x) * np.sqrt(Pn)
    return noise

### Cost Function ###
def sensor_system(env):
    sensor = config.sens
    Ts = 1
    init_sens = sensor[0]
    sensor_out = first_order_system(env, Ts, init_sens)

    residual = abs(np.array(sensor_out) - np.array(sensor)).sum()
    # print(residual)
    return residual

def post_process(signal, width):
    filtered = np.convolve(signal, np.ones(width), 'valid') / width
    return filtered


def grad(f, x):
    '''
    CENTRAL FINITE DIFFERENCE CALCULATION
    '''
    h = np.cbrt(
        np.finfo(float).eps)  
    # cbrt (take cube root)
    # finfo (a class about float number)
    # eps (1.0 and the smallest representable number greater than 1.0)
    
    d = len(x)  # Length of data
    nabla = np.zeros(d)  # vector used to store gradient
    for i in range(d):
        x_for = np.copy(x)  # Deep copy: variables before and after copying do not affect each other
        x_back = np.copy(x)
        x_for[i] = x_for[i] + h  # Add a minimum representable number to the calculation position
        x_back[i] = x_back[i] - h  # Subtract a minimum representable number from the calculated position
        nabla[i] = (f(x_for) - f(x_back))/(2*h)  # Calculate the gradient at position i

        # print("in grad")

    # print("out grad")
    return nabla  # return gradient


def line_search(f, x, p, nabla):
    '''
    BACKTRACK LINE SEARCH WITH WOLFE CONDITIONS
    '''
    init_step = 4
    a = init_step  # Length of Step: 1 (initial value)
    step = init_step / 2
    c1 = 1e-4
    c2 = 0.9
    fx = f(x)  # Function to be optimized
    x_new = x + a * p  # New position of x ��where p is the product of the Hessien matrix and the gradient, and then taking the negative number��
    nabla_new = grad(f, x_new)  # New gradient of x
    
    
    """
    @ here means matrix multiplication
    (loop condition: 
     new function value > previous function value + gradient related function
     or 
     gradient correlation function I < gradient related function II)
    """
    while f(x_new) >= fx + (c1*a*nabla.T@p) or (nabla_new.T@p) <= (c2*nabla.T@p):
        if a - step <= 0:
            step = step / 2
            a = init_step

        a = a - step  # Mininize the length of step to 1/2 of last value
        print('{} {} {}'.format(step, 'step length: ', a))
        x_new = x + a * p  # Calculate the new potision x
        nabla_new = grad(f, x_new)  # Calculate the new gradient
        # print("in line_search")

        if step < 0.05:
            print("Step Not Found!!!")
            a = 0.001
            break
    return a  # return length of step


def BFGS(f, x0, max_it):
    '''
    DESCRIPTION
    BFGS Quasi-Newton Method

    INPUTS:
    f:      function to be optimised 
    x0:     intial guess
    max_it: maximum iterations 
    OUTPUTS: 
    x:      the optimal solution of the function f 
    '''
    d = len(x0)  # dimension of problem
    nabla = grad(f, x0)  # initial gradient
    H = np.eye(d)  # initial hessian
    x = x0[:]  # Give the initial value to x
    it = 2  # Set the number of iterations to 2

    while np.linalg.norm(nabla) > 1e-5:  # Norm of gradient matrix
        if it > max_it:
            # Stop the loop when the maximum number of iterations is reached
            print('Maximum iterations reached!')
            break
        it = it + 1  # Update iteration number
        # print("in BFGS")

        p = -H@nabla  # search direction (Newton Method)

        a = line_search(f, x, p, nabla)  # line search
        s = a * p  # step length (difference of parameter vector)
        x_new = x + a * p  # Update x use s
        nabla_new = grad(f, x_new)  # Calculate new gradient using new x
        y = nabla_new - nabla  # (difference of gradient vector)
        y = np.array([y])  # Transpose of y
        s = np.array([s])  # Transpose of s
        y = np.reshape(y, (d, 1))  # Ensure the correct matrix form
        s = np.reshape(s, (d, 1))  # Ensure the correct matrix form
        r = 1/(y.T@s)  # inverse of y(T) * s
        li = (np.eye(d)-(r*((s@(y.T)))))  # np.eye() Generate an [M, N] matrix with diagonal 1 and the rest 0
        ri = (np.eye(d)-(r*((y@(s.T)))))
        hess_inter = li@H@ri
        H = hess_inter + (r*((s@(s.T))))  # BFGS Update
        nabla = nabla_new[:]  # gradient
        x = x_new[:]  # parameter (to be estimated)

    return x  # return parameters


def LBFGS(f, x0, max_it, m=5):
    # https://blog.csdn.net/google19890102/article/details/46389869
    '''
    DESCRIPTION
    L-BFGS Quasi-Newton Method

    INPUTS:
    f:      function to be optimised 
    x0:     intial guess
    max_it: maximum iterations 
    m:      Number of iteration used to calculate H
    OUTPUTS: 
    x:      the optimal solution of the function f 
    '''
    d = len(x0)  # length of parameters
    H0 = np.eye(d)  # initial value of H
    x = x0[:]

    s = []
    y = []

    k = 1
    gradi = grad(f, x0)  # gradient of initial point
    dirc = -H0@gradi  # search direction of initial point
    while np.linalg.norm(gradi) > 1e-5:
        if k > max_it:
            # Stop the loop when the maximum number of iterations is reached
            print('Maximum iterations reached!')
            break

        a = line_search(f, x, dirc, gradi)  # line search
        x_new = x + a * dirc  # Update x use s

        gradi_new = grad(f, x_new)  # Calculate new gradient using new x

        if k > m:
            s.pop(0)
            y.pop(0)

        sk = x_new - x
        yk = gradi_new - gradi
        qk = gradi_new
        s.append(sk)
        y.append(yk)

        r = s[0].T@y[0] / (y[0].T@y[0])
        H0 = r * np.eye(d)

        t = len(s)
        a = []
        for i in range(t):
            alpha = s[t-i-1].T@qk / (y[t-i-1].T@s[t-i-1])
            qk = qk - alpha * y[t-i-1]
            a.append(alpha)

        r = H0@qk

        for i in range(t):
            beta = (y[i].T@r) / (y[i].T@s[i])
            r = r + s[i] * (a[t-i-1] - beta)

        # if(yk.T@sk > 0):
        #     dirc = -r
        #     print("change direction!!!")

        dirc = -r
        gradi = gradi_new
        x = x_new

        k = k + 1

        # print(k)
    return x


def LBFGSB(f, x0, max_it, m=5):
    '''
    DESCRIPTION
    L-BFGS-B Quasi-Newton Method

    INPUTS:
    f:      function to be optimised 
    x0:     intial guess
    max_it: maximum iterations 
    m:      Number of iteration used to calculate H
    OUTPUTS: 
    x:      the optimal solution of the function f 
    '''

    d = len(x0)  # length of parameters
    H0 = np.eye(d)  # initial value of H
    x = x0[:]

    s = []
    y = []

    k = 1
    gradi = grad(f, x0)  # gradient of initial point
    dirc = -H0@gradi  # search direction of initial point
    while np.linalg.norm(gradi) > 1e-5:
        if k > max_it:
            # Stop the loop when the maximum number of iterations is reached
            print('Maximum iterations reached!')
            break

        a = line_search(f, x, dirc, gradi)  # line search
        x_new = x + a * dirc  # Update x use s

        for i in range(len(x_new)):
            if x_new[i] > 100:
                x_new[i] = 100

            if x_new[i] < 0:
                x_new[i] = 0

        gradi_new = grad(f, x_new)  # Calculate new gradient using new x

        if k > m:
            s.pop(0)
            y.pop(0)

        sk = x_new - x
        yk = gradi_new - gradi
        qk = gradi_new
        s.append(sk)
        y.append(yk)

        r = s[0].T@y[0] / (y[0].T@y[0])
        H0 = r * np.eye(d)

        t = len(s)
        a = []
        for i in range(t):
            alpha = s[t-i-1].T@qk / (y[t-i-1].T@s[t-i-1])
            qk = qk - alpha * y[t-i-1]
            a.append(alpha)

        r = H0@qk

        for i in range(t):
            beta = (y[i].T@r) / (y[i].T@s[i])
            r = r + s[i] * (a[t-i-1] - beta)

        # if(yk.T@sk > 0):
        #     dirc = -r
        #     print("change direction!!!")

        dirc = -r
        gradi = gradi_new
        x = x_new

        k = k + 1

        # print(k)

    return x

