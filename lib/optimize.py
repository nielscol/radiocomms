""" Methods for numerical optimization
    Cole Nielsen 2019
"""
from math import sqrt
import numpy as np
from copy import copy
from time import clock

######################################################################################
# Golden section search algorithm
######################################################################################

def gss(func, arg, params, _min, _max, target=0.0, conv_tol=1e-5, norm="l1", timeout=None):
    """Golden section search minimization algorithm. Finds arg = argmin(func).
    args:
        func - function to minimize
        arg - arg to find argmin(func) for
        params - dictionary with keys and values for other arguments of func
        _min - min value for arg in search
        _max - max value for arg in search
        target - used to find arg = argmin(abs(func - target)). Default 0.0 for minimization.
            i.e. tries to find argument that results in func(arg) = target
        conv_tol - relative difference (convergence) between iteration that should result in function exit
        norm - type of normalization to be used in convergence check. Use {"l1", "l2"}.
    returns:
        value of arg that minimizes func
    """
    params_a = copy(params)
    params_b = copy(params)
    phi = (1.0 + sqrt(5.0))/2.0
    iter = 1
    err = np.inf # Initial error (%)
    guess_l = a = _min
    guess_h = b = _max
    delta = 0
    t0=clock()
    # Iterate until termination criterion is reached
    while err > conv_tol:
        if timeout and clock() - t0 > timeout:
            print("!!!!!! Timeout of %f s exceeded in golden section search. Returning current value..."%timeout)
            break
        delta = (phi - 1.0)*(b - a)
        guess_a = a + delta
        guess_b = b - delta
        params_a[arg] = guess_a
        params_b[arg] = guess_b
        if abs(target - func(**params_b)) < abs(target - func(**params_a)):
            b = guess_a
            val = b
        else:
            a = guess_b
            val = a
        if val == 0.0:
            err = np.inf
        else:
            if norm == "l1":
                err = ((2.0 - phi)*abs((guess_a - guess_b)/val))
            elif norm == "l2":
                err = ((2.0 - phi)*abs((guess_a - guess_b)/val)**2)
            else:
                raise Exception("Invalid value for norm. Either \"l1\" or \"l2\" permissable.")
        params[arg] = val
        iter += 1
    return val

######################################################################################
# Gradient descent using GSS for gamma selection
######################################################################################

def grad(f, args, params, deriv_step):
    vals = np.zeros(len(args))
    _params = copy(params)
    for n, arg in enumerate(args):
        _params[arg] += deriv_step
        vals[n] = (f(**_params) - f(**params))/deriv_step
        _params[arg] -= deriv_step
    return vals


def line_slice_f(f, grad_f, args, params):
    """ Makes parametric line function following the gradient of f at the point
        in the latest iteration of the gradient descent solver
    """
    _params = copy(params)
    def f_slice(gamma):
        """ Line folloing gradient of f passing point p defined in the params dictionary
        """
        for n, arg in enumerate(args):
            _params[arg] = params[arg] - gamma*grad_f[n]
        return f(**_params)
    return f_slice


def grad_descent(f, args, params, conv_tol=1e-5, timeout=None):
    """Gradient descent solver
    args:
        f - input cost function to minimize
        args - arguments of f for which the minimum should be found. Mustn't be all arguments of f.
        params - dictionary with initial values for all arguments of f
        conv_tol - relative difference between iterations for which the solver will exit and return the result
            also used to determine derivative step size
    returns:
        dictionary of all arguments of f which minimize f.
    """
    for n, arg in enumerate(args):
        if arg not in params:
            raise Exception("Please have initial guess for arg %s in params dictionary"%arg)
    curr = np.array([params[arg] for arg in args])
    last = np.full(len(args), np.inf)
    deriv_step = conv_tol*np.linalg.norm(curr)
    t0 = clock()
    while np.linalg.norm(curr-last)/np.linalg.norm(curr) > conv_tol:
        if timeout and clock() - t0 > timeout:
            print("!!!!!! Timeout of %f s exceeded in gradient descent operation. Returning current value..."%timeout)
            break
        grad_f = grad(f, args, params, deriv_step)
        f_slice = line_slice_f(f, grad_f, args, params)
        gamma = gss(f_slice, "gamma", target=0, params={}, _min=0.0, _max=1.0, conv_tol=1e-5, norm='l1')
        last = copy(curr)
        for n, arg in enumerate(args):
            params[arg] = params[arg] - gamma*grad_f[n]
        curr = np.array([params[arg] for arg in args])
        deriv_step = np.linalg.norm(curr-last)*conv_tol
    return params


