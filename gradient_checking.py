import numpy as np

def num_grad(func, A, dout, eps=1e-4, fivept=True):
    """Inputs: function func that takes sum(func(A)*dout) -> scalar. A and dout
    are arrays of equal size. Evaluates dfunc/dA at point A and returns array
    of shape A"""
    dA = np.zeros_like(A)
    # use numpy builtin iterator for fast iteration of array
    it = np.nditer(A, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        old = A[it.multi_index]
        if fivept:
            A[it.multi_index] = old - 2*eps
            f1 = func(A)
            A[it.multi_index] = old - eps
            f2 = func(A)
            A[it.multi_index] = old + eps
            f3 = func(A)
            A[it.multi_index] = old + 2*eps
            f4 = func(A)
            dA[it.multi_index] = (1/(12*eps))*np.sum((-f4+8*f3-8*f2+f1)*dout)
            A[it.multi_index] = old
        else:
            A[it.multi_index] = old + eps/2
            fp = func(A)
            A[it.multi_index] = old - eps/2
            fm = func(A)
            A[it.multi_index] = old
            dA[it.multi_index] = (1/eps)*np.sum((fp-fm)*dout)
        it.iternext()
    return dA

def rel_error(A1, A2, use_max=False):
    A1.astype(np.float64)
    A2.astype(np.float64)
    if use_max:
        num = np.max(np.abs(A1-A2))
    else:
        num = np.sum(np.abs(A1-A2))
    den = np.sum(np.abs(A1)+np.abs(A2))/2
    return num/den
