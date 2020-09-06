from gradient_checking import *
import unittest
import numpy as np
from layers import *

# class TestNumGrad(unittest.TestCase):
#   def test_num_grad(self):

if False:
    # Matrix multiplication
    N = 13
    D = 8
    H = 11
    A = np.random.randn(N,D)
    B = np.random.randn(D,H)
    dout = np.random.randn(N,H)
    dA = dout.dot(B.T)
    dB = A.T.dot(dout)

    func = lambda A: A.dot(B)
    numA = num_grad(func, A, dout, fivept=True)
    func = lambda B: A.dot(B)
    numB = num_grad(func, B, dout, fivept=True)

    print('-'*40)
    print("Test matrix multiplication A*B")
    print("Derivative dA: ",rel_error(numA, dA))
    print("Derivative dB: ",rel_error(numB, dB))
    # self.assertAlmostEqual(numA, analyticA)
    # self.assertAlmostEqual(numB, analyticB)

    # vector dot product
    A = np.random.randn(7)
    B = np.random.randn(7)
    dout = np.random.randn()

    func = lambda A: A.dot(B)
    numA = num_grad(func, A, dout)
    func = lambda B: A.dot(B)
    numB = num_grad(func, B, dout)

    print("\nTest vector dot product A*B")
    print("dA error: ",rel_error(numA,dout*B))
    print("dB error: ",rel_error(numB,dout*A.T))
    # print(dout*B, numA)

if True:
    N, T, H, D = 4, 5, 6, 7
    print('-'*40)
    print("Test affine_backward")
    x = np.random.randn(N,D)
    W = np.random.randn(D,H)
    b = np.random.randn(H)
    dout = np.random.randn(N,H)

    h, cache = affine_forward(x, W, b)
    dx, dW, db = affine_backward(dout, cache)

    funcx = lambda x: affine_forward(x, W, b)[0]
    funcW = lambda W: affine_forward(x, W, b)[0]
    funcb = lambda b: affine_forward(x, W, b)[0]

    num_dx = num_grad(funcx, x, dout)
    num_dW = num_grad(funcW, W, dout)
    num_db = num_grad(funcb, b, dout)

    print(f"Testing x=({N,D}), W=({D,H}), b=({H})")
    print("dx error:", rel_error(num_dx, dx))
    print("dW error:", rel_error(num_dW, dW))
    print("db error:", rel_error(num_db, db))

if True:
    print('-'*40)
    print("Test affine_all_backward")
    N, T, H, V = 4, 5, 6, 7
    x = np.random.randn(N,T,H)
    W = np.random.randn(H,V)
    b = np.random.randn(V)
    dout = np.random.randn(N,T,V)

    p, cache = affine_all_forward(x, W, b)
    dx, dW, db = affine_all_backward(dout, cache)

    funcx = lambda x: affine_all_forward(x, W, b)[0]
    funcW = lambda W: affine_all_forward(x, W, b)[0]
    funcb = lambda b: affine_all_forward(x, W, b)[0]

    num_dx = num_grad(funcx, x, dout)
    num_dW = num_grad(funcW, W, dout)
    num_db = num_grad(funcb, b, dout)

    print(f"Testing x=({N,D}), W=({D,H}), b=({H})")
    print("dx error:", rel_error(num_dx, dx))
    print("dW error:", rel_error(num_dW, dW))
    print("db error:", rel_error(num_db, db))

##############################################
if False:
    N, D, H = 13, 8, 11
    print('-'*40)
    print("Test vanilla_backward")
    x = np.random.randn(N,D)
    Wx = np.random.randn(D,H)
    Wh = np.random.randn(H,H)
    h_prev = np.random.randn(N,H)
    b = np.random.randn(H)
    dout = np.random.randn(N,H)

    h, cache = vanilla_forward(x, Wx, b, h_prev, Wh)
    dx, dWx, db, dh_prev, dWh = vanilla_backward(dout, cache)

    funcx = lambda x: vanilla_forward(x, Wx, b, h_prev, Wh)[0]
    funcWx = lambda Wx: vanilla_forward(x, Wx, b, h_prev, Wh)[0]
    funcb = lambda b: vanilla_forward(x, Wx, b, h_prev, Wh)[0]
    funch = lambda h_prev: vanilla_forward(x, Wx, b, h_prev, Wh)[0]
    funcWh = lambda Wh: vanilla_forward(x, Wx, b, h_prev, Wh)[0]

    num_dx = num_grad(funcx, x, dout)
    num_dWx = num_grad(funcWx, Wx, dout)
    num_db = num_grad(funcb, b, dout)
    num_dh = num_grad(funch, h_prev, dout)
    num_dWh = num_grad(funcWh, Wh, dout)

    print(f"Testing x=({N,D}), Wx=({D,H}), b=({H}), h_prev=({N,H}), Wh=({H,H})")
    print("dx error:", rel_error(num_dx, dx))
    print("dWx error:", rel_error(num_dWx, dWx))
    print("db error:", rel_error(num_db, db))
    print("dh_prev error:", rel_error(num_dh, dh_prev))
    print("dWh error:", rel_error(num_dWh, dWh))

##############################################
if False:
    print('-'*40)
    print("Test lstm_backward. There are two outputs c and h, \\\
    so the total gradient is the sum from each contribution")
    N, D, H = 7,6,8
    x = np.random.randn(N,D)
    Wx = np.random.randn(D,4*H)
    Wh = np.random.randn(H,4*H)
    h_prev = np.random.randn(N,H)
    c_prev = np.random.randn(N,H)
    b = np.random.randn(4*H)
    dc = np.random.randn(N,H)
    dh = np.random.randn(N,H)

    c, h, cache = lstm_forward(x, Wx, b, h_prev, Wh, c_prev)
    dx, dWx, db, dh_prev, dWh, dc_prev = lstm_backward(dc, dh, cache)

    funcx1 = lambda x: lstm_forward(x, Wx, b, h_prev, Wh, c_prev)[0]
    funcx2 = lambda x: lstm_forward(x, Wx, b, h_prev, Wh, c_prev)[1]

    funcWx1 = lambda Wx: lstm_forward(x, Wx, b, h_prev, Wh, c_prev)[0]
    funcWx2 = lambda Wx: lstm_forward(x, Wx, b, h_prev, Wh, c_prev)[1]

    funcb1 = lambda b: lstm_forward(x, Wx, b, h_prev, Wh, c_prev)[0]
    funcb2 = lambda b: lstm_forward(x, Wx, b, h_prev, Wh, c_prev)[1]

    funch1 = lambda h_prev: lstm_forward(x, Wx, b, h_prev, Wh, c_prev)[0]
    funch2 = lambda h_prev: lstm_forward(x, Wx, b, h_prev, Wh, c_prev)[1]

    funcWh1 = lambda Wh: lstm_forward(x, Wx, b, h_prev, Wh, c_prev)[0]
    funcWh2 = lambda Wh: lstm_forward(x, Wx, b, h_prev, Wh, c_prev)[1]

    funcc1 = lambda c: lstm_forward(x, Wx, b, h_prev, Wh, c_prev)[0]
    funcc2 = lambda c: lstm_forward(x, Wx, b, h_prev, Wh, c_prev)[1]

    num_dx = num_grad(funcx1, x, dc) + num_grad(funcx2, x, dh)
    num_dWx = num_grad(funcWx1, Wx, dc) + num_grad(funcWx2, Wx, dh)
    num_db = num_grad(funcb1, b, dc) + num_grad(funcb2, b, dh)
    num_dh = num_grad(funch1, h_prev, dc) + num_grad(funch2, h_prev, dh)
    num_dWh = num_grad(funcWh1, Wh, dc) + num_grad(funcWh2, Wh, dh)
    num_dc = num_grad(funcc1, c_prev, dc) + num_grad(funcc2, c_prev, dh)


    print(f"Testing x={N,D}, Wx={D,H}, b={H}, h_prev={N,H}, Wh={H,H}")
    print("dx error:", rel_error(num_dx, dx))
    print("dWx error:", rel_error(num_dWx, dWx))
    print("db error:", rel_error(num_db, db))
    print("dh_prev error:", rel_error(num_dh, dh_prev))
    print("dWh error:", rel_error(num_dWh, dWh))
    print("dc_prev error:", rel_error(num_dc, dc_prev))

##############################################
if False:
    print('-'*40)
    print("Test softmax_loss.")
    N, D, H = 7,6,8
    y = np.random.randint(D,size=N)
    x = np.random.randn(N,D)

    loss, dx = softmax_loss(x,y)

    func = lambda x: softmax_loss(x,y)[0]
    num_dx = num_grad(func, x, 1)

    print(f"Testing x={N,D}, y = {N,}")
    print(f"Loss should be ~{np.log(N)}, computed value:{loss}")
    print(f"dx error: {rel_error(num_dx, dx)}")

##############################################
if False:
    print('-'*40)
    print("Test vanilla_all_backward")
    N, T, D, H = 5,6,3,7
    x = np.random.randn(N,T,D)
    Wx = np.random.randn(D,H)*.01
    Wh = np.random.randn(H,H)*.01
    h_prev = np.random.randn(N,H)
    b = np.random.randn(H)*.01
    dh = np.random.randn(N,T,H)

    h, cache = vanilla_all_forward(x, Wx, b, h_prev, Wh)
    dx, dWx, db, dWh = vanilla_all_backward(dh, cache)

    funcx = lambda x: vanilla_all_forward(x, Wx, b, h_prev, Wh)[0]
    funcWx = lambda Wx: vanilla_all_forward(x, Wx, b, h_prev, Wh)[0]
    funcb = lambda b: vanilla_all_forward(x, Wx, b, h_prev, Wh)[0]
    funcWh = lambda Wh: vanilla_all_forward(x, Wx, b, h_prev, Wh)[0]

    num_dx = num_grad(funcx, x, dh)
    num_dWx = num_grad(funcWx, Wx, dh)
    num_db = num_grad(funcb, b, dh)
    num_dWh = num_grad(funcWh, Wh, dh)

    # print(num_dWx.shape)
    # print(num_dx.shape,dx.shape)

    # print(f"Testing x=({N,D}), Wx=({D,H}), b=({H}), h_prev=({N,H}), Wh=({H,H})")
    print("dx error:", rel_error(num_dx, dx))
    print("dWx error:", rel_error(num_dWx, dWx))
    print("db error:", rel_error(num_db, db))
    print("dWh error:", rel_error(num_dWh, dWh))

##############################################
if False:
    # Note there is only one backprop input (dh) in contrast to the single layer case
    # we then only care about derivative w.r.t upstream dh.
    print('-'*40)
    print("Test lstm_all_backward.")
    N, T, D, H = 7,4,6,8
    x = np.random.randn(N,T,D)
    Wx = np.random.randn(D,4*H)*.01
    Wh = np.random.randn(H,4*H)*.01
    h_prev = np.random.randn(N,H)
    b = np.random.randn(4*H)*.01
    dc = np.random.randn(N,T,H)
    dh = np.random.randn(N,T,H)

    h, cache = lstm_all_forward(x, Wx, b, h_prev, Wh)
    dx, dWx, db, dWh = lstm_all_backward(dh, cache)

    funcx2 = lambda x: lstm_all_forward(x, Wx, b, h_prev, Wh)[0]
    funcWx2 = lambda Wx: lstm_all_forward(x, Wx, b, h_prev, Wh)[0]
    funcb2 = lambda b: lstm_all_forward(x, Wx, b, h_prev, Wh)[0]
    funcWh2 = lambda Wh: lstm_all_forward(x, Wx, b, h_prev, Wh)[0]

    num_dx = num_grad(funcx2, x, dh)
    num_dWx = num_grad(funcWx2, Wx, dh)
    num_db = num_grad(funcb2, b, dh)
    num_dWh = num_grad(funcWh2, Wh, dh)

    # print(f"Testing x={N,D}, Wx={D,H}, b={H}, h_prev={N,H}, Wh={H,H}")
    print("dx error:", rel_error(num_dx, dx))
    print("dWx error:", rel_error(num_dWx, dWx))
    print("db error:", rel_error(num_db, db))
    print("dWh error:", rel_error(num_dWh, dWh))

if False:
    print('-'*40)
    print("Testing W_embed backward derivative")
    N, T, D, V = 5, 10, 4, 9
    inputs = np.random.randint(V, size =[N,T])
    W_embed = np.random.randn(V,D)*.01
    dout = np.random.randn(N,T,D)*.01

    x, cache = embed_forward(inputs, W_embed)
    dW_embed = embed_backward(dout, cache)

    func = lambda W: embed_forward(inputs, W_embed)[0]
    num_dW_embed = num_grad(func, W_embed, dout)

    print("dW_embed error:", rel_error(num_dW_embed, dW_embed))
