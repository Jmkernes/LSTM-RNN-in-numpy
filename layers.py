import numpy as np

def sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def embed_forward(inputs, W):
    """Inputs: inputs (N,T), with values in [0,V), W (V, D). Outputs: x (N, T, D), cache.
    May work with N=1 as well"""
    x = W[inputs]
    return x, (inputs, W)

def embed_backward(dx, cache):
    """Inputs: dx (N,T,D), cache. Outputs: dW_embed (V,D).
    Only need to compute dW_embed as output"""
    inputs, W = cache
    dW = np.zeros_like(W)
    np.add.at(dW, inputs, dx)
    return dW

def affine_forward(x, W, b):
    """Inputs: x (N,H), W, (H,V), b (V,)
    Outputs: hidden layer (N,H) and cache for backward pass"""
    h =  b + x.dot(W)
    cache = x, W
    return h, cache

def affine_backward(dout, cache):
    """dout (H, V), Outputs: dh (N, T, H), dW_out (H, V), db_out (V)"""
    x, W = cache
    dx, dW, db, = dout.dot(W.T), x.T.dot(dout), np.sum(dout, axis=0)
    return dx, dW, db

def affine_all_forward(h, W, b):
    """h (N,T,H), W (H, V), b (V) """
    N, T, H = h.shape
    H, V = W.shape
    out = np.zeros((N,T,V))
    cache = [0]*T
    for t in range(T):
        out[:,t,:], cache[t] = affine_forward(h[:,t,:], W, b)
    return out, cache

def affine_all_backward(dprobs, cache):
    """Outputs dh, dW_out, db_out"""
    N, T, V = dprobs.shape
    _, H = cache[0][0].shape
    dx = np.zeros((N,T,H))
    dW, db = 0, 0
    for t in reversed(range(T)):
        dx[:,t,:], dW, db = tuple(map(sum,zip((0,dW,db),affine_backward(dprobs[:,t,:],cache[t]))))
    return dx, dW, db

def vanilla_forward(x, Wx, b, h_prev, Wh):
    """ y_prev    y_curr   y_next
            |        |       |
        h_prev -> h_curr -> h_next
            |        |       |
        x_prev    x_curr   x_next
        propagate from h_prev, x_curr to h_curr
        Inputs: x_curr (N,D), h_prev (N,H), Wx (D,H), Wh (H,H)
        Outputs: h_curr, cache"""
    if len(h_prev.shape) == 1:
        h_prev = h_prev.reshape(1,-1)
    h_curr = np.tanh(b + h_prev.dot(Wh) + x.dot(Wx))
    cache = x, Wx, h_prev, h_curr, Wh
    return h_curr, cache

def vanilla_backward(dh_curr, cache):
    """Two upstream derivatives from y_curr and h_next.
    Inputs: dh_curr (H,H), Outputs: dx_curr, dWx, db, dh_prev, dWh"""
    x_curr, Wx, h_prev, h_curr, Wh = cache
    htan = (1-h_curr**2)
    dh_curr = dh_curr*htan
    dx_curr = dh_curr.dot(Wx.T)
    dh_prev = dh_curr.dot(Wh.T)
    dWx = x_curr.T.dot(dh_curr)
    dWh = h_prev.T.dot(dh_curr)
    db = np.sum(dh_curr, axis=0)
    return dx_curr, dWx, db, dh_prev, dWh

def lstm_forward(x, Wx, b, h_prev, Wh, c_prev):
    """Includes a gate layer c of the same shape as hidden layer h.
    Inputs: x (N,D), Wx (D,4H), b (4H), h_prev (N,H), Wh (H,4H), c_prev (N,H)
    Outputs: c_curr, h_curr, cache"""
    if len(h_prev.shape) == 1:
        h_prev = h_prev.reshape(1,-1)
    N, H = h_prev.shape
    gate = b + x.dot(Wx) + h_prev.dot(Wh)
    gate[:,:3*H], gate[:,3*H:] = sigmoid(gate[:,:3*H]), np.tanh(gate[:,3*H:])
    i, f, o, g = gate[:,:H], gate[:,H:2*H], gate[:,2*H:3*H], gate[:,3*H:]
    c = f*c_prev + i*g
    h = o*np.tanh(c)
    cache = x, Wx, h_prev, Wh, i, f, o, g, c_prev, c
    return c, h, cache

def lstm_backward(dc, dh, cache):
    """Two upstream derivatives for dc, dh flow into lstm gate.
    Inputs: dc (N,H), dh (N,H), cache
    Outputs: dx (N,D), dWx (N,4H), db (4H), dh_prev (N,4H),
    dWh (N,4H), dc_prev (N, H)"""
    x, Wx, h_prev, Wh, i, f, o, g, c_prev, c = cache
    d_up = (dc + dh*o*(1-np.tanh(c)**2))
    dc_prev = d_up*f
    di = d_up*g*i*(1-i)
    df = d_up*c_prev*f*(1-f)
    do = dh*np.tanh(c)*o*(1-o)
    dg = d_up*i*(1-g**2)
    dgate = np.hstack([di,df,do,dg])
    dh_prev = dgate.dot(Wh.T)
    dx = dgate.dot(Wx.T)
    dWh = h_prev.T.dot(dgate)
    dWx = x.T.dot(dgate)
    db = np.sum(dgate, axis=0)
    return dx, dWx, db, dh_prev, dWh, dc_prev

def vanilla_all_forward(x, Wx, b, h_prev, Wh):
    """Inputs: x, Wx, b, h_prev, Wh. Outputs: h (N,T,D), caches (T)
    output is a list of h at each time step, and the corresponding cache"""
    N, T, D = x.shape
    _, H = h_prev.shape
    h = np.zeros((N,T,H))
    caches = [0]*T
    for t in range(T):
        h_prev, cache = vanilla_forward(x[:,t,:], Wx, b, h_prev, Wh)
        h[:,t,:] = h_prev
        caches[t] = cache
    return h, caches

def vanilla_all_backward(dh, caches):
    """Inputs: dh (N,T,D) upstream derivative from final affine layer
    at all time steps. Output: dx, dWx, db, dWh"""
    N, T, H = dh.shape
    _, D = caches[0][0].shape
    dx = np.zeros((N,T,D))
    dx[:,-1,:], dWx, db, dh_next, dWh = vanilla_backward(dh[:,-1,:], caches[-1])
    for t in reversed(range(T-1)):
        g = vanilla_backward(dh[:,t,:]+dh_next, caches[t])
        dx[:,t,:], dWx, db, dh_next, dWh = tuple(map(sum,zip((0, dWx, db, 0, dWh), g)))
    return dx, dWx, db, dWh

def lstm_all_forward(x, Wx, b, h_prev, Wh):
    """Inputs: x, Wx, b, h_prev, Wh. Outputs: c (N,T,D), h (N,T,D), caches (T)
    output is a list of h and c at each time step, and the corresponding cache"""
    N, T, D = x.shape
    _, H = h_prev.shape
    c = np.zeros((N,T,H))
    h = np.zeros((N,T,H))
    c_prev = np.zeros((N,H))
    caches = [0]*T
    for t in range(T):
        c_prev, h_prev, cache = lstm_forward(x[:,t,:], Wx, b, h_prev, Wh, c_prev)
        c[:,t,:] = c_prev
        h[:,t,:] = h_prev
        caches[t] = cache
    return h, caches

def lstm_all_backward(dh, caches):
    """Inputs: dh (N,T,D) upstream derivative from final affine layer
    at all time steps. Output: dx, dWx, db, dWh"""
    N, T, H = dh.shape
    _, D = caches[0][0].shape
    dc_next = 0
    dx = np.zeros((N,T,D))
    dx[:,-1,:], dWx, db, dh_next, dWh, dc_next = lstm_backward(dc_next, dh[:,-1,:], caches[-1])
    for t in reversed(range(T-1)):
        g = lstm_backward(dc_next, dh[:,t,:]+dh_next, caches[t])
        dx[:,t,:], dWx, db, dh_next, dWh, dc_next = tuple(map(sum,zip((0,dWx,db,0,dWh,0),g)))
    return dx, dWx, db, dWh

def softmax_loss(x, y):
    """Inputs: x (N,D), y (N). x is score matrix, y is list of N integers
    with values y[i] in [0,D) of correct output.
    Outputs: loss, dx"""
    N = len(y)
    probs = np.exp(x-np.max(x, axis=1, keepdims=True))
    norms = np.sum(probs, axis=1, keepdims=True)
    probs = probs/norms
    loss = -np.sum(np.log(probs[np.arange(N),y]))/N
    dx = probs
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx

def softmax_loss_all(x, y):
    loss, dprobs = softmax_loss(x.reshape(-1, x.shape[-1]), y.reshape(-1))
    return loss, dprobs.reshape(x.shape)
