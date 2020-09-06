import numpy as np
from layers import *

class RNN(object):
    def __init__(self, vocab_dim, idx_to_char, input_dim=30, hidden_dim=25, cell_type='lstm'):
        """Takes as arguments
        vocab_dim: The number of unique characters/words in the dataset
        idx_to_char: A dictionary converting integer representations of vocabulary to string form.
                    Mainly used in the sample function in order to neatly output results
        input_dim: Reduces one-hot encoding dimension of character from size vocab_dim to a vector of size input_dim
        hidden_dim: Size of hidden dimension
        cell_type: must choose one of 'lstm' or 'vanilla'
        Automatically intiializes all weights"""
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vocab_dim = vocab_dim
        self.cell_type = cell_type
        self.idx_to_char = idx_to_char
        if cell_type != 'lstm' and cell_type != 'vanilla':
            raise ValueError('Invalid cell type. Please choose lstm or vanilla')
        self.dim_mul = (1 if cell_type == 'vanilla' else 4)
        # self.idx_to_char = idx_to_char
        self._initialize_params()

    def _initialize_params(self):
        """Initialize all weights. We use He normalization for weights and
        zeros for biases. We also intialize zeroth hidden layer h0"""
        D, H, V = self.input_dim, self.hidden_dim, self.vocab_dim
        self.params = {}
        self.params['b'] = np.zeros(self.dim_mul*H)
        self.params['Wx'] = 2*np.random.randn(D,self.dim_mul*H)/np.sqrt(D)
        self.params['Wh'] = 2*np.random.randn(H,self.dim_mul*H)/np.sqrt(H)
        self.params['b_out'] = np.zeros(V)
        self.params['W_out'] = 2*np.random.randn(H,V)/np.sqrt(H)
        self.params['W_embed'] = 2*np.random.randn(V,D)/np.sqrt(V)
        self.h0 = np.random.randn(1,H)


    def loss(self, inputs, targets):
        """inputs: an array of size (N,T,D), for N the minibatch size, T the sequence length, and D the input_dim
        targets: an array of size (N,T) consisting of integers in [0,vocab_dim). Each value is the target characters
                given the (N,T)^th input.
        Outputs:
            Loss -> the loss function taken over all N and T
            grads -> a dictionary containing the gradients of all parameters in self.parameters
        """
        loss, grads = 0, {}
        # VERY IMPORTANT. We must name the items in grads identical to their names in self.params!

        # Unpack params
        b = self.params['b']
        b_out = self.params['b_out']
        Wx = self.params['Wx']
        Wh = self.params['Wh']
        W_out = self.params['W_out']
        W_embed = self.params['W_embed']


        # x is a sequence of integers of length T, with integers in [0,V).
        # we can always change this input later if we choose
        # We use an embedding matrix W_embed: (N,T) -> (N,T,D) that generalizes the one-hot-encoding
        # i.e. one-hot would be directly x: (N,T) -> (N,T,V) for V size of vocabulary

        # Forward pass
        inputs = (np.expand_dims(inputs, axis=0) if len(inputs.shape)==1 else inputs)
        x, cache_embed = embed_forward(inputs, W_embed)
        h_prev = np.broadcast_to(self.h0,(len(inputs),self.h0.shape[1]))
        h, cache_h = (lstm_all_forward(x, Wx, b, h_prev, Wh) if self.cell_type=='lstm' else vanilla_all_forward(x, Wx, b, h_prev, Wh))
        probs, cache_probs = affine_all_forward(h, W_out, b_out)
        loss, dprobs = softmax_loss_all(probs, targets)

        # Backward pass
        dh, grads['W_out'], grads['b_out'] = affine_all_backward(dprobs, cache_probs)
        dx, grads['Wx'], grads['b'], grads['Wh'] = (lstm_all_backward(dh, cache_h) if self.cell_type=='lstm' else vanilla_all_backward(dh, cache_h))
        grads['W_embed'] = embed_backward(dx, cache_embed)

        # reset memory layer to last in batch, last in sequence
        self.h0 = h[-1,-1,:].reshape(1,-1)

        # return loss and gradient
        return loss, grads

    def sample(self, seed_idx=None, T=200, h0=None, p_power=1):
        """Inputs: seed_idx=None -> the starting character index for the generated sequences
        T=200 -> the default length of sequence to output
        h0=self.h0 -> the current memory, i.e. initial hidden state. Defaults to last computed h0
        p_power=1 -> raises probability distribution of next character by power p_power.
                    higher p_power produces more deterministic, higher prob words.
                    Will result in short repeating sequences, but with well-defined words"""
        if h0 is None:
            h0 = self.h0
        if seed_idx is None:
            seed_idx = np.random.choice(self.vocab_dim)

        #initialize word
        idxs = [seed_idx]

        # unpack weights
        b = self.params['b']
        b_out = self.params['b_out']
        Wx = self.params['Wx']
        Wh = self.params['Wh']
        W_out = self.params['W_out']
        W_embed = self.params['W_embed']

        # Forward pass only
        x, _ = embed_forward(seed_idx, W_embed)
        x = np.expand_dims(x, axis=0)
        c = np.zeros_like(h0)
        for t in range(T):
            if self.cell_type == 'lstm':
                c, h0, _ = lstm_forward(x, Wx, b, h0, Wh, c)
            else:
                h0, _ = vanilla_forward(x, Wx, b, h0, Wh)
            probs, _ = affine_forward(h0, W_out, b_out)
            probs = np.squeeze(probs)

            # predict next entry
            probs = np.exp(probs-np.max(probs))
            probs = probs**p_power
            probs /= np.sum(probs)
            idx = np.random.choice(np.arange(len(probs)),p=probs.ravel())
            idxs.append(idx)
            x, _ = embed_forward(idx, W_embed)
            x = np.expand_dims(x, axis=0)

        # return index list
        print(''.join([self.idx_to_char[i] for i in idxs]))
