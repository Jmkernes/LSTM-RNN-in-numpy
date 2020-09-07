import numpy as np
from optim import *

class Solver:
    def __init__(self, data, model, seq_length=25, batch_size=8, lr=1e-1, num_iters=2000, print_every=100, update_rule='RMSProp'):
        """A helper function that can be used to train a given model.
        Inputs: data -> must be a 1D array of integers representing characters.
                model -> a class instance of the model to train. This will NOT reset model parameters
                        this way, we can stop training, and pick back up where we left off (however,
                        optimizer moving averages will reset)
                seq_length -> length of input sequence
                batch_size -> We train batch_size number of successive sequences concurrently
                lr -> learning rate for gradient descent
                num_iters -> number of iterations
                print_every -> Prints the current loss every print_every number of iterations.
                                prints a sample sentence every 10*print_every iterations
                """
        self.data = data
        self.model = model
        self.seq_length = seq_length
        self.lr = lr
        self.batch_size = batch_size
        self.num_iters = num_iters
        self.print_every = print_every
        self.update_rule = update_rule
        self.optimizer = Optimizer(self.model, lr=self.lr)
        self.loss_history = []

    def _minibatch_generator(self):
        """Cuts data into batches
        of size batch_size, then creates a generator that yields
        the next batch, taking care to reset after one epoch"""
        N, T = self.batch_size, self.seq_length
        # make sure data is an even split of sequences
        data = self.data[:(T+1)*(len(self.data)//(T+1))]
        data = np.split(data, len(self.data)//(T+1))
        i = 0
        while True:
            if i > len(data)-N:
                i = 0
            a = np.hstack(data[i:i+N]).reshape(N,-1)
            yield np.array(a[:,:-1]), np.array(a[:,1:])
            i += N

    def train(self):
        """Trains the model over num_iters iterations. Updates the array loss_history to later output.
        You must manually specify the choice of optimizer here, replacing optimizer.RMSprop(grads) with
        optimizer.(...). See optim.py for options. We implement learning rate decay every 500 iterations"""
        mini_batch = self._minibatch_generator()
        for i in range(self.num_iters):#epoch_size*self.num_epochs):
            inputs, targets = next(mini_batch)
            loss, grads = self.model.loss(inputs, targets)
            self.loss_history.append(loss)
            if self.update_rule == 'RMSProp':
                self.optimizer.RMSProp(grads)
            elif self.update_rule == 'adam':
                self.optimizer.adam(grads, slow_start=False)
            elif self.update_rule == 'SGD':
                self.optimizer.SGD(grads)
            if i % self.print_every == 0:
                print(f"Iteration {i}/{self.num_iters}. Loss = {loss}")
            if i % (self.print_every*3) == 0:
                print(f"{self.model.sample(np.random.choice(self.model.vocab_dim), T=60)}")
            # learning rate decay
            if i % 500 == 0:
                self.lr *= 0.95

    def plot_loss(self):
        """Plots the loss over all training iterations"""
        epoch_size = len(self.data)//((self.seq_length+1)*self.batch_size)
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.loss_history))/epoch_size,self.loss_history)
        plt.xlabel("Number of Epochs")
        plt.ylabel("Loss")
        plt.show()
