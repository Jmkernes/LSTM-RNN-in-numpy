import numpy as np

class Optimizer:
    def __init__(self, model, lr=1e-3):
        """ Different optimizers used for update rules
        Inputs: model -> the current model, lr -> the learning rate"""
        self.model = model
        self.lr = lr
        self.s = {}
        self.v = {}
        self.iter_num = 1
        for k in model.params.keys():
            self.s[k], self.v[k] = 0, 0

    def SGD(self, grads, clip=True):
        """Inputs: grads -> dictionary of gradients. Normal stochastic gradient descent.
        Option to clip gradients betwen -5 an 5 setting clip =True"""
        for k in self.model.params.keys():
            delta = (np.clip(self.lr*grads[k],-5,5) if clip else self.lr*grads[k])
            self.model.params[k] -= delta
        self.iter_num += 1

    def RMSProp(self, grads, beta2=0.9, eps=1e-6):
        """Inputs: grads -> gradient dictionary, beta2=0.9, rmsprop decay parameter"""
        for k in self.model.params.keys():
            self.s[k] = beta2*self.s[k] + (1-beta2)*(grads[k]**2)
            self.model.params[k] -= self.lr*grads[k]/(eps+np.sqrt(self.s[k]))
        self.iter_num += 1

    def adam(self, grads, beta1=0.9, beta2=0.999, slow_start=True, eps=1e-5, clip=True):
        """ inputs: grads -> gradient dictioary, beta1=0.9, beta2=0.9, slow_start=True,eps=1e-5"""
        for k in self.model.params.keys():
            self.v[k] = beta1*self.v[k] + (1-beta1)*grads[k]
            self.s[k] = beta2*self.s[k] + (1-beta2)*(grads[k]**2)
            if not slow_start:
                self.v[k] /= (1-beta1**self.iter_num)
                self.s[k] /= (1-beta2**self.iter_num)
            delta = self.lr*self.v[k]/(eps+np.sqrt(self.s[k]))
            self.model.params[k] -= (np.clip(delta,-5,5) if clip else delta)
        self.iter_num += 1

    def adagrad(self, grads, eps=1e-8, clip=True):
        for k in self.model.params.keys():
            self.s[k] += grads[k]**2
            delta = self.lr*grads[k]/(eps+np.sqrt(self.s[k]))
            self.model.params[k] -= (np.clip(delta,-5,5) if clip else delta)
        self.iter_num += 1
