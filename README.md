# LSTM-RNN-in-numpy
Implementation of long-short term memory recurrent neural network using only numpy. This is heavily influenced by Andrej Karpathy's min_char model, though with many more features.

To directly get started, you can use the pretrained model "", which was trained over 20,000 iterations at sequence length 50, hidden dimension of 200, input dimension of 1/4 vocab_size, LSTM cells, and minibatch size 16. To import it into a python session, run the commands

import pickle
filehandler = open(filename, 'r') 
model = pickle.load(filehandler)

To generate sentences, you must first supply an input character. To write a sequence, call the method model.sample(char_to_idx[char], T=seq_length), where char is an input character like 'a' and seq_length is the length of the desired sequence. To change this from a print statement to an output, you must change the method under Rnns.py, under the class RNN

The directory layout is as follows:
The file main.ipynb is a Jupyter notebook showing the training of a single model. This is the main working file
Rnns.py contains the RNN class, which instantiates either an LSTM recurrent neural network or a vanilla recurrent neural network
solver.py contains the solver class, which takes as arguments an RNN class and data, and trains the RNN.

The following modules are called by the previous two: layers.py -> contains all of the forward and backward pass functions. The backpropagation gradients are checked numerically via the num_grad function defiend in the file "gradient_checking.py". We can run a full suite of tests on all layer functions by running -> python3 test_gradient_checking.py.
optim.py contains the Optimizer class. This keeps track of moving averages in various update rules. It's methods update parameters via gradient descent provided we supply them with a parameter dictionary (form the model class) and a gradient dictionary. These dictionaries MUST use the keys.

To test our model, we train the RNN on the full text of Jane Austen's Pride and Prejudice, availble for free on the web from gutenberg books. A zip file is contained in pride.zip, as well as text file pride.txt consiting of one long string.

To train a model, open the jupyter notebook. Change the ellipse under open( ... , 'r') to the name of your file, and run the cell. This will extract all unique characters from the text, create a charcter to integer dictionary (and vice versa), then map the text into an array of such integers. This is the default data input to the solver class.

In general, we may choose any input to the solver class, provided it is in the form of an array of integers in the range 0 -> vocab_dim-1. The model has a parameter to reduce the input dimension from vocab_dim to input_dim, representing each one-hot-encoded character as a vector. In this way, we could put in any sequence for the RNN to solve.

After running the initial cell, we isntantiate our model, choose 'lstm' or 'vanilla', and set the size of the input dimensions and hidden layer. We then instantiate the solver class with the learning rate, sequence length, and batch size. The solver hidden method "minibatch_generator" automatically converts the array into mini batches of size batch_size containing sequences of length seq_length.

After training, you can plot the loss function vs. epochs via the method model.plot_loss(), as well as look at some sample sequences generated via model.sample(seed_integer).


