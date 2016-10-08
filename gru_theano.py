import numpy as np
import theano as theano
import theano.tensor as T
from theano.gradient import grad_clip
import time
import operator

class GRUTheano:
    
    def __init__(self, word_dim, hidden_dim=128, bptt_truncate=-1):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Initialize the network parameters
        E = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (128, word_dim))
        U = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (3, hidden_dim, 128))
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (3, hidden_dim, hidden_dim))
	V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (128, hidden_dim))
        b = np.zeros((3, hidden_dim))
        c = np.zeros(hidden_dim)
        # Theano: Created shared variables
        self.E = theano.shared(name='E', value=E.astype(theano.config.floatX))
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
	self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
        # SGD / rmsprop: Initialize parameters
        self.mU = theano.shared(name='mU', value=np.zeros(U.shape).astype(theano.config.floatX))
        self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
	self.mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(theano.config.floatX))
        self.mb = theano.shared(name='mb', value=np.zeros(b.shape).astype(theano.config.floatX))
        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()
    
    def __theano_build__(self):
        E, U, W, V, b = self.E, self.U, self.W, self.V, self.b
        
        x = T.iscalar('x')
        y = T.dmatrix('y')
        
        def forward_prop_step(s_t1_prev, f_t1_prev, x_t):
            # This is how we calculated the hidden state in a simple RNN. No longer!
            # s_t = T.tanh(U[:,x_t] + W.dot(s_t1_prev))
            
            # Word embedding layer
            x_e = E[:,x_t]
            
            # GRU Layer 1
            z_t1 = T.nnet.hard_sigmoid(U[0].dot(x_e) + W[0].dot(s_t1_prev) + b[0])
            r_t1 = T.nnet.hard_sigmoid(U[1].dot(x_e) + W[1].dot(s_t1_prev) + b[1])
            c_t1 = U[2].dot(x_e) + W[2].dot(s_t1_prev * r_t1) + b[2]
            s_t1 = (T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev
	    f_t1 = T.nnet.hard_sigmoid(V.dot(s_t1)) + 0.001
	    
            return [s_t1, f_t1]
        
        outputs_info_s = T.as_tensor_variable(np.zeros((self.hidden_dim), dtype=theano.config.floatX))
	outputs_info_f = T.as_tensor_variable(np.zeros(128, dtype=theano.config.floatX))
        [s, f], updates = theano.scan(
            forward_prop_step,
            outputs_info=[outputs_info_s, outputs_info_f],
            non_sequences=x,
            n_steps=32)
	
        o_error = T.sum(T.nnet.binary_crossentropy(f, y))
        # Total cost (could add regularization here)
        cost = o_error
        
        # Gradients
        dU = T.grad(cost, U)
        dW = T.grad(cost, W)
	dV = T.grad(cost, V)
        db = T.grad(cost, b)
        
        # Assign functions
        self.ce_error = theano.function([x, y], cost)
        self.bptt = theano.function([x, y], [dU, dW, dV, db])
	
        # SGD parameters
        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')
        
        # rmsprop cache updates
        mU = decay * self.mU + (1 - decay) * dU ** 2
        mW = decay * self.mW + (1 - decay) * dW ** 2
	mV = decay * self.mV + (1 - decay) * dV ** 2
        mb = decay * self.mb + (1 - decay) * db ** 2
        
        self.sgd_step = theano.function(
            [x, y, learning_rate, theano.In(decay, value=0.9)],
            [f], 
            updates=[(U, U - (learning_rate * dU / T.sqrt(mU + 1e-6))),
                     (W, W - (learning_rate * dW / T.sqrt(mW + 1e-6))),
		     (V, V - (learning_rate * dV / T.sqrt(mV + 1e-6))),
                     (b, b - (learning_rate * db / T.sqrt(mb + 1e-6))),
                     (self.mU, mU),
                     (self.mW, mW),
		     (self.mV, mV),
                     (self.mb, mb)
                    ])
        
        
    def calculate_total_loss(self, X, Y):
        return self.ce_error(X, Y)
    
    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        return self.calculate_total_loss(X,Y) / 32

