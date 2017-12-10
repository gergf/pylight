import numpy 

class layer():
    def __init__(self, input_shape, trainable, name):
        self.input_shape = input_shape
        self.trainable = trainable 
        self.name = name 

    def forward(self, X): 
        raise NotImplementedError("Please Implement the Forward method") 

    def backward(self, dL_Z): 
        raise NotImplementedError("Please Implement the Backward method") 

class dense(layer):
    def __init__(self, W, b, input_shape, trainable=True, name="Dense"):
        """
            Computes Z = dot(X, W) + b 

            Parameters 
            ----------
                W : Numpy array with shape (sample_features, hidden_units)
                b : Numpy array with shape (1, hidden_units)
                input_shape : Shape of the inputs. Must be equal to sample_features.

        """
        assert len(input_shape) == 1, "Error! In dense layers the input shape only can have one dimension: 'features'"
        assert W.shape[0] == input_shape[0], "Error! Input shape and W matrix do not match."

        # Initialization 
        super().__init__(input_shape, trainable, name)
        self.W = W.astype(numpy.float32)
        self.b = b 

        if not isinstance(self.b, type(None)):
            assert self.b.shape[0] == self.W.shape[1], "Error! Bias and weights shape do not match."
            self.b = self.b.astype(numpy.float32) 

        # Store variables to make the computation easier 
        self.output_shape = (self.W.shape[1],)

        # Store computation  
        self.X = None 
        self.Z = None 

        # Local gradient 
        self.dZ_W = None 
        self.dZ_X = None 
        self.dZ_b = None 

    def forward(self, X): 
        """
            Performs the computation : dot(X, W)

            Parameters 
            ----------
                X : Numpy array with shape  (batch_size, sample_features)
        """
        assert (X.shape[1] == self.input_shape[0]), "Shapes of W and inputs do not match."
        self.X = X.astype(numpy.float32) 

        self.batch_size = self.X.shape[0]
        self.Z = self.X @ self.W 

        if not isinstance(self.b, type(None)): 
            self.Z = self.Z + self.b 

        return self.Z 

    def backward(self, dL_Z):
        """ Given dL/dZ returs dL/dX and stores dL/dW"""
        assert self.Z.shape == dL_Z.shape, "The gradient wrt Z do not have the same shape than Z"
        self.dL_Z   = dL_Z.astype(numpy.float32)
        
        # Local gradient of dZ/dX
        # Remember that all the input samples have the same derivatives of Z wrt X, because all of them are 
        # multiplied by the W, and the W are the same ones for all the samples in one iteration 
        self.dZ_X = self.W 

        # Local gradient of dZ/dW 
        # This is a bit tricky... so first remember some easy concepts. The matrix W is (IUxOU), and 
        # Z is (NXOU), being N : num samples, IU : Input units and OU : Output units (neurons of this layer)
        # The partial derivative of Z wrt to W is a three-dimensional array because by the definition
        # we need to see the rate of change of every Z respect every weight. BUT, one weight only contributes 
        # to ONE NEURON, and its derivative wrt the other neurons is zero!! 
        # So, we can store the non-trivial information of the three-dimensiona array into a two dimensional array. 
        # In summary, the way you should read the array dZ_W is as follows: 
        # | dZ1/dW11   dZ2/dW12 | <- This row is all the W which come from X1, so the derivative is always X1
        # | dZ1/dW21   dZ2/dW22 | <- This row is all the W which come from X2,so the derivative is always X2...
        # Futhermore, we can take advantage of matrix multiplication and compress even more this matrix, because 
        # as you can see above, the rows are always the same value. You can check that the computation is still the 
        # same when the dL/dW is computed a few lines below (;
        self.dZ_W = self.X

        # Local gradient of dZ/db
        # Z is linearly dependent of the biases. If we increase one bias by one; z increases by one. 
        # Remember that each bias only contributes to one neuron, so again, we can reduce the non-trivial 
        # part of the Jacobian matrix to a vector. 
        self.dZ_b = None if isinstance(self.b, type(None)) else numpy.ones(self.b.shape)

        # Output gradient dL/dW 
        # Remember the matrix generated in dZ/dW, now we need to apply the chain rule, or, in other words
        # we need to multiply the rate of change each dZ/dW by the rate of change of dL/dZ
        # dL_Wi,j = sum_{num samples} dL/dZj * dZj/dWi,j
        # Finally, divide by the number of samples to preserve the magnitude of the gradient independently of 
        # the number of samples
        self.dL_W = (self.X.T @ self.dL_Z) / self.batch_size

        # Output gradient dL/dX
        # This could be ignored if there are constraints of memory/time, because we can not modify how our 
        # data looks like. Anyways, we compute the gradients to perform future studies about the behaviour of the model 
        self.dL_X = (self.dL_Z @ self.dZ_X.T) / self.batch_size

        # Output gradient dL/db 
        # Because the local gradient dZ/db is a vector of ones, we do not need to perform the vector multiplication 
        # We only need to reduce dL/dZ among the samples axis, so each sample contributes, and divide by the batch size
        self.dL_b = self.dL_Z.sum(axis=0) / self.batch_size

        return self.dL_X

    def get_trainable_vars_and_grads(self):
        """ Returns a list with (var, grad) for all the trainable variables of this layers """
        vars_and_grads = [(self.W, self.dL_W)]
        
        if not isinstance(self.b, type(None)): 
            vars_and_grads.append((self.b, self.dL_b))

        return vars_and_grads

    def set_trainable_vars(self, new_vars): 
        """ 
            Updates the variables. The list is expected to have form:
                [weights, biases]
        """
        self.W = new_vars[0]

        if not isinstance(self.b, type(None)): 
            self.b = new_vars[1]

if __name__ == '__main__':
    print("This script is not mean to be run.")
