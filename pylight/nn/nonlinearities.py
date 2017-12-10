import numpy 

class nonlinearities():
    def __init__(self, input_shape, name):
        # Stored Input and Output values
        self.X = None 
        self.Z = None

        self.input_shape  = input_shape
        self.output_shape = input_shape
        self.trainable = False 
        self.name = name
        
        # Local gradients
        self.dZ_X = None 

    def forward(self, X): 
        raise NotImplementedError("Please Implement the Forward method") 

    def backward(self, dL_Z): 
        raise NotImplementedError("Please Implement the Backward method") 

class softmax(nonlinearities): 
    def __init__(self, input_shape, name="softmax"):
        super().__init__(input_shape, name)

    def forward(self, X): 
        """ Computes the Softmax function of a matrix X with shape (batch, features) """ 
        self.X = X.astype(numpy.float32)
        exps = numpy.exp(X - numpy.max(X, axis=1, keepdims=True))
        self.Z = exps / exps.sum(axis=1, keepdims=True)
        return self.Z

    def backward(self, dL_Z): 
        """ Given dL/dZ returs dL/dX """
        # TODO: Check that de backward pass of the softmax is computed properly... 
        assert self.Z.shape == dL_Z.shape, "The gradient wrt Z do not have the same shape than Z"
        self.dL_Z = dL_Z.astype(numpy.float32)
        self.dL_X = numpy.empty(shape=self.X.shape)
        
        # Local gradient 
        num_samples  = self.X.shape[0]
        num_features = self.X.shape[1]
        # Because each element of X affects to all Z, we have a Jacobian
        Jacobian     = numpy.empty(shape=(num_features, num_features))
        self.dZ_X    = numpy.empty(shape=(num_samples, num_features, num_features))
        for s in range(num_samples):
            for i in range(num_features):
                for j in range(num_features):
                    kronecker_delta = 1 if i == j else 0  
                    Jacobian[i, j] = self.Z[s, i] * (kronecker_delta - self.Z[s, j])

            self.dZ_X[s] = Jacobian
            self.dL_X[s] = self.dZ_X[s] @ self.dL_Z[s]

        return self.dL_X

class relu(nonlinearities): 
    def __init__(self, input_shape, name="relu"):
        super().__init__(input_shape, name)

    def forward(self, X): 
        """ Computes the ReLU function of a matrix X with shape (batch, features) """
        self.X = X.astype(numpy.float32)
        self.Z = numpy.maximum(self.X, 0)
        return self.Z

    def backward(self, dL_Z): 
        """ Given dL/dZ returs dL/dX """
        assert self.Z.shape == dL_Z.shape, "The gradient wrt Z do not have the same shape than Z"
        self.dL_Z = dL_Z.astype(numpy.float32)

        # Local gradient 
        self.dZ_X = self.X > 0

        # wrt L 
        self.dL_X = self.dL_Z * self.dZ_X # Element wise 

        return self.dL_X

class sigmoid(nonlinearities):
    def __init__(self, input_shape, name="sigmoid"):
        super().__init__(input_shape, name)

    def forward(self, X):
        """ Computes the Sigmoid function of a matrix X with shape (batch, features) """ 
        self.X = X.astype(numpy.float32)
        self.Z = numpy.empty(shape=self.X.shape) 
        
        num_samples = self.X.shape[0]
        for n in range(num_samples):
            self.Z[n] = 1 / (1 + numpy.exp(-self.X[n]))

        return self.Z 

    def backward(self, dL_Z):
        """ Given dL/dZ returs dL/dX """
        assert dL_Z.shape == self.Z.shape, "The gradient wrt Z do not have the same shape than Z"
        
        self.dL_Z = dL_Z.astype(numpy.float32)
        self.dZ_X = numpy.zeros(shape=self.X.shape)

        num_samples = self.X.shape[0]
        for n in range(num_samples):
            self.dZ_X[n] = (1 - self.Z[n]) * self.Z[n]

        self.dL_X = self.dL_Z * self.dZ_X

        return self.dL_X


class linear(nonlinearities):
    def __init__(self, input_shape, name="linear"):
        super().__init__(input_shape, name)

    def forward(self, X): 
        """ Computes the Linear function (with scope 1) of a matrix X with shape (batch, features).  """ 
        self.X = X.astype(numpy.float32)
        self.Z = self.X 
        return self.Z 

    def backward(self, dL_Z): 
        """ Given dL/dZ returs dL/dX """
        assert self.Z.shape == dL_Z.shape, "The gradient wrt Z do not have the same shape than Z"

        self.dL_Z = dL_Z.astype(numpy.float32)

        # Local gradient 
        self.dZ_X = numpy.ones(self.X.shape)

        # wrt L 
        self.dL_X = self.dL_Z * self.dZ_X

        return self.dL_X

if __name__ == "__main__": 
    print("This script is not mean to be run.")
