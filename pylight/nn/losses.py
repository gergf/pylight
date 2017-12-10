import numpy 
from .nonlinearities import softmax

class cost_function():
    def __init__(self, name):
        self.name = name
        self.trainable = False

    def forward(self, X): 
        raise NotImplementedError("Please Implement the Forward method") 

    def backward(self): 
        raise NotImplementedError("Please Implement the Backward method") 

class softmax_cross_entropy(cost_function):
    def __init__(self, name="softmax_cross_entropy"):
        super().__init__(name)

    def forward(self, one_hot_labels, logits):
        """ 
            Given the logits and the labels, returns the cross entropy for each case.

            Parameters
            ----------
                one_hot_labels : A numpy array with the true labels in one hot format. Shape (batch_size, classes)
                logits : A numpy array with the logits. Shape (batch_size, classes)

            Returns 
            -------
                A numpy array with shape (batch_size) whith the cross entropy for each sample.

        """
        assert logits.shape == one_hot_labels.shape, "Error! Logits must have the same shape than labels." 

        self.logits = logits.astype(numpy.float32)
        self.labels = one_hot_labels.astype(numpy.float32) 

        # To ensure numerical stability we undo the exp of the softmax with the log of the cross entropy 
        # so we have: - sum_j label_j * (logits_j - log sum_k exp(logits_k))
        # We compute the scaled logits to avoid a huge exponential value 
        scaled_logits = self.logits - numpy.amax(self.logits, axis=-1, keepdims=True) 

        norm_factor   = numpy.log(numpy.sum(numpy.exp(scaled_logits), axis=-1, keepdims=True))
        self.errors   = - numpy.sum(self.labels * (scaled_logits - norm_factor), axis=-1)

        return self.errors.mean()

    def backward(self):
        """ Retunrs the gradient of the inputs using the softmax_cross_entropy as loss function. """
        # Algebra explained here: [https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function]
        self.probs = softmax(input_shape = self.logits.shape).forward(self.logits) 
        self.dL_logits = self.probs - self.labels
        return self.dL_logits

if __name__ == "__main__": 
    print("This script is not mean to be run.")