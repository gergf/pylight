import numpy 

class optimizer(): 
	def __init__(self, name):
		self.name = name

class GradientDescent(optimizer):
	def __init__(self, learning_rate, name="GradientDescent"):
		super().__init__(name)
		self.learning_rate = numpy.float32(learning_rate)

	def update_variable(self, variable, gradient):
		assert variable.shape == gradient.shape, "Error! The variable and its gradient must have the same shape."
		return variable - self.learning_rate * gradient
