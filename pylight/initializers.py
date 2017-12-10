import numpy 

class initializer():
	def __init__(self, dtype, name): 
		self.dtype = dtype 
		self.name = name 

class uniform(initializer):
	""" 
		Initializer using a uniform distribution : Unif[a, b), b > a  

		Parameters
		----------
			a : Lower bound of the distribution (inclusive). Default -1. 
			b : Upper bound of the distribution (exclusive). Default 1.
			dtype : Data type. Default float32. 
			name : Name of the initializer.

		Returns 
		-------
			A numpy array with dtype float32 and the initialized random values. 

	"""
	def __init__(self, a=-1, b=1, dtype=numpy.float32, name="random_initializer"):
		assert b > a, "Error! b must be bigger than a."

		super().__init__(dtype, name)
		
		self.a = a
		self.b = b 

	def get_variable(self, shape):
		# If shape is 1, then random_sample returns an scalar. We cant an array with only one element. 
		# Then cast the array dtype to the defined by the user. 
		var = numpy.asarray(numpy.random.random_sample(size=shape)).astype(self.dtype)
		return (self.b - self.a) * var + self.a

class constant(initializer): 
	"""
		Initializes the variables to a constant value. 

		Parameters 
		----------
			cte : Constant value. Default 1.0.
			dtype : Data type. Default float32. 
			name : Name of the initializer.
		
		Returns 
		-------
			A numpy array with initialized values. 

	"""
	def __init__(self, cte=1, dtype=numpy.float32, name="constant_initializer"): 
		super().__init__(dtype, name)
		self.cte = cte 

	def get_variable(self, shape): 
		return numpy.ones(shape=shape, dtype=self.dtype) * self.cte 