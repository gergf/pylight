import numpy 

class initializer():
	def __init__(self, name): 
		self.name = name 

class uniform(initializer):
	""" 
		Initializer using a uniform distribution : Unif[a, b), b > a  

		Parameters
		----------
			a : Lower bound of the distribution (inclusive)
			b : Upper bound of the distribution (exclusive) 
	"""
	def __init__(self, a=-1, b=1, name="random_initializer"):
		assert b > a, "Error! b must be bigger than a."

		super().__init__(name)
		
		self.a = a
		self.b = b 

	def get_variable(self, shape): 
		return (self.b - self.a) * numpy.random.random_sample(size=shape) + self.a