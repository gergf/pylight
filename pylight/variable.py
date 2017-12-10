# STILL NOT USED

class Variable():
	def __init__(self, value, operation, parents, childrens, trainable=True, 
		         name="tensor:0"): 
		"""
			Creates a Variable. 

			Note: A Variable can only belong to a unique computational graph. 

			Parameters
			----------
				value : A Numpy array with the value of the tensor
				operation : The operation which have created this Variable
				parents : A list with the variables that are parents of this Variable
				childrens : A list with the variables that are childrens of this Variable
				trainable : If True, the variable will be updated by the optimizer
				name : String defining the name of the Variable
		
		"""

		self.value = value 
		self.operation = operation
		self.parents = parents
		self.childrens = childrens
		self.trainable = True 
		self.name = name 
