from . import optimizers
from .nn import layers 
from .nn import losses 
from .nn import nonlinearities

class neural_network():
	"""
		This class encapsulates the computational graph of a neural network. 

		The graph has four main methods: 
			- _forward(inputs, labels) : Computes the forward from the inputs until the loss lass operations
			- _backprop() : Computes the backward for all the variables in the graph
			- update_parameters(inputs, labels) : Performs an iteration of the training process (forwards and backward)
			- predict(inputs) : Performs a prediction of the given inputs 

			Parameters
			----------
				operations : A list with the definition of each operations of the net. To see all the avaiable 
				operations check "supported operations" at the README.
	"""

	def __init__(self, operations, optimizer):
		assert len(operations) >= 2, "Error! At least one input layer and one loss layer are required."
		assert type(operations[0]["input_shape"]) == tuple, "Error! The first layer must define the input shape."
		assert issubclass(type(optimizer), optimizers.optimizer), "Error! Optimizer must be a subclass of pylight.optimizers"

		self.optimizer = optimizer
		self.graph = self._build_graph(operations)

		# For now, we do not need to set the batch size 
		self.input_shape = ("?", self.graph[0].input_shape[0])

	def _build_graph(self, operations): 
		""" Returns the computational graph defined by the operations """
		graph = list()

		# Build the computational graph 
		for operation in operations: 
			op_type = operation["type"]
			op_name = operation["name"]
			op_input_shape = operation["input_shape"] if len(graph) == 0 else graph[-1].output_shape

			# Layers 
			if   op_type == "dense": 
				graph_op = layers.dense(
								  operation["W"],
								  operation["b"],
								  op_input_shape, 
								  op_name)

			# Nonlinearities 
			if   op_type == "sigmoid":
				graph_op = nonlinearities.sigmoid(op_input_shape, op_name)
			elif op_type == "relu":
				graph_op = nonlinearities.relu(op_input_shape, op_name)
			elif op_type == "softmax":
				graph_op = nonlinearities.softmax(op_input_shape, op_name)
			elif op_type == "linear":
				graph_op = nonlinearities.linear(op_input_shape, op_name)

			# Losses
			if   op_type == "softmax_cross_entropy":
				graph_op = losses.softmax_cross_entropy(op_name)   
			
			graph.append(graph_op)

		# Finally, check that the last layer is a loss function layer 
		if not issubclass(type(graph[-1]), losses.cost_function):
			raise ValueError("The last operation of the graph must be a loss operation.")

		return graph

	def _forward(self, X, Y):
		""" Compute all the operations from the input to the loss """
		for operation in self.graph[:-1]:
			X = operation.forward(X)

		# Return value of the cost function  
		return self.graph[-1].forward(Y, X) 

	def _backprop(self): 
		""" Computes all the gradients from the loss to the input """
		gradient = self.graph[-1].backward() # Gradient of the loss (1) 
		for operation in reversed(self.graph[:-1]):
			# Remember that each operation MUST return ONLY the gradient wrt its inputs. 
			# The gradient wrt its W is stored in each operation. 
			# Furthermore, we limit the graph to be a sequential graph.
			gradient = operation.backward(gradient)

	def train(self, X, Y): 
		""" Updates the parameters following the rule defined by the optimizer """
		self._forward(X, Y)
		self._backprop()

		# Update parameters of all the trainable operations 
		for operation in self.graph[:-1]:
			if operation.trainable: 
				# Returns a list with tuples (var, grad) 
				var_and_grads = operation.get_trainable_vars_and_grads()
				
				# Update the variables 
				new_vars = list()
				for var, grad in var_and_grads: 
					new_vars.append(self.optimizer.update_variable(var, grad))
				
				# Store the new variables
				# The vars are expected to have the same order that 
				operation.set_trainable_vars(new_vars)

	def predict(self, X): 
		""" Returns the class probabilities for the input samples """ 
		# Compute all the operations in the graph except the cost funtion 
		for operation in self.graph[:-1]:
			X = operation.forward(X)

		return X 
