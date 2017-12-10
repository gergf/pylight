# pyLight-_alpha_

pyLight is a python3 library designed to ease the understanding of Neural Networks. It is light because it is simple, without additives, and because we hope that our work helps you to see the light. 

pyLight tries to be as faithful as possible to a real implementation, but without adding the needed optimizations to be an efficient tool to use neural networs in a real-world environment. Therefore, we priorize being education rather than efficient. 

## Supported Operations

### Layers

- Dense Layer. Note that the input_shape it is only required in the first layer.
	`{
 		"type" : "dense", 
 		"W" : weights,
 		"b" : biases, 
 		"input_shape" : input_shape,
 		"name" : "name"
 	}`
 	Note: If you don't want to use biases, set them to None. 
   

### Losses 

- Softmax Cross Entropy Loss
	`{
		"type" : "softmax_cross_entropy", 
		"name" : "name"
	}`

### Nonlinearities 
- Also known as activation functions 
	
	`{"type" : "sigmoid", "name" : "name"}`

	`{"type" : "relu", "name" : "name"}`

	`{"type" : "softmax", "name" : "name"}`

	`{"type" : "linear", "name" : "name"}`

## Contributions
We are open to any kind of contributions. The contributions are classified as follow:
1. **New operations contributions.** The only rule is that you only can use numpy to implement the maths required by your operation. 
2. **Change already implemented code**. The new code **must** be easier to understand than the old one, not more efficient nor more "computationally" correct.
3. **API changes**. Any change which helps the API to be cleaner is welcomed. 

## TODO List 

- Ensure numerical stability 
- Gradient checking 
- Momentum optimizer 
- Dropout 
- Batch Normalization 
- ...

## LICENSE 

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
