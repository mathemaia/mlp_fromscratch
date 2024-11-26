# mlp_fromscratch

## Backpropagation
Backpropagation is the mainly algorithm for the learning process in a Neural Network. In resume, it calculates the parameters’s gradient of all layers in relation to the Loss Function applying the chain rule of calculus. We can interpratate it as “how the loss will behave if a minimum value is added in this parameter?”. The answer for this question is that if the gradient is positive, add minimum value will make the loss increase, othewise, if it is negative, the loss will decrease. So, after obtaining the gradients, we can update the parameters using these values in the Optimizer Function. The math is given by the formulas bellow:

$$\delta^{(l)} = (\delta^{(l+1)} \cdot \theta^{(l+1)}) \odot{a^{'(l)}}$$

The global gradient of any layer is computed as the dot product of the global gradient and the weight matrix of the next layer. The result is then multiplied element-wise by the derivative of the activation function of the current layer. So, with the global gradient obtained, we transpose and use it in a dot product with the activation of the previous layer. This gives us the gradient of the weights for any layer, which can then be used in the Optimizer Function to update the parameters.

$$\nabla{J_{\theta^{(l)}}} = \delta^{(l)^{T}} \cdot{a^{(l-1)}}$$

Well, but how is the first global gradient calculated? It is determined as the Hadamard product (element-wise multiplication) of the derivative of the Loss Function with respect to the activation of the last layer and the derivative of the activation function of the last layer with respect to its pre-activation (the input to the activation function).

$$\delta^{(L)} = \frac{\partial J}{\partial a^{(L)}} \odot \frac{\partial a^{(L)}}{\partial z^{(L)}}$$

The core of the code is:

```python
    def _backprop(self, X, y):
        '''Do the Backpropagation algorithm.'''
        
        # reverse of the lists
        layers = self.layers[::-1]
        archtecture = self.archtecture[::-1]
        activations = self.activations[::-1]
        bias = self.bias[::-1]

        for l in range(len(layers)):
            ### if it's the last layer
            if l == 0:
                J_a = self._loss_derivatives(name=self.loss, y_pred=activations[l], y_true=y)
                a_z = self._activations_derivatives(a=activations[l], name=archtecture[l]['activation'])
                z_theta = activations[l+1]

                # global gradient calculation is different for Softmax
                if self.loss == 'categorical_crossentropy' and archtecture[l]['activation'] == 'softmax':
                    glob = np.einsum('ijk,ik->ij', a_z, J_a)
                else:
                    glob = J_a * a_z

                # delta calculation
                delta = np.dot(glob.T, z_theta)
            
            ### if the current layer is a hidden layer
            elif l > 0 and l < len(activations) - 1:
                z_a = layers[l-1]
                a_z = self._activations_derivatives(a=activations[l], name=archtecture[l]['activation'])
                z_theta = activations[l+1]
                glob = np.dot(glob, z_a) * a_z
                delta = np.dot(glob.T, z_theta)
            
            ### if the current layer is the input layer
            else:
                z_a = layers[l-1]
                a_z = self._activations_derivatives(a=activations[l], name=archtecture[l]['activation'])
                glob = np.dot(glob, z_a) * a_z
                delta = np.dot(glob.T, X)
```
