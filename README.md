# MLP_fromscratch


The global gradient of any layer is given by the dot product of the next global gradient with the next layer. The he result is multiplied element-wise by the derivative of the next activation function, as we can see on the formula bellow:
## $\delta^{(l)} = (\delta^{(l+1)} \cdot \theta^{(l+1)}) \odot \sigma^{'(l+1)}$
## $\frac{\partial J}{\partial \theta^{(l)}} = \delta^{(l)^{T}} a^{(l-1)}$
## $\delta^{(L)} = \frac{\partial J}{\partial a^{(L)}} \frac{\partial a^{(L)}}{\partial z^{(L)}}$
