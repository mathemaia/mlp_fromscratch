import numpy as np
import json
import matplotlib.pyplot as plt
import pickle


class Layer:
    def __init__(self, units: int, activation: str = None, input_shape: tuple = None, initialization: str = 'normal') -> None:
        self.units = units
        self.activation = activation
        self.initialization = initialization
        
        # é uma camada de entrada
        if input_shape != None:
            self.shape = (units, input_shape[1])
               
        # não é uma camada de entrada
        else:
            self.shape = (units,)
           
            

class MLP:
    def __init__(self, seed: int = 42, lr: float = 0.01) -> None:
        '''Classe que instancia uma rede neural do tipo Multi-Layer Perceptron.
        Args:
            seed (int): Semente de números aleatórios.
            lr (float): Valor da taxa de aprendizado.
        '''
        
        # define uma seed
        np.random.seed(seed)
        
        # lista de camadas e bias
        self.layers = []
        self.bias = []

        # dicionário com informações sobre as camadas
        self.archtecture = []
        
        # possíveis funções de erro
        self.losses = ['mse', 'binary_crossentropy', 'categorical_crossentropy']
        self.loss = ''
        
        # possíveis otimizadores
        self.optimizers = ['sgd', 'adam']
        self.optimizer = ''
        
        # lista de ativações - usada no backpropagation
        self.activations = []
        
        # parâmetros do Adam
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.m = {'theta': {}, 'bias': {}}
        self.v = {'theta': {}, 'bias': {}}
        self.t = 0
          
        # taxa de aprendizado
        self.lr = lr
        

    def add(self, layer: Layer) -> None:
        '''Cria um dicionário com informações das camadas. Estas informações serão usadas na
        função de distribuir os pesos, onde as matrizes das camadas serão criadas.
        
        Args:
            layer: Objeto da classe Layer
        '''
        
        outputs = layer.shape[0]
        
        # se é uma camada de entrada, usa as dimensões de 'input_shape'
        if len(layer.shape) > 1:
            inputs = layer.shape[1]

        # se não, pega a saída da última camada
        else:
            inputs = self.archtecture[-1]['shape'][0]
        
        self.archtecture.append({
            'shape': (outputs, inputs),
            'activation': layer.activation,
            'initialization': layer.initialization
        })

    
    def _forward(self, X: np.ndarray):
        '''Função interna que calcula a saída da rede para uma determinada entrada X.
        Args:
            X (np.array): Matriz de entrada.
        '''

        self.activations = []
        for i, layer in enumerate(self.layers):
            # pega a função de ativação da camada atual
            activation_function = self._activations_functions(name=self.archtecture[i]['activation'])
            
            # se for a camada de entrada, multiplica pelo X
            if i == 0:
                z = np.dot(X, layer.T) + self.bias[i].T

            # se não, multiplica pela ultima ativação
            else:
                z = np.dot(self.activations[-1], layer.T) + self.bias[i].T
                
            # ativa z
            a = activation_function(z)
            self.activations.append(a)
        
        return self.activations[-1]
 
    
    def _loss_functions(self, name: str):
        '''Devolve uma função de custo para calcular o erro do forward.
        Args:
            name (str): Nome da função de custo.
        '''
        
        def mse(y_true, y_pred):
            '''Função de custo Mean Squared Error'''

            return 0.5 * np.mean((y_true - y_pred)**2)

        def binary_crossentropy(y_true, y_pred):
            '''Função de custo Binary Cross-Entropy'''
            epsilon=1e-5
            y_pred_ = np.clip(y_pred, epsilon, 1 - epsilon)
            return -np.mean(y_true * np.log(y_pred_) + ((1 - y_true) * np.log(1 - y_pred_))) 
        
        def categorical_crossentropy(y_true, y_pred):
            '''Função de custo Categorical Cross-Entropy'''
            epsilon = 1e-12
            y_pred = np.clip(y_pred, epsilon, 1. - epsilon)  
            loss = -np.sum(y_true * np.log(y_pred), axis=1)
            
            return np.mean(loss)


        functions = {
            'mse': mse,
            'binary_crossentropy': binary_crossentropy,
            'categorical_crossentropy': categorical_crossentropy
        }
        
        return functions[name] 


    def _params_init(self) -> None:
        '''Faz a inicialização dos parâmetros da rede. A distribuição de valores é feita de acordo
        com a ativação.'''
        
        # inicializa os parametros
        theta = []
        bias = []
        for layer in self.archtecture:
            # inicialização He
            if layer['activation'] in ('relu', 'leaky_relu'):
                n_in = layer['shape'][1]
                
                # normal
                if layer['initialization'] == 'normal':
                    weights = np.random.normal(loc=0, scale=np.sqrt(2/n_in), size=layer['shape'])
                    
                # uniforme
                elif layer['initialization'] == 'uniform':
                    weights = np.random.uniform(low=-np.sqrt(6 / n_in), high=np.sqrt(6 / n_in), size=layer['shape'])
                theta.append(weights)
            
            # inicialização Xavier
            elif layer['activation'] in ('sigmoid', 'tanh', 'linear', 'softmax'):
                n_in = layer['shape'][1]
                n_out = layer['shape'][0]
                
                # normal
                if layer['initialization'] == 'normal':
                    weights = np.random.normal(loc=0, scale=np.sqrt(2/(n_in+n_out)), size=layer['shape'])
                    
                # uniforme
                elif layer['initialization'] == 'uniform':
                    weights = np.random.uniform(low=-np.sqrt(6 / (n_in + n_out)), high=np.sqrt(6 / (n_in + n_out)), size=layer['shape'])

                theta.append(weights)
        
        # inicialização do bias
        for layer in theta:
            bias.append(np.zeros((layer.shape[0], 1)))
        
        self.layers = theta
        self.bias = bias

    
    def _activations_functions(self, name: str):
        def sigmoid(z):
            z_ = np.clip(z, -500, 500)
            return 1 / (1 + np.exp(-z_))
        
        def tanh(z):
            return np.tanh(z)
        
        def leaky_relu(z, alpha=0.01):
            return np.where(z>0, z, alpha*z)

        def linear(z):
            return z

        def relu(z):
            return np.maximum(0, z)
        
        def softmax(z):
            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)

        functions = {
            'sigmoid': sigmoid,
            'tanh': tanh,
            'leaky_relu': leaky_relu,
            'linear': linear,
            'relu': relu,
            'softmax': softmax
        }

        return functions[name]
        

    def _activations_derivatives(self, a: np.ndarray, name: str):
        '''Obtém a derivada de uma função de ativação.
        Args:
            a (np.ndarray): Resultado da camada ativada.
            name (str): Nome da função de ativação.
        '''
        
        # Sigmoid
        if name == 'sigmoid':
            sigmoid = self._activations_functions('sigmoid')
            return sigmoid(a) * (1 - sigmoid(a))
        
        # ReLu
        elif name == 'relu':
            return np.where(a > 0, 1, 0)
        
        # Tanh
        elif name == 'tanh':
            tanh = self._activations_functions('tanh')
            return 1 - tanh(a)**2
        
        # Leaky ReLu
        elif name == 'leaky_relu':
            alpha = 0.01
            return np.where(a > 0, 1, alpha)
        
        # Linear
        elif name == 'linear':
            return np.ones_like(a)
        
        # Softmax
        elif name == 'softmax':
            jacobians = []
            for i in range(a.shape[0]):
                y = a[i].reshape(-1, 1)  # Reshape para vetor coluna
                jacobian = np.diagflat(y) - np.dot(y, y.T)
                jacobians.append(jacobian)
            return np.array(jacobians)


    def _loss_derivatives(self, name: str, y_pred: np.ndarray, y_true: np.ndarray):
        '''Obtém as derivadas das funções de erro.
        Args:
            name (str): Nome da função de erro.
            y_pred (np.ndarray): Predição da rede.
            y_true (np.ndarray): Valores de saída verdadeiros.
        '''
        
        # Binary Cross Entropy
        if name == 'binary_crossentropy':
            return -(y_true / y_pred) + ((1 - y_true) / (1 - y_pred))
        elif name == 'categorical_crossentropy':
            return y_pred - y_true
    

    def _backprop(self, X, y):
        '''Do the Backpropagation algorithm.'''
        
        # reverse the lists
        layers = self.layers[::-1]
        archtecture = self.archtecture[::-1]
        activations = self.activations[::-1]
        bias = self.bias[::-1]
        
        # increment the Adam time counter
        self.t += 1

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

                # calculation of delta
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

            ### Adam
            if self.optimizer == 'adam':
                # initialize moving averages and variance if necessary
                if l not in self.m['theta']:
                    self.m['theta'][l] = np.zeros_like(delta)
                    self.v['theta'][l] = np.zeros_like(delta)
                    self.m['bias'][l] = np.zeros_like(bias[l])
                    self.v['bias'][l] = np.zeros_like(bias[l])
                
                # update m and v for theta
                self.m['theta'][l] = self.beta1 * self.m['theta'][l] + (1 - self.beta1) * delta
                self.v['theta'][l] = self.beta2 * self.v['theta'][l] + (1 - self.beta2) * (delta ** 2)
                
                # bias correction for m and v
                m_hat_theta = self.m['theta'][l] / (1 - self.beta1 ** self.t)
                v_hat_theta = self.v['theta'][l] / (1 - self.beta2 ** self.t)

                # update weights theta
                layers[l] = layers[l] - self.lr * m_hat_theta / (np.sqrt(v_hat_theta) + self.epsilon)
                
                # gradient used for the bias
                grad_bias = glob.T.sum(axis=1, keepdims=True)
                
                # update m and v for bias
                self.m['bias'][l] = self.beta1 * self.m['bias'][l] + (1 - self.beta1) * grad_bias
                self.v['bias'][l] = self.beta2 * self.v['bias'][l] + (1 - self.beta2) * (grad_bias ** 2)
                
                # bias correction
                m_hat_bias = self.m['bias'][l] / (1 - self.beta1 ** self.t)
                v_hat_bias = self.v['bias'][l] / (1 - self.beta2 ** self.t)

                # update the bias
                bias[l] = bias[l] - self.lr * m_hat_bias / (np.sqrt(v_hat_bias) + self.epsilon)
            
            ### SGD
            elif self.optimizer == 'sgd':
                layers[l] = layers[l] - (self.lr * delta)
                bias[l] = bias[l] - (self.lr * glob.T.sum(axis=1, keepdims=True))

        # reverse the lists to the original order
        self.layers = layers[::-1]
        self.bias = bias[::-1]
        
        
    def evaluate(self, y_true, y_pred):
        if self.archtecture[-1]['activation'] == 'sigmoid':
            return np.sum(y_true == y_pred) / len(y_true)
        else:
            return np.sum(y_true.argmax(axis=1) == y_pred) / len(y_true)


    def fit(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int, lr: float = 0.01):
        '''Faz o treinamento'''
        
        # lança um erro caso a quantidade de neuronios de saída da rede seja diferente da quantidade de classes
        output_units =self.archtecture[-1]['shape'][0]
        n_classes = y_train.shape[1]
        if output_units != n_classes:
            raise Exception(f"Output units {output_units} doesn't match with number of classes {n_classes}.")
        
        # define função de custo e taxa de aprendizado
        J = self._loss_functions(self.loss)
        self.lr = lr

        # loop de treinamento
        for epoch in range(epochs):
            # faz a predição
            pred = self._forward(X_train)
            
            # calcula o erro
            loss = J(y_true=y_train, y_pred=pred)
            
            # faz a otimização dos parâmetros
            self._backprop(X=X_train, y=y_train)
            
            # infos
            acc = self.evaluate(y_true=y_train, y_pred=self.predict(X_train))
            print(f"epoch: {epoch:<3}  |  loss: {loss:<20}  |  acc: {acc:5}")
            

    def predict(self, X, probs: bool = False):
        '''Função utilizada pelo usuário para fazer predição - chama a função interna _forward()
        Args: 
            X (np.array): Matriz de entrada.
        '''

        # 
        if not probs:
            if len(X) > 1:
                # caso a última ativação ser sigmoid, usa threshold de 50% 
                if self.archtecture[-1]['activation'] == 'sigmoid':
                    pred = self._forward(X)
                    return np.where(pred >= 0.5, 1, 0)
                
                # se não, calcula argmax em batch
                else:
                    return self._forward(X).argmax(axis=1)
            else:
                return self._forward(X).argmax()
        else:
            return self._forward(X)


    def compile(self, loss: str, optimizer: str = 'sgd') -> None:
            '''
            Prepara o modelo para o treinamento. Esta função deve ser chamada quando o usuário
            já terminou de construir a arquitetura, pois ela chamará todas as funções essencias
            para criar a rede neural. Além disso, lançará um erro caso a função de ativa
            
            Args:
                optimizer (str - ['sgd', 'adam']): Otimizador que ajustará os parametros.
                loss (str -['bce', 'mse']): Função que calcula o erro de saída.
                params_init (str -['he', 'xavier', 'normal', 'uniform']): Tipo de inicialização de pesos.
            '''
            
            # checa a validade do parametro 'optimizer'
            if optimizer not in ('sgd', 'adam'):
                raise Exception('Parameter "optimizer" not recognized. Please, choose a valid option.')
            
            # checa a validade da ultima camada
            last_layer = self.archtecture[-1]
            last_n_out = last_layer['shape'][0]
            last_activation = last_layer['activation']
            if last_activation in ('sigmoid', 'linear'):
                if last_n_out != 1:
                    raise Exception(f'Invalid output {last_n_out} for last layer with "{last_activation}" activation.')
            elif last_activation == 'softmax' and last_layer['shape'][0] < 2:
                raise Exception(f'Invalid output {last_n_out} for last layer with "{last_activation}" activation.')
                
            # lança um erro caso a ativação da última camada não seja softmax, sigmoid ou linear
            if last_layer['activation'] not in ('softmax', 'sigmoid', 'linear'):
                wrong_activation = last_layer['activation']
                raise Exception(f'Activation function "{wrong_activation}" is not applied on last layer.')
            else:
                correct_losses = {
                'softmax': 'categorical_crossentropy',
                'sigmoid': 'binary_crossentropy',
                'linear': 'mse'
                }
                
                # lança um erro caso a ativação da ultima camada e a função de erro não combinem
                activation = self.archtecture[-1]['activation']
                correct = correct_losses[activation]
                if loss != correct:
                    raise Exception(f'Activation function "{activation}" does not match with "{loss}" loss function.')
            
            # inicializa os parametros
            self._params_init()
            
            # define a função de custo e o otimizador
            self.optimizer = optimizer
            self.loss = loss
    

    def save(self, filename: str):
        '''Salva todos os atributos do objeto em um arquivo usando pickle.'''
        
        with open(f'{filename}.pkl', 'wb') as file:
            pickle.dump(self, file)


    @classmethod
    def load(cls, filename: str):
        '''Carrega um objeto a partir de um arquivo usando pickle.'''
        
        with open(filename, 'rb') as file:
            return pickle.load(file)