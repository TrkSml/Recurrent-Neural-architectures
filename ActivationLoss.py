import numpy as np

#Activation functions
class sigmoid:
	def __call__(self,x):
		return 1/(1+np.exp(-x))

	def prime(self,x):
		p=self.__call__(x)
		return p*(1-p)

class ReLU:
	def __call__(self,X):
		if X.ndim<=1:
			return array(map(lambda x:0 if x<0 else x,X))
		else:
			return array([map(lambda x:0 if x<0 else x,el) for el in X])

class softmax():
    def __call__(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def prime(self, x):
        p = self.__call__(x)
        return p * (1 - p)

class tanh_func():
    def __call__(self, x):
        return np.tanh(x)

    def prime(self, x):
        return 1 - np.power(self.__call__(x),2)


#Loss functions
#Cross Entropy class
class lossCE:
    def __init__(self): 
    	self.clip_minimum=.0001
    	self.clip_maximum=.9999

    def loss(self, y, p):
        p = np.clip(p, self.clip_minimum, self.clip_maximum)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def derivative(self, y, p):
        p = np.clip(p, self.clip_minimum, self.clip_maximum)
        return -(y/p)+ (1-y)/(1-p)


