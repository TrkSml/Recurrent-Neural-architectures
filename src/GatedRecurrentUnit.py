# __author__ = Tarek Samaali

import numpy as np
from Optimizers import SGD
from ActivationLoss import lossCE, sigmoid, tanh_func, softmax

def array(x):
	return x if type(x).__module__ == np.__name__ else np.array(x)


def to_categorical(x, n_col=None):
    """ One-hot encoding of nominal values """
    """ Written by Erik Linder-Noren"""
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot


def gen_mult_ser(nums,timesteps):
        """ Method which generates multiplication series """
        """ Written by Erik Linder-Noren """
        X = np.zeros([nums, timesteps, 61], dtype=float)
        y = np.zeros([nums, timesteps, 61], dtype=float)
        for i in range(nums):
            start = np.random.randint(2, 7)
            mult_ser = np.linspace(start, start*10, num=timesteps, dtype=int)
            X[i] = to_categorical(mult_ser, n_col=61)
            y[i] = np.roll(X[i], -1, axis=0)
        return X, y


def train_test_split(X, y, split_size=0.3):
    # Split the training data from test data in the ratio specified in
    # test_size
    length=int(len(X)*split_size)
    X_train, X_test = X[:length], X[length:]
    y_train, y_test = y[:length], y[length:]

    return X_train, X_test, y_train, y_test


lossCE=lossCE()
SGD=SGD()
softmax=softmax()
tanh=tanh_func()
sigmoid=sigmoid()

class GRU:

	def __init__(self,size_of_hidden_layers=100,epochs=100,truncation=10):

		self.epochs=epochs
		self.size_of_hidden_layers=size_of_hidden_layers
		self.truncation=truncation

	def forward_and_backwardpropagation(self,X_train,y_train):
		batch_size, timesteps, input_dim = X_train.shape
		size_of_hidden_layers=self.size_of_hidden_layers
		X=X_train

		## Initializing the wight matrices according to some external material

		self.Uz=np.random.uniform(-1./np.sqrt(input_dim),1./np.sqrt(input_dim),((size_of_hidden_layers, input_dim)))
		self.Ur=np.random.uniform(-1./np.sqrt(input_dim),1./np.sqrt(input_dim),((size_of_hidden_layers, input_dim)))
		self.Uh=np.random.uniform(-1./np.sqrt(input_dim),1./np.sqrt(input_dim),((size_of_hidden_layers, input_dim)))

		self.Wz=np.random.uniform(-1./np.sqrt(size_of_hidden_layers),1./np.sqrt(size_of_hidden_layers),(size_of_hidden_layers, size_of_hidden_layers))
		self.Wr=np.random.uniform(-1./np.sqrt(size_of_hidden_layers),1./np.sqrt(size_of_hidden_layers),(size_of_hidden_layers, size_of_hidden_layers))
		self.Wh=np.random.uniform(-1./np.sqrt(size_of_hidden_layers),1./np.sqrt(size_of_hidden_layers),(size_of_hidden_layers, size_of_hidden_layers))

		self.Wo=np.random.uniform(-1./np.sqrt(size_of_hidden_layers),1./np.sqrt(size_of_hidden_layers),(size_of_hidden_layers, input_dim))

		Uz=self.Uz
		Ur=self.Ur
		Uh=self.Uh

		Wz=self.Wz
		Wr=self.Wr
		Wh=self.Wh

		Wo=self.Wo

		crawler=0

		# Beginning of the forward propagation for a certain number of epochs

		for crawler in range(self.epochs):

			r=np.zeros((batch_size, timesteps, size_of_hidden_layers))
			sor=np.zeros((batch_size, timesteps, size_of_hidden_layers))
			z=np.zeros((batch_size, timesteps, size_of_hidden_layers))
			q=np.zeros((batch_size, timesteps, size_of_hidden_layers))
			ht=np.zeros((batch_size, timesteps, size_of_hidden_layers))
			h=np.zeros((batch_size, timesteps+1, size_of_hidden_layers))
			output=np.zeros((batch_size, timesteps, input_dim))
				
			for t in range(timesteps):

				## Applying the GRU equations

				r[:,t]=sigmoid(X[:,t].dot(Ur.T)+h[:,t-1].dot(Wr))
				z[:,t]=sigmoid(X[:,t].dot(Uz.T)+h[:,t-1].dot(Wz))
				sor[:,t]=h[:,t-1]*r[:,t]
				q[:,t]=X[:,t].dot(Uh.T)+sor[:,t].dot(Wh)
				ht[:,t]=tanh(q[:,t])
				h[:,t]=(1-z[:,t])*ht[:,t]+z[:,t]*h[:,t-1]

				output[:,t]=softmax(h[:,t].dot(Wo))

			# Initializing the gradients which will hold the errors during backpropagation
			gradient_Uz = np.zeros_like(Uz) #U
			gradient_Ur = np.zeros_like(Ur) #U
			gradient_Uh = np.zeros_like(Uh) #U

			gradient_Wz = np.zeros_like(Wz) #U
			gradient_Wr = np.zeros_like(Wr) #U
			gradient_Wh = np.zeros_like(Wh) #U

			gradient_Wo = np.zeros_like(Wo) #U

			dr=np.zeros_like(r)
			dsor=np.zeros_like(sor)
			dz=np.zeros_like(z)
			dht=np.zeros_like(ht)
			dh=np.zeros_like(h)

			#???
			lossgradients=[]
			loss=[]
			for y_i,o_i in zip(y_train,output):
				loss.append(abs(y_i-o_i))
				lossgradients.append(lossCE.derivative(y_i,o_i))

			lossgradients=array(lossgradients)
			loss=array(loss)

			dh[:,-1]=np.zeros((batch_size, size_of_hidden_layers))
			if not crawler%100:
			 	print "LOSS: ",np.mean(array(lossgradients))


			# Backpropagation
			for t in reversed(range(timesteps)):
				
		  		delta=np.zeros((batch_size,size_of_hidden_layers))
		  		#dh[:,t]=(loss[:,t]).dot(Wz.T)

		  		# Instead of going all the way back to timestep 0
		  		# We truncate our backpropagation
				for ti in reversed(np.arange(max(0, t - self.truncation), t+1)):

					delta+=(lossgradients[:,t]*softmax.prime(h[:,t].dot(Wo))).dot(Wo.T)*(1-z[:,t])*tanh.prime(q[:,t])

		        	dr[:,ti]=(Wr.dot(delta.T)).T*h[:,ti-1] #
		        	dz[:,ti]=(Wz.dot(delta.T)).T*r[:,ti]*(h[:,ti-2]-ht[:,ti-1])#
		        	
		        	gradient_Wo+=h[:,ti].T.dot(loss[:,ti]) #

		        	gradient_Uz+=(softmax.prime(z[:,ti])*dz[:,ti]).T.dot(X[:,ti])
		        	gradient_Ur+=(softmax.prime(r[:,ti])*dr[:,ti]).T.dot(X[:,ti])#
		        	gradient_Uh+=(X[:,ti-1].T.dot(delta)).T #

		        	gradient_Wz+=(softmax.prime(z[:,ti])*dz[:,ti]).T.dot(h[:,ti-1])
		        	gradient_Wr+=(softmax.prime(r[:,ti])*dr[:,ti]).T.dot(h[:,ti-1])#
		        	gradient_Wh+=delta.T.dot((h[:,t-1]*r[:,t]))#
			

			# Updating our Weight matrices
			self.Uz=SGD.update(Uz,gradient_Uz)
			self.Ur=SGD.update(Ur,gradient_Ur)
			self.Uh=SGD.update(Uh,gradient_Uh)

			self.Wz=SGD.update(Wz,gradient_Wz)
			self.Wr=SGD.update(Wr,gradient_Wr)
			self.Wh=SGD.update(Wh,gradient_Wh)

			self.Wo=SGD.update(Wo,gradient_Wo)


	def predict(self,X_test,y_test,number_of_batches_to_test=10):

		batch_size, timesteps, input_dim = X_test.shape
		size_of_hidden_layers=self.size_of_hidden_layers
		X=X_test

		r=np.zeros((batch_size, timesteps, size_of_hidden_layers))
		sor=np.zeros((batch_size, timesteps, size_of_hidden_layers))
		z=np.zeros((batch_size, timesteps, size_of_hidden_layers))
		ht=np.zeros((batch_size, timesteps, size_of_hidden_layers))
		h=np.zeros((batch_size, timesteps+1, size_of_hidden_layers))
		output=np.zeros((batch_size, timesteps, input_dim))

		#Performing a one last propagation for our final prediction

		for t in range(timesteps):
			r[:,t]=sigmoid(X[:,t].dot(self.Ur.T)+h[:,t-1].dot(self.Wr))
			z[:,t]=sigmoid(X[:,t].dot(self.Uz.T)+h[:,t-1].dot(self.Wz))
			###
			sor[:,t]=h[:,t-1]*r[:,t]
			ht[:,t]=tanh(X[:,t].dot(self.Uh.T)+sor[:,t].dot(self.Wh))
			h[:,t]=(1-z[:,t])*ht[:,t]+z[:,t]*h[:,t-1]

			output[:,t]=softmax(h[:,t].dot(self.Wo))

		print ("Results:")

		for i in range(number_of_batches_to_test):
			tmp_X = np.argmax(X_test[i], axis=1)
			tmp_y1 = np.argmax(y_test[i], axis=1)
			tmp_y2 = np.argmax(output[i], axis=1)

			
			print'\n'
			print "input number "+str(i)+" :	"
			print "X      = [" +  str(tmp_X) + "]"
			print "y_true = [" +str(tmp_y1)+ "]"
			print "y_predicted = [" +str(tmp_y2)+ "]"
			

def  main():
	X, y = gen_mult_ser(50,100)

	X_train, X_test, y_train, y_test = train_test_split(X, y, 0.2)
	Gru=GRU()
	Gru.forward_and_backwardpropagation(X_train,y_train)
	Gru.predict(X_test,y_test)

if __name__ == '__main__':
	main()

















