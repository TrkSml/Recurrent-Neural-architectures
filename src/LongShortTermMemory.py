# __author__ = Tarek Samaali 

import numpy as np
from Optimizers import SGD
from ActivationLoss import lossCE, sigmoid, tanh_func, softmax

def array(x):
	return x if type(x).__module__ == np.__name__ else np.array(x)


def to_categorical(x, n_col=None):
    """ One-hot encoding of nominal values """
    """ Written by Erik Linder-Norén"""
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot


def gen_mult_ser(nums,timesteps):
        """ Method which generates multiplication series """
        """ Written by Erik Linder-Norén """
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

class LSTM:

	def __init__(self,size_of_hidden_layers=100,epochs=100,truncation=10):

		self.epochs=epochs
		self.size_of_hidden_layers=size_of_hidden_layers
		self.truncation=truncation

	def forward_and_backwardpropagation(self,X_train,y_train):
		batch_size, timesteps, input_dim = X_train.shape
		size_of_hidden_layers=self.size_of_hidden_layers
		X=X_train

		## Initializing the wight matrices according to some external material

		self.Uc=np.random.uniform(-1./np.sqrt(input_dim),1./np.sqrt(input_dim),((size_of_hidden_layers, input_dim)))
		self.Ui=np.random.uniform(-1./np.sqrt(input_dim),1./np.sqrt(input_dim),((size_of_hidden_layers, input_dim)))
		self.Uf=np.random.uniform(-1./np.sqrt(input_dim),1./np.sqrt(input_dim),((size_of_hidden_layers, input_dim)))
		self.Uo=np.random.uniform(-1./np.sqrt(input_dim),1./np.sqrt(input_dim),((size_of_hidden_layers, input_dim)))

		self.Wc=np.random.uniform(-1./np.sqrt(size_of_hidden_layers),1./np.sqrt(size_of_hidden_layers),(size_of_hidden_layers, size_of_hidden_layers))
		self.Wi=np.random.uniform(-1./np.sqrt(size_of_hidden_layers),1./np.sqrt(size_of_hidden_layers),(size_of_hidden_layers, size_of_hidden_layers))
		self.Wf=np.random.uniform(-1./np.sqrt(size_of_hidden_layers),1./np.sqrt(size_of_hidden_layers),(size_of_hidden_layers, size_of_hidden_layers))
		self.Wo=np.random.uniform(-1./np.sqrt(size_of_hidden_layers),1./np.sqrt(size_of_hidden_layers),(size_of_hidden_layers, size_of_hidden_layers))

		self.Wz=np.random.uniform(-1./np.sqrt(size_of_hidden_layers),1./np.sqrt(size_of_hidden_layers),(size_of_hidden_layers, input_dim))

		Uc=self.Uc
		Ui=self.Ui
		Uf=self.Uf
		Uo=self.Uo

		Wc=self.Wc
		Wi=self.Wi
		Wf=self.Wf
		Wo=self.Wo

		Wz=self.Wz

		# Beginning of the forward propagation for a certain number of epochs

		for crawler in range(self.epochs):

			f=np.zeros((batch_size, timesteps, size_of_hidden_layers))
			i=np.zeros((batch_size, timesteps, size_of_hidden_layers))
			o=np.zeros((batch_size, timesteps, size_of_hidden_layers))
			c=np.zeros((batch_size, timesteps, size_of_hidden_layers))
			g=np.zeros((batch_size, timesteps, size_of_hidden_layers))
			h=np.zeros((batch_size, timesteps+1, size_of_hidden_layers))

			output=np.zeros((batch_size, timesteps, input_dim))

			h[:,-1] = np.zeros((batch_size, size_of_hidden_layers))

			for t in range(timesteps):

				## Applying the LSTM equations

				f[:,t]=sigmoid(X[:,t].dot(Uf.T)+h[:,t-1].dot(Wf))
				i[:,t]=sigmoid(X[:,t].dot(Ui.T)+h[:,t-1].dot(Wi))
				o[:,t]=tanh(X[:,t].dot(Uo.T)+h[:,t-1].dot(Wo))
				g[:,t]=sigmoid(X[:,t].dot(Uc.T)+h[:,t-1].dot(Wc))
				c[:,t]=f[:,t]*c[:,t-1]+i[:,t]*g[:,t]

				h[:,t]=o[:,t]*tanh(c[:,t])

				output[:,t]=softmax(h[:,t].dot(Wz))

			# Initializing the gradients which will hold the errors during backpropagation
			gradient_Uc = np.zeros_like(Uc) #U
			gradient_Ui = np.zeros_like(Ui) #U
			gradient_Uf = np.zeros_like(Uf) #U
			gradient_Uo = np.zeros_like(Uo) #U

			gradient_Wc = np.zeros_like(Wc) #U
			gradient_Wi = np.zeros_like(Wi) #U
			gradient_Wf = np.zeros_like(Wf) #U
			gradient_Wo = np.zeros_like(Wo) #U

			gradient_Wz = np.zeros_like(Wz) #U

			df=np.zeros_like(f)
			di=np.zeros_like(i)
			do=np.zeros_like(o)
			dg=np.zeros_like(g)
			dc=np.zeros_like(c)

			dh=np.zeros_like(h)

			#???

			## Calculating the Cross Entropy loss and its derivative
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
				
		  		dh[:,t]=dh[:,t-1]+(loss[:,t]).dot(Wz.T)
		  		#dh[:,t]=(loss[:,t]).dot(Wz.T)

		  		# Instead of going all the way back to timestep 0
		  		# We truncate our backpropagation
				for ti in reversed(np.arange(max(0, t - self.truncation), t+1)):

					do[:,ti]=tanh(c[:,ti])*dh[:,ti]
		        	dc[:,ti]=tanh.prime(c[:,ti])*o[:,ti]*dh[:,ti]
		        	df[:,ti]=c[:,ti-1]*dc[:,ti]
		        	dc[:,ti-1]+=f[:,ti-1]*dc[:,ti]
		        	di[:,ti]=g[:,ti]*dc[:,ti]
		        	dg[:,ti]=di[:,ti]*dc[:,ti]

		        	#gradient_Wz+=h[:,ti].T.dot(lossgradients[:,ti])
		        	gradient_Wz+=h[:,ti].T.dot(loss[:,ti])

		        	gradient_Uo+=(tanh.prime(o[:,ti])*do[:,ti]).T.dot(X[:,ti])
		        	gradient_Ui+=(softmax.prime(i[:,ti])*di[:,ti]).T.dot(X[:,ti])
		        	gradient_Uf+=(softmax.prime(f[:,ti])*df[:,ti]).T.dot(X[:,ti])
		        	gradient_Uc+=(softmax.prime(g[:,ti])*dg[:,ti]).T.dot(X[:,ti])

		        	gradient_Wo+=(tanh.prime(o[:,ti])*do[:,ti]).T.dot(h[:,ti+1])
		        	gradient_Wi+=(softmax.prime(i[:,ti])*di[:,ti]).T.dot(h[:,ti+1])
		        	gradient_Wf+=(softmax.prime(f[:,ti])*df[:,ti]).T.dot(h[:,ti+1])
		        	gradient_Wc+=(softmax.prime(g[:,ti])*dg[:,ti]).T.dot(h[:,ti+1])
			

			# Updating our Weight matrices
			self.Ui=SGD.update(Ui,gradient_Ui)
			self.Uf=SGD.update(Uf,gradient_Uf)
			self.Uo=SGD.update(Uo,gradient_Uo)
			self.Uc=SGD.update(Uc,gradient_Uc)

			self.Wi=SGD.update(Wi,gradient_Wi)
			self.Wf=SGD.update(Wf,gradient_Wf)
			self.Wo=SGD.update(Wo,gradient_Wo)
			self.Wc=SGD.update(Wc,gradient_Wc)

			self.Wz=SGD.update(Wz,gradient_Wz)


	def predict(self,X_test,y_test,number_of_batches_to_test=10):

		batch_size, timesteps, input_dim = X_test.shape
		size_of_hidden_layers=self.size_of_hidden_layers
		X=X_test

		f=np.zeros((batch_size, timesteps, size_of_hidden_layers))
		i=np.zeros((batch_size, timesteps, size_of_hidden_layers))
		o=np.zeros((batch_size, timesteps, size_of_hidden_layers))
		c=np.zeros((batch_size, timesteps, size_of_hidden_layers))
		g=np.zeros((batch_size, timesteps, size_of_hidden_layers))
		h=np.zeros((batch_size, timesteps+1, size_of_hidden_layers))

		output=np.zeros((batch_size, timesteps, input_dim))
		pre_output=np.zeros_like(output)

		h[:,-1] = np.zeros((batch_size, size_of_hidden_layers))

		# Given the weight matrices we may now perform a last backpropagation
		# on new data to test

		for t in range(timesteps):

			f[:,t]=sigmoid(X[:,t].dot(self.Uf.T)+h[:,t].dot(self.Wf))
			i[:,t]=sigmoid(X[:,t].dot(self.Ui.T)+h[:,t].dot(self.Wi))
			o[:,t]=tanh(X[:,t].dot(self.Uo.T)+h[:,t].dot(self.Wo))
			g[:,t]=sigmoid(X[:,t].dot(self.Uc.T)+h[:,t].dot(self.Wc))
			c[:,t]=f[:,t]*c[:,t-1]+i[:,t]*g[:,t]

			h[:,t]=o[:,t]*tanh(c[:,t])

			pre_output[:,t]=h[:,t].dot(self.Wz)
			output[:,t]=softmax(h[:,t].dot(self.Wz))

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
	Lstm=LSTM()
	Lstm.forward_and_backwardpropagation(X_train,y_train)
	Lstm.predict(X_test,y_test)

if __name__ == '__main__':
	main()




