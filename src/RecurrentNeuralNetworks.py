import numpy as np
from Optimizers import SGD
from ActivationLoss import lossCE, sigmoid, tanh_func, softmax

def array(x):
	return x if type(x).__module__ == np.__name__ else np.array(x)

def to_categorical(x, n_col=None):
    """ One-hot encoding of nominal values """
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot


def gen_mult_ser(nums,timesteps):
        """ Method which generates multiplication series """
        """ Written by Erik Linder-Norén"""
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
    """Written by Erik Linder-Norén """
    length=int(len(X)*split_size)
    X_train, X_test = X[:length], X[length:]
    y_train, y_test = y[:length], y[length:]

    return X_train, X_test, y_train, y_test


lossCE=lossCE()
SGD=SGD()
softmax=softmax()
tanh=tanh_func()

class RNN:

	def __init__(self,size_of_hidden_layers=100,epochs=100,truncation=10):

		self.epochs=epochs
		self.size_of_hidden_layers=size_of_hidden_layers
		self.truncation=truncation

	def forward_and_backwardpropagation(self,X_train,y_train):
		batch_size, timesteps, input_dim = X_train.shape
		size_of_hidden_layers=self.size_of_hidden_layers
		X=X_train

		## Initializing the wight matrices according to some external material

		self.from_input_to_hidden_state=np.random.uniform(-1./np.sqrt(input_dim),1./np.sqrt(input_dim),((size_of_hidden_layers, input_dim)))
		self.from_hidden_state_to_hidden_state=np.random.uniform(-1./np.sqrt(size_of_hidden_layers),1./np.sqrt(size_of_hidden_layers),(size_of_hidden_layers, size_of_hidden_layers))
		self.from_hidden_state_to_output=np.random.uniform(-1./np.sqrt(size_of_hidden_layers),1./np.sqrt(size_of_hidden_layers),(input_dim, size_of_hidden_layers))

		# Beginning of the forward propagation for a certain number of epochs

		i=0
		for i in range(self.epochs):
			i+=1
			s=np.zeros((batch_size, timesteps, size_of_hidden_layers))
			vst=np.zeros((batch_size, timesteps+1, size_of_hidden_layers))
			o=np.zeros((batch_size, timesteps, input_dim))
			output=np.zeros((batch_size, timesteps, input_dim))


			s[:,-1] = np.zeros((batch_size, size_of_hidden_layers))

			for t in range(timesteps):

				## Using the RNN equations
				#s[i]=np.tanh(from_input_to_hidden_state.dot(x[i-1])+from_hidden_state_to_hidden_state.dot(s[i-1]))
				s[:,t]=X[:,t].dot(self.from_input_to_hidden_state.T) + s[:,t-1].dot(self.from_hidden_state_to_hidden_state.T)
				#vst[i]=from_hidden_state_to_output.T.dot(s[i])
				vst[:, t] = tanh(s[:,t])
				o[:,t]=vst[:,t].dot(self.from_hidden_state_to_output.T)
				output[:,t]=softmax(o[:,t])




			# Initializing the gradients which will hold the errors during backpropagation

			gradient_input_to_hidden = np.zeros_like(self.from_input_to_hidden_state) #U
			gradient_hidden_to_hidden = np.zeros_like(self.from_hidden_state_to_hidden_state) #W
			gradient_hidden_to_output = np.zeros_like(self.from_hidden_state_to_output) #V

			## Calculating the Cross Entropy loss and its derivative

			lossgradients=[]
			for y_i,o_i in zip(y_train,output):
				lossgradients.append(lossCE.derivative(y_i,o_i))

			lossgradients=array(lossgradients)

			if not i%100:
				print "LOSS: ",np.mean(array(lossgradients))

			for t in reversed(range(timesteps)):
				
				gradient_hidden_to_output += lossgradients[:,t].T.dot(vst[:,t])
		        deltas = lossgradients[:,t].dot(self.from_hidden_state_to_output) * tanh.prime(s[:,t])
		        # Instead of going all the way back to timestep 0
		  		# We truncate our backpropagation

		        for i in reversed(np.arange(max(0, t - self.truncation), t+1)):

					gradient_input_to_hidden += deltas.T.dot(X[:,i])
					gradient_hidden_to_hidden += deltas.T.dot(vst[:,i-1])
			        # Calculate gradient w.r.t previous state
					deltas = deltas.dot(gradient_hidden_to_hidden) * tanh.prime(s[:,i-1])

			# Updating our Weight matrices
			self.from_input_to_hidden_state=SGD.update(self.from_input_to_hidden_state,gradient_input_to_hidden)
			self.from_hidden_state_to_hidden_state=SGD.update(self.from_hidden_state_to_hidden_state,gradient_hidden_to_hidden)
			self.from_hidden_state_to_output=SGD.update(self.from_hidden_state_to_output,gradient_hidden_to_output)


	def predict(self,X_test,y_test,number_of_batches_to_test=10):

		batch_size, timesteps, input_dim = X_test.shape
		size_of_hidden_layers=self.size_of_hidden_layers
		X=X_test

		s=np.zeros((batch_size, timesteps, size_of_hidden_layers))
		vst=np.zeros((batch_size, timesteps+1, size_of_hidden_layers))
		o=np.zeros((batch_size, timesteps, input_dim))

		s[:,-1] = np.zeros((batch_size, size_of_hidden_layers))

		output=np.zeros((batch_size, timesteps, input_dim))

		# Given the weight matrices we may now perform a last backpropagation
		# on new data to test

		for t in range(timesteps):
			#s[i]=np.tanh(from_input_to_hidden_state.dot(x[i-1])+from_hidden_state_to_hidden_state.dot(s[i-1]))
			s[:,t]=X[:,t].dot(self.from_input_to_hidden_state.T) + s[:,t-1].dot(self.from_hidden_state_to_hidden_state.T)
			#vst[i]=from_hidden_state_to_output.T.dot(s[i])
			vst[:, t] = tanh(s[:,t])
			o[:,t]=vst[:,t].dot(self.from_hidden_state_to_output.T)
			output[:,t]=softmax(o[:,t])



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
			

def main():
	X, y = gen_mult_ser(50,100)

	X_train, X_test, y_train, y_test = train_test_split(X, y, 0.2)
	Rnn=RNN()
	Rnn.forward_and_backwardpropagation(X_train,y_train)
	Rnn.predict(X_test,y_test)

if __name__ == '__main__':
	main()
















