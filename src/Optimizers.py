import numpy as np 

class SGD:
    
    def update(self, w, loss_deriv):
        self.momentum = .01
        self.learningrate=.01
        self.w_update = None
        # If not initialized
        if not self.w_update :
            self.w_update = np.zeros_like(w)
        # Use momentum if set
        self.w_update = self.momentum * self.w_update + (1 - self.momentum) * loss_deriv
        # Move against the gradient to minimize loss
        return w - self.learningrate * self.w_update


