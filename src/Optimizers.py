import numpy as np 

class SGD:
    
    def update(self, w, loss_deriv):
        self.momentum = .01
        self.learningrate=.01
        self.w_update = None
        if not self.w_update :
            self.w_update = np.zeros_like(w)
        self.w_update = self.momentum * self.w_update + (1 - self.momentum) * loss_deriv
        return w - self.learningrate * self.w_update


