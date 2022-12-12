from scipy.optimize import newton
import numpy as np

class root_finder():
    
    def __init__(self, function, kmax = 200, eps = 1e-8):
        self.f = function
        self.tol = eps
        self.iter = kmax
    
    'Implement Secant Method'
    def secant(self, xs):
        f0 = self.f(xs[0])
        for k in range(self.iter + 1):
            f1 = self.f(xs[1])
            ratio = (xs[1] - xs[0])/(f1 - f0)
            x2 = xs[1] - f1*ratio
            xdiff = np.abs(x2 - xs[1])
            xs[0], xs[1] = xs[1], x2
            f0 = f1
            
            if np.abs(xdiff/x2) < self.tol:
                break
        else:
            x2 = None
        
        return x2

class ODE_methods:
    def __init__(self, ri, rf, inc, F, h = 1e-1):
        '''
        F: callable variable [Function]
        '''
        self._grid = np.arange(ri, rf + h, h) # Take care of the upper limit
        self._F_array = np.zeros(len(self._grid))
        self._F_array[0] = inc #Important, initial condition
        self._F = F
        self._step = h 
        
    def Forward_Euler_Method(self):
        for j in range(len(self._grid) - 1):
            self._F_array[j+1] = self._F_array[j] + self._step * self._F(self._grid[j], self._F_array[j])
        return self._F_array

    def Backward_Euler_Method(self):
        guess  = self.Forward_Euler_Method() # Type -> Numpy Array
        g = lambda z,j: z - self._F_array[j] - self._step*self._F(self._grid[j+1],z) # z -> S_{t+1}; Equation to solve: z = sj + h*F(t_{j+1},z)
        for k in range(len(self._grid)-1):
            self._F_array[k+1] = newton(g,guess[k+1], args= [k])
        return self._F_array

    def Trapezoidal_Euler_Method(self):
        guess  = self.Forward_Euler_Method()
        g = lambda z,j: z - self._F_array[j] - (self._step/2)*( self._F(self._grid[j], self._F_array[j]) + self._F(self._grid[j+1],z) )                                              
        for k in range(len(self._grid)-1):
            self._F_array[k+1] = newton(g,guess[k+1], args= [k])
        return self._F_array
    
    def RK2(self):
        pass
    def RK4_Single(self):
        for k in range(len(self.grid) - 1):
            k0 = self._step*self._F(self._grid[k], self._F_array[k])
            k1 = self._step*self._F(self._grid[k] + (1/2)*self.step, self._F_array[k] + (1/2)*k0 )
            k2 = self._step*self._F(self._grid[k] + (1/2)*self.step, self._F_array[k] + (1/2)*k1 )
            k3 = self._step*self._F(self._grid[k] + self.step, self._F_array[k] + k2 )
            self._F_array[k+1] = self._F_array[k] + (1/6)*(k0 + 2*k1 + 2*k2 + k3)
        return self._F_array
            
    def mid_point(self):
        pass
if __name__=='__main__':
    f = lambda x: np.exp(x - np.sqrt(x)) - x
    root0 = root_finder(f).secant([0.,1.7])
    root1 = root_finder(f).secant([2.,2.1])
    print(root0, root1)
    
        