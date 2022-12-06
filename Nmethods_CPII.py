from scipy.optimize import newton
import numpy as np

class root_finder():
    'Implement Secant Method'
    pass


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
    def RK4(self):
        pass
    def mid_point(self):
        pass
        