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
    
    def RK2_midpoint(self):
        for k in range(len(self.grid) - 1):
            k0 = self._step*self._F(self._grid[k], self._F_array[k])
            k1 = self._step*self._F(self._grid[k] + self._step/2, self._F_array[k] + k0/2)
            self._F_array[k+1] = self._F_array[k] + k1
        
        return self._F_array
    
    def RK2_trapezoidal(self):
        for k in range(len(self.grid) - 1):
            k0 = self._step*self._F(self._grid[k], self._F_array[k])
            k1 = self._step*self._F(self._grid[k] + self.step, self._F_array[k] + k0 )
            self._F_array[k+1] = self._F_array[k] + (1/2)*(k0 + k1)
        
        return self._F_array
    
    def RK4(self):
        for k in range(len(self.grid) - 1):
            k0 = self._step*self._F(self._grid[k], self._F_array[k])
            k1 = self._step*self._F(self._grid[k] + (1/2)*self.step, self._F_array[k] + (1/2)*k0 )
            k2 = self._step*self._F(self._grid[k] + (1/2)*self.step, self._F_array[k] + (1/2)*k1 )
            k3 = self._step*self._F(self._grid[k] + self.step, self._F_array[k] + k2 )
            self._F_array[k+1] = self._F_array[k] + (1/6)*(k0 + 2*k1 + 2*k2 + k3)
        return self._F_array
            
'''
 1.- Code a system of ordinary differential equations solver - Done
 2.- Code the Shooting Method 
 3.- Code the Finite Difference Method
 
'''

'''   
Theory:
    Book [Computational Methods for Physics by Joel Franklin]
        Read Physical Motivation [2.1] - Done 
        Read "The Verlet method" [2.2]
        Read "Adaptive step size" [2.4.1]
        Read "Vectors" [2.4.2]
        Read "Stability of numerical methods" [2.5]
        Read "Multi-step methods" [2.6]
        
%%%%%%%%%%%%%%%Then decide which problems, we will do later.
'''

class ODE_SYS_solver:
    def __init__(self, interval_limits, F, yinits_ybound, shoot = False, h = 1e-3):
        self.step = h
        self.x_array = np.arange(interval_limits[0], interval_limits[1] + self.step,
                                 self.step)
        
        self.yinits_ybound = yinits_ybound
        
        if shoot:
            self.boundary = yinits_ybound[1] 
            
        self._ys =  np.zeros ((len(self.x_array), self.yinits_ybound.size))
        
        self._F = F
        
    def RK4(self):
        self._ys[0,:] = self.yinits_ybound
        for k in range(len(self.x_array) - 1):
            
            k0 = self.step*self._F(self.x_array[k], self._ys[k,:])
            k1 = self.step*self._F(self.x_array[k] + (1/2)*self.step, self._ys[k,:] + (1/2)*k0 )
            k2 = self.step*self._F(self.x_array[k] + (1/2)*self.step, self._ys[k,:] + (1/2)*k1 )
            k3 = self.step*self._F(self.x_array[k] + self.step, self._ys[k,:] + k2 )
            self._ys[k+1,:] = self._ys[k,:] + (1/6)*(k0 + 2*k1 + 2*k2 + k3)
            
        return self._ys
    
    def shoot(self, sig):
        self.yinits_ybound = np.array([self.yinits_ybound[0], sig])  # From Boundary value problem to an initial-valued problem.
        
        ys = self.RK4()
        
        return ys[-1, 0] - self.boundary
        
        
        
        

if __name__=='__main__':
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    def fs(x,yvals):
        y0, y1 = yvals
        f0 = y1
        f1 = - (30/(1-x**2))*y0 + ((2*x)/(1-x**2))*y1
        return np.array([f0, f1])
    
    
    solver = ODE_SYS_solver([0.05,0.49], fs, np.array([0.0926587109375, 0.1117705085875]))
    wder = root_finder(solver.shoot).secant([0.,1.])
    print(wder)
    
    '''
    
    # This section seems to work. Compare two different Runge-Kutta method implementation 
    
    solver = ODE_SYS_solver([0.05,0.98], fs, np.array([0.0926587109375, 1.80962109375]))
    
    sol = solve_ivp(fs, [0.05,0.99], [0.0926587109375, 1.80962109375], t_eval = solver.x_array)
    
    
    plt.figure(figsize = (8,6))
    plt.plot(solver.x_array, solver.RK4()[:,0], "b", linestyle = '--', label ="Home-made method")
    plt.plot(sol.t, sol.y[0], "r", linestyle = '-', label ="SciPy method")
    plt.xlabel('x')
    plt.ylabel('f(x)')
    
    #plt.xlim(0,2.0)
    #plt.ylim(0,1.1)
    
    plt.legend()
    plt.show()
    '''
        