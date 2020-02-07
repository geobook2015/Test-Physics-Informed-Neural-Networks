import numpy as np 
import types

class base():

    def __init__(self, func_list, N):
        self.func_list = func_list
        self.N = N

    def generate(self):
        return NotImplementedError

class Nothing(base):
    def __init__(self): self.N = 0

    def generate(self):
        xb = np.zeros((1, 3), dtype=np.float32)
        ub = np.zeros(1)

        return xb, ub
class dirichlet(base):

    def __init__(self, func_list, u_func, N=-1):
        super(dirichlet, self).__init__(func_list, N)
        self.u_func = u_func
        if N == -1:
            self.N = len(self.func_list)

    def generate(self):
    
        N_ = self.N // len(self.func_list)
    
        ret = []
        if type(self.u_func) is float or type(self.u_func) is int:
            for f_ in self.func_list:
                ret.append(f_.sampling(N_))
            
            ret = np.concatenate([r for r in ret], 0)
            u = self.u_func * np.ones(ret.shape[0])[:, None]
        
        elif type(self.u_func) is types.FunctionType:
            for f_ in self.func_list:
                temp = f_.sampling(N_)
                #print('temp', temp.shape)
                ret.append(temp)
            
            ret = np.concatenate([r for r in ret], 0)
            #print(ret.shape)
            u = self.u_func(ret)
            #print(u)
        
        elif type(self.u_func) is list:
            if type(self.u_func[0]) is list:
                u_list = []
                v_list = []
                for i, f_ in enumerate(self.func_list):
                    temp = f_.sampling(N_)
                    ret.append(temp)
                    u_list.append(self.u_func[0][i](temp))
                    v_list.append(self.u_func[1][i](temp))
               
                ret = np.concatenate([r for r in ret], 0)
                u_list = np.concatenate([u for u in u_list], 0)
                v_list = np.concatenate([u for u in v_list], 0)
                u = [u_list, v_list]
            else:
            #init
                u_list = []
                v_list = []
                for i, f_ in enumerate(self.func_list):
                    temp = f_.sampling(N_)
                    ret.append(temp)
                    u_list.append(self.u_func[0](temp))
                    v_list.append(self.u_func[1](temp))
                
           
                ret = np.concatenate([r for r in ret], 0)
                u_list = np.concatenate([u for u in u_list], 0)
                v_list = np.concatenate([v for v in v_list], 0)
           
                u = [u_list, v_list]
        
            #print('dirichlet', ret.shape, u.shape)
        return ret, u


class condition_list():

    def __init__(self, c_list, N=-1):
        self.N = N
        self.c_list = c_list

    def generate(self):
    
        N_ = self.N // len(self.c_list)
    
        ret = []
        for f_ in self.func_list:
            temp = f_.sampling(N_)
            ret.append(f_.sampling(N_))
        
            ret = np.concatenate([r for r in ret], 0)
    
        u = []
        for f_ in self.u_func:
            u.append(f_(ret))

        return ret, u


class Points_1d():

    def __init__(self, x):
        self.x = x

    def sampling(self, N):
        ret_x = self.x * np.ones(N)[:, None]
        return ret_x
class Line_1d():
    def __init__(self, lower, upper, dx, order_type='rand'): 
        #print(lower, upper, dx) 
        self.lower = lower + dx 
        self.upper = upper - dx

    def sampling(self, N, order_type='rand'):
        if order_type == 'rand':
            temp = (self.upper - self.lower) * np.random.random(N) + self.lower
        elif order_type == 'order':
            temp = np.linspace(self.lower, self.upper, N)
        
        #print('line', temp.shape)
        return temp[:, None]
class Points_2d():

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def sampling(self, N):
        ret_x = self.x * np.ones(N)
        ret_y = self.y * np.ones(N)
        return np.concatenate((ret_x[:, None], ret_y[:, None]), 1)

class Rectangle():
    def __init__(self, lower, upper, dx, order_type='rand'): 

        self.lower = lower 
        self.upper = upper
        if type(dx) is float:
            self.dx = [dx] * len(lower)
        elif type(dx) is list:
            self.dx = dx

    def sampling(self, N, order_type='rand'):

        if order_type == 'rand':
            ret = []
            for l, u, _dx in zip(self.lower, self.upper, self.dx):
                ret.append(Line_1d(l, u, _dx).sampling(N)[:, 0].tolist())
            ret = np.array(ret).T
        return ret

    def sampling_diagonal(self, N):
        #Diagonal sampling
        ret = []
        for l, u, _dx in zip(self.lower, self.upper, self.dx):
            #temp = np.arange(start = l, stop = u + x_step, step=x_step)
            ret.append(np.linspace(start = l, stop = u, num=N).tolist())
        ret = np.array(ret).T
        return ret
class Line_AxisX_2d():

    def __init__(self, X, lower, upper, dx=0, order_type='rand'): 
        self.x = X
        self.lower = lower + dx
        self.upper = upper - dx

    def sampling(self, N, order_type='rand'):
        ret_x = self.x * np.ones(N)
        if order_type == 'rand':
            ret_y = (self.upper - self.lower) * np.random.random(N) + self.lower
        return np.concatenate((ret_x[:, None], ret_y[:, None]), 1)


class Line_AxisY_2d(Line_AxisX_2d):
    def __init__(self, X, lower, upper, dx=0, order_type='rand'):
        super(Line_AxisY_2d, self).__init__(X, lower, upper, dx, order_type)

    def sampling(self, N, order_type='rand'):
        ret_y = self.x * np.ones(N)
        if order_type == 'rand':
            ret_x = (self.upper - self.lower) * np.random.random(N) + self.lower
        return np.concatenate((ret_x[:, None], ret_y[:, None]), 1)



class volume(base):

    def __init__(self, func_list, init_func, N):
        super(volume , self).__init__(func_list, N)
        self.init_func = init_func
        self.lower = func_list[0].lower
        self.upper = func_list[0].upper

    def generate(self):

        N_ = self.N // len(self.func_list)

        ret = []
        for f_ in self.func_list:

            ret.append(f_.sampling(N_).tolist())

        ret = np.array(ret)
        ret = np.swapaxes(ret, 1, 2)
        ret = ret.reshape((ret.shape[0] * ret.shape[1], ret.shape[2]))

        return ret, self.init_func(ret)[:, None]


    def generate_diagonal(self):

        N_ = self.N // len(self.func_list)

        ret = []
        for f_ in self.func_list:

            ret.append(f_.sampling_diagonal(N_).tolist())

        ret = np.array(ret)
        ret = np.swapaxes(ret, 1, 2)
        ret = ret.reshape((ret.shape[0] * ret.shape[1], ret.shape[2]))

        return ret, self.init_func(ret)[:, None]

if __name__ == '__main__':

    N = 200
    f_list1 = Line_OnX_2d(5, 0.1, 0.2, 0)
    f_list2 = Line_OnX_2d(5, 0.1, 0.2, 0)
    f_list3 = Line_OnY_2d(7, 0.1, 0.3, 0)
    #print(f_list3.sampling(N))

    cond = dirichlet([f_list1, f_list2, f_list3], 0, N)
    ret, u = cond.generate()

    #print(ret[:, 0])
    #print(ret[:, 1])
    print(ret)
    print(ret.shape, u.shape)


    #if False:
    if True:
        i_func = lambda d : np.sin(np.pi * d[:, 0]) * np.sin(np.pi * d[:, 1])
        #rec = Line(0, 1, 1e-5) + Line(0, 2, 1e-5)
        rec = Rectangle([0, 0], [1, 1], [1e-5, 1e-5])
        i_cond = volume([rec], i_func, 200)
        ret, u = i_cond.generate()
        #ret, u = i_cond.generate_diagonal()


        #print(ret[:, 0])
        #print(ret[:, 1])
        print(ret)
        print(ret.shape, np.max(ret), np.min(ret))
        print(u.shape, np.max(u), np.min(u))

        import pylab as pl
        pl.figure()
        pl.scatter(ret[:, 0], ret[:, 1])
        pl.grid()
        pl.show()
