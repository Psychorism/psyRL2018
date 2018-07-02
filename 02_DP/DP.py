import numpy as np
import random
import matplotlib.pyplot as plt

class Env :
    
    def __init__(self):
        
        self.course_reward = 0.
        self.end_reward = 1.
        
        self.terminal = [(2,2)]
        self.directions = ('up', 'down', 'west', 'east')
    
    def get_reward(self, state, action) :
        
        if self.move(state, action) in self.terminal :
            
            return self.end_reward
        
        else : return self.course_reward
    
    def move(self, state, action):
        a = [(-1,0),(1,0),(0,-1),(0,1)] # up, down, left, right
        Next = tuple(np.array(state) + np.array(a[action]))
        if not(Next[0] in range(3)): return state
        if not(Next[1] in range(3)): return state
        return Next
    
    def toGrid(self, position, ro) :
        pos_x = .5 + position[1]
        pos_y = ro - (.5 + position[0])
        return pos_x, pos_y
    
    def tracePlot(self, trace, colu, ro) :
        plt.xlim(0,colu)
        plt.ylim(0,ro)
        
        for i in range(ro+1) : plt.axhline(y=i, linewidth=2, color='black')
        for j in range(colu+1) : plt.axvline(x=j, linewidth=2, color='black')
        
        acts = np.diff(trace, axis=0) *.8
        
        for k, tr in enumerate(trace):
            
            new_x, new_y = self.toGrid(tr, ro)
            
            if k == len(trace) -1:
                plt.scatter(new_x, new_y, s=300, color='red', alpha=.4)
                break
            
            if k == 0 : colr='blue'
            else : colr='green'
            
            plt.scatter(new_x, new_y, s=300, color=colr, alpha=.4)
            plt.arrow(new_x, new_y, acts[k][1], -1.*acts[k][0], head_width=.2, head_length=.2, fc='k', ec='k')
        
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off')
        plt.draw()
        plt.show()


class randomPolicy :
    
    def __init__(self, env):
        self.env = env
        self.gamma = .7
        
        self.state = (0,0)
        self.action = np.random.choice(4)
        
        self.state_trace  = [self.state]
        self.action_trace = [self.action]
        
        self.printTuples()
    
    def randomWalk(self):
        while (self.state not in self.env.terminal) :
            Next = self.env.move(self.state, self.action)
            
            self.state = Next
            self.state_trace.append(self.state)
            
            self.action = np.random.choice(4)
            self.action_trace.append(self.action)

    def printTuples(self):
        self.randomWalk()
        for index, s in enumerate(self.state_trace) :
            if s in self.env.terminal :
                print ("t=%d"%index, "\tS=%s"%(s,), "\t", "\t", "\tR=%s"%self.env.end_reward, \
                       "\tgamma=%s"%self.gamma)
            else :
               print ("t=%d"%index, "\tS=%s"%(s,), "\tA=%s"%self.env.directions[self.action_trace[index]], \
                      "\tP=1", "\tR=%s"%self.env.course_reward, "\tgamma=%s"%self.gamma)
            
        self.env.tracePlot(self.state_trace, 3,3)


class Simulator :
    
    def __init__(self, env):
        self.env = env
        self.gamma = .7
        
        self.V = np.zeros((3,3))
        self.N = np.zeros((3,3))
        
        print (self.simulate())
    
    def simulate(self, iteration=1000) :
        
        for i in range(iteration) :
            
            self.state  = (0,0)
            self.action = np.random.choice(4)
            
            self.state_trace  = [self.state]
            self.action_trace = [self.action]
            
            while (self.state not in self.env.terminal) :
                Next = self.env.move(self.state, self.action)
                
                self.state = Next
                self.state_trace.append(self.state)
                
                self.action = np.random.choice(4)
                self.action_trace.append(self.action)
        
            self.steps = len(self.state_trace)
            
            for index, s in enumerate(self.state_trace) :
                self.returns = 0.
                
                for j in range(self.steps-index-2) :
                    self.returns += self.env.course_reward * (self.gamma**j)
            
                self.returns += self.env.end_reward * (self.gamma**(j+1))
                
                self.N[s] += 1.
                self.V[s] += (self.returns - self.V[s])/self.N[s]
        
        return self.V


class Bellman :
    
    def __init__(self, env):
        self.env = env
        self.gamma = .7
        
        self.V = np.zeros((3,3))
        self.states = []
        for i in range(3) :
            for j in range(3) :
                self.states.append((i,j))
        
        random.shuffle(self.states)
        print (self.DP())
    
    def DP(self):
        for self.state in self.states :
            
            # Bellman Expectation Equation
            
            for action in range(4):
                Next   = self.env.move(self.state, action)
                reward = self.env.get_reward(self.state, action)
                self.V[self.state] += (.25 * (reward + self.gamma * self.V[Next]))

        return self.V


if __name__ == "__main__":
    env = Env()
    # randomPolicy(env)
    # Simulator(env)
    Bellman(env)
