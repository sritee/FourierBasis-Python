# Simulate the maze
import numpy as np
class Environment(object):
    def __init__(self,size,start,goal,obstacles): #Choose the walls, choose the goal state, start state, and size of your gridworl
                                                  #input the parameter as numpy arrays
        self.obstacles=obstacles
        self.goal=goal
        self.size=size
        self.curs=start
        
    def step(self,action):
        
        r=self.curs[0]
        c=self.curs[1]
        
        if action==0:
            c=c+1 # move right
        elif action==1:
            c=c-1 #move left
        elif action==2:
            r=r-1 # move up
        elif action==3:
            r=r+1 #move down
            
        ns=np.array([r,c])
        self.ns=ns
                
        if (c> self.size[1] or c< 1  or  r> self.size[0] or r<1): #checked moved out of gridworld
            self.ns=self.curs
        
        if (any((self.ns==self.obstacles).all(axis=1))):  # Have we moved  into an obstacle
            self.ns=self.curs
            
        

        if (self.goal==self.ns).all(): # reached goal state
            self.signal=1
            self.reward=1
        else:
            self.reward=-1
            self.signal=0
        self.curs=self.ns
        

