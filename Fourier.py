import numpy as np
import gym

env = gym.make('Acrobot-v1') # change if desired to required environment, ensure you change num_actions,sample given below
#env=  gym.make('MountainCar-v0') 
#env=gym.make('CartPole-v1') #change num_actions to 2
num_actions=3 #number of available actions
num_episodes=300
fourier_order=3 #change order as desired.
basealpha=0.001#change required base alpha
observations_dim=np.shape(env.observation_space.high)[0] #the observations in the environment
w=np.zeros([pow(fourier_order+1,observations_dim),num_actions])  #weight matrix, with number of columns equal number of actions
stepcount=np.zeros([num_episodes,1])
gamma=1 #discount factor
zeta=0.9 #bootstrapping parameter, note that lambda is keyword in python
epsilon=0 #set exploration parameter as desired
visualize_after_steps=10 #start the display

def createalphas(basealpha,fourier_order,observations_dim):  #different alpha for different order terms of fourier
    temp=tuple([np.arange(fourier_order+1)]*observations_dim)
    b=np.array(np.meshgrid(*temp)).T.reshape(-1,observations_dim)
    c=np.linalg.norm(b,axis=1)
    d=basealpha/c
    d[0]=basealpha
    d = np.expand_dims(d, axis=1)
    alphavec=np.tile(d,num_actions)
    alphavec=np.reshape(alphavec,(-1,num_actions))
    return alphavec
    
def normalize(state):
    
    normstate=np.empty(np.shape(state))
    val=env.observation_space.low
    val1=env.observation_space.high
    
    for i in range(np.shape(state)[0]):
        normstate[i]=translate(state[i],val[i],val1[i],0,1)
    return normstate
    
def computeFourierBasis(state,fourier_order,observations_dim):
    normstate=normalize(state)
    temp=tuple([np.arange(fourier_order+1)]*observations_dim)
    b=np.array(np.meshgrid(*temp)).T.reshape(-1,observations_dim)
    return np.cos(np.pi*np.dot(b,normstate))        

def computevalue(w,action,state): #compute value of taking some state in some state
    return np.dot(w[:,action],computeFourierBasis(state,fourier_order,observations_dim))
    
def updateweights(w,e,alphavec,delta):
    
    w= w+ delta*alphavec*e;
    return w
    
def epsilon_greedy(state,epsilon,w): #pass a state where agent is eps-greedy, weight matrix w
    
    temp=np.zeros([1,num_actions])
    for k in range(num_actions):
        temp[0,k]=computevalue(w,k,state)
    c=np.argmax(temp) 
    
    if np.random.rand(1)< epsilon:
        c=env.action_space.sample() #epsilon greedy
    return c
    
def translate(value, leftMin, leftMax, rightMin, rightMax):
   
    leftrange = leftMax - leftMin
    rightrange = rightMax - rightMin
    #Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / leftrange
     #Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightrange)
    
alphavec=createalphas(basealpha,fourier_order,observations_dim)
#env.monitor.start('/tmp/acrobot-experiment-1',force='True')
for i in range(int(num_episodes)):
    env.reset()
    curstate = env.observation_space.sample()
    e=np.zeros(np.shape(w))
    curaction=epsilon_greedy(curstate,epsilon,w)  #epsilon greedy selection
    while True:
        #print(normalize(curstate))
        if i>visualize_after_steps:
            env.render()
        stepcount[i,0]=stepcount[i,0]+1
        e[:,curaction]=e[:,curaction]+ computeFourierBasis(curstate,fourier_order,observations_dim); #accumulating traces
        nextstate,reward, done, info = env.step(curaction) 
        delta = reward - computevalue(w,curaction,curstate);   #The TD Error                    

        if done:
            print("Episode %d finished after %d timesteps" %(i,stepcount[i,0]))
            w=updateweights(w,e,alphavec,delta)
            break
        
        nextaction=epsilon_greedy(nextstate,epsilon,w)
        #print(nextaction)
        delta=delta+ gamma*computevalue(w,nextaction,nextstate)
        w=updateweights(w,e,alphavec,delta) #update the weight vector
        e=e*gamma*zeta             #trace decay parameter zeta
        curstate=nextstate
        curaction=nextaction
        
        #if stepcount[i]>1000:
            #print('failed')
            #break
            
env.monitor.close()

    
    
    
