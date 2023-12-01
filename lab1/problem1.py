# Implementation done by Tawsiful Islam - tawsiful@kth.se
#                    and Guanyu Lin     - guanyul@kth.se
# unless stated someone else.
# Lab assignment in Reinforcement Learning EL2805

import numpy as np
import matplotlib.pyplot as plt
import time
import random
from IPython import display

# Implemented methods
methods = ['DynProg', 'ValIter'];

# Some colours
LIGHT_RED    = '#FFC4CC'
LIGHT_GREEN  = '#95FD99'
BLACK        = '#000000'
WHITE        = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'
LIGHT_ORANGE = '#FAE0C3'
BROWN = '#A0522D'

class Maze:
    """This class uses the same class created by Alessio Russo alessior@kth.se"""
    ACTIONS = {0: "stay",
               1: "left",
               2: "up",
               3: "right",
               4: "down"}
    REWARD_IMPOSSIBLE = -10000
    REWARD_MINOTAUR = -100
    REWARD_STEP = -1
    REWARD_GOAL = 0

    def __init__(self, maze_mat, weights=None, random_rewards=False, allow_stay=True):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze_mat # Matrix of the maze
        self.stay                     = allow_stay
        self.actions                  = self.__create_actions() #Callable function to get action
        self.states, self.map         = self.__create_states() #Callable function to get state
        self.n_actions                = len(self.actions)
        self.n_states                 = len(self.states)
        self.transition_probabilities = self.__transitions()
        self.rewards                  = self.__rewards(weights=weights,
                                                random_rewards=random_rewards)
        
        self.minotaur_pos = None # Position of the minotaur that needs to be sent here

    def __create_actions(self):
        ACTIONS = {0: (0,0),    #stay
               1: (0,-1),   #left (y_pos, x_pos) the inner list corresponds to one row in the maze
               2: (-1,0), #up (we go up in matrix thus smaller index)
               3: (0,1),  #right
               4: (1,0)}   #down
        return ACTIONS
    
    def __create_states(self):
        # state is [(y_pos,x_pos),(ym_pos,xm_pos)] ym and xm represent the monster location
        states = dict()
        map = dict()
        s = 0
        for y_pos in range(self.maze.shape[0]):
            for x_pos in range(self.maze.shape[1]):
                for ym_pos in range(self.maze.shape[0]):
                    for xm_pos in range(self.maze.shape[1]):
                        if self.maze[y_pos][x_pos] != 1:
                            states[s] = ((y_pos,x_pos),(ym_pos,xm_pos))
                            map[((y_pos,x_pos),(ym_pos,xm_pos))] = s
                            s+=1
        return states, map

    def getMinotaur_actions(self, state):
        # Get possible minotaur actions.
        currentM_row = self.states[state][1][0]
        currentM_col = self.states[state][1][1]
        #check if minotaur is on the border of the maze
        border_of_maze =  (currentM_row == 0) or (currentM_row== self.maze.shape[0]-1) or \
                                (currentM_col == 0) or (currentM_col== self.maze.shape[1]-1)
        #collect minotaur possible actions and store them in m_possibleActions.
        m_possibleActions = []
        if border_of_maze:
            if self.stay:
                for m_action in list(self.actions.keys()):
                    row_m = self.states[state][1][0] + self.actions[m_action][0]
                    col_m = self.states[state][1][1] + self.actions[m_action][1]
                    outside_of_maze =  (row_m == -1) or (row_m == self.maze.shape[0]) or \
                                        (col_m == -1) or (col_m == self.maze.shape[1])
                    if not outside_of_maze:
                        m_possibleActions.append(m_action)
            else:
                for m_action in list(self.actions.keys())[1:]:
                    row_m = self.states[state][1][0] + self.actions[m_action][0]
                    col_m = self.states[state][1][1] + self.actions[m_action][1]
                    outside_of_maze =  (row_m == -1) or (row_m == self.maze.shape[0]) or \
                                        (col_m == -1) or (col_m == self.maze.shape[1])
                    if not outside_of_maze:
                        m_possibleActions.append(m_action)
        else:
            m_possibleActions = list(self.actions.keys())[1:]
        return m_possibleActions
    
    def __move(self, state, action,m_action=None):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        # Compute the future position given current (state, action)
        new_y_pos= self.states[state][0][0] + self.actions[action][0]
        new_x_pos = self.states[state][0][1] + self.actions[action][1]
        # Is the future position an impossible one ?
        hitting_maze_walls =  (new_y_pos == -1) or (new_y_pos == self.maze.shape[0]) or \
                              (new_x_pos == -1) or (new_x_pos == self.maze.shape[1]) or \
                              (self.maze[new_y_pos][new_x_pos] == 1)
        # Based on the impossiblity check return the next state.

        # Get monster action
        if m_action:
            minotaur_action = m_action
        else:
            m_possibleActions = self.getMinotaur_actions(state)
            minotaur_action = np.random.choice(m_possibleActions)
        new_ym_pos = self.states[state][1][0] + self.actions[minotaur_action][0]
        new_xm_pos = self.states[state][1][1]  + self.actions[minotaur_action][1]

        if hitting_maze_walls:
            return self.map[((self.states[state][0][0], self.states[state][0][1]),(new_ym_pos,new_xm_pos))]   # Return current state/position
        else:
            return self.map[((new_y_pos, new_x_pos),(new_ym_pos,new_xm_pos))] #Return the next state at next step
        
    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions)
        transition_probabilities = np.zeros(dimensions)

        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        for s in range(self.n_states):
                for a in range(self.n_actions):
                    next_s = self.__move(s,a)
                    transition_probabilities[next_s, s, a] = 1 #TODO: Have to change what the probability is
        return transition_probabilities
    
    def __rewards(self, weights=None, random_rewards=None):
        rewards = np.zeros((self.n_states, self.n_actions))

        # If the rewards are not described by a weight matrix
        if weights is None:
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    next_s = self.__move(s,a)
                    get_eaten = False
                    m_possibleActions = self.getMinotaur_actions(s)
                    for m_action in m_possibleActions:
                        next_Ms = self.__move(s, a, m_action)
                        if self.states[next_Ms][0] == self.states[next_Ms][1]:
                            get_eaten = True
                    # Reward for hitting a wall, move() returns current state instead of new
                    if self.states[s][0] == self.states[next_s][0] and a != 0: #if player collision happened and we chose to move a!=0 (Stay = 0)
                        rewards[s,a] = self.REWARD_IMPOSSIBLE
                    # Reward for reaching the exit, current is returned because element in matrix was not 0
                    elif get_eaten and self.maze[self.states[s][0][0],self.states[s][0][1]] != 2:
                        rewards[s,a] = self.REWARD_MINOTAUR/len(m_possibleActions)
                    elif self.states[s][0] == self.states[next_s][0] and \
                        self.maze[self.states[next_s][0][0],self.states[next_s][0][1]] == 2:
                        rewards[s,a] = self.REWARD_GOAL
                    # Reward for taking a step to an empty cell that is not the exit
                    else:
                        rewards[s,a] = self.REWARD_STEP

                    # If there exists trapped cells with probability 0.5, Trapped cells have negative values in maze matrix
                    #TODO: This will possibly be modified to handle minotaur going through walls 
                    if random_rewards and self.maze[self.states[next_s][1],self.states[next_s][0]]<0:
                        x_pos, y_pos = self.states[next_s]
                        # With probability 0.5 the reward is
                        r1 = (1 + abs(self.maze[y_pos, x_pos])) * rewards[s,a] #n+1 accounts for n rounds skipped that we collect that reward
                        # With probability 0.5 the reward is
                        r2 = rewards[s,a]   # The reward of stepping inside but not getting trapped.
                        # The average reward
                        rewards[s,a] = 0.5*(r1 + r2)
        else:
            for s in range(self.n_states):
                 for a in range(self.n_actions):
                     next_s = self.__move(s,a)
                     i,j = self.states[next_s]
                     # Simply put the reward as the weights o the next state.
                     rewards[s,a] = weights[i][j]
        
        return rewards
    
    def simulate(self, start, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods);
            raise NameError(error)

        path = list()
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1]
            # Initialize current state and time
            t = 0
            s = self.map[start]
            # Add the starting position in the maze to the path
            path.append(start)
            while t < horizon-1:
                # Move to next state given the policy and the current state
                next_s = self.__move(s,policy[s,t])
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1
                s = next_s
                # When get eaten, stop immediately
                if self.states[s][0] == self.states[s][1]:
                    break
        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1
            gamma = 29/30
            s = self.map[start]
            # Add the starting position in the maze to the path
            path.append(start)
            # Move to next state given the policy and the current state
            next_s = self.__move(s,policy[s])
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s])
            arrive_goal = False
            # Loop while state is not the goal state
            while self.states[s][0] != self.states[s][1] and arrive_goal == False and random.random() < gamma:
                # Update state
                s = next_s
                # Move to next state given the policy and the current state
                next_s = self.__move(s,policy[s])
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                if self.states[s][0] == (6,5): 
                    arrive_goal = True
                # Update time and state for next iteration
                t +=1
        return path


def dynamic_programming(env, horizon):
    """ Implementation taken from Alessio Russo.
        Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path. These come from maze class
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities
    r         = env.rewards
    n_states  = env.n_states
    n_actions = env.n_actions
    T         = horizon

    # The variables involved in the dynamic programming backwards recursions
    V      = np.zeros((n_states, T+1))
    policy = np.zeros((n_states, T+1))
    Q      = np.zeros((n_states, n_actions))


    # Initialization
    Q            = np.copy(r)
    V[:, T]      = np.max(Q,1)
    policy[:, T] = np.argmax(Q,1)

    # The dynamic programming bakwards recursion
    for t in range(T-1,-1,-1):
        # Update the value function acccording to the bellman equation
        for s in range(n_states): # For each initial state
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s,a] = r[s,a] + np.dot(p[:,s,a],V[:,t+1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:,t] = np.max(Q,1) # Find best values and it gives a list, each element is for each state
        # The optimal action is the one that maximizes the Q function
        policy[:,t] = np.argmax(Q,1) # For each time step find the best action, gives a list
    return V, policy

def value_iteration(env, gamma, epsilon):
    """ Implementation taken from Alessio Russo.
        Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities
    r         = env.rewards
    n_states  = env.n_states
    n_actions = env.n_actions

    # Required variables and temporary ones for the VI to run
    V   = np.zeros(n_states)
    Q   = np.zeros((n_states, n_actions))
    BV  = np.zeros(n_states)
    # Iteration counter
    n   = 0
    # Tolerance error
    tol = (1 - gamma)* epsilon/gamma

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V)
    BV = np.max(Q, 1)

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 200:
        # Increment by one the numbers of iteration
        n += 1
        # Update the value function
        V = np.copy(BV)
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V)
        BV = np.max(Q, 1)
        # Show error
        #print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q,1)
    # Return the obtained policy
    return V, policy

def draw_maze(maze, player_pos, minotaur_pos):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

    # Give a color to each cell
    rows,cols    = maze.shape
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('The Maze')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    rows,cols    = maze.shape
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_maze,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed')
    # Modify the hight and width of the cells in the table
    grid.get_celld()[player_pos].set_facecolor(LIGHT_PURPLE)
    grid.get_celld()[player_pos].get_text().set_text('Player')
    grid.get_celld()[minotaur_pos].set_facecolor(BROWN)
    grid.get_celld()[minotaur_pos].get_text().set_text('Minotaur')
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)

def draw_policy(maze, env, minotaur_pos, policy, t):
    action_dict = {0: "S",    #stay
               1: "L",   #left (y_pos, x_pos) the inner list corresponds to one row in the maze
               2: "U", #up (we go up in matrix thus smaller index)
               3: "R",  #right
               4: "D"} 
    
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

    # Give a color to each cell
    rows,cols    = maze.shape
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))
    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('Policy map at t='+str(t))
    ax.set_xticks([])
    ax.set_yticks([])
    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_maze,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed')
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)
    
    for state_id, state in env.states.items():
        if state[1] == minotaur_pos:
            action_text = action_dict.get(policy[state_id,t])
            grid.get_celld()[state[0]].get_text().set_text(action_text)
            
            
    grid.get_celld()[minotaur_pos].set_facecolor(LIGHT_PURPLE)
    grid.get_celld()[minotaur_pos].get_text().set_text('Minotaur')
    

def animate_solution(maze, path):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Size of the maze
    rows,cols = maze.shape

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0,0),
                     edges='closed')

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)

    history = dict()
    # Update the color at each frame
    for i in range(len(path)):
        grid.get_celld()[(path[i][0])].set_facecolor(LIGHT_ORANGE)
        player_text = grid.get_celld()[(path[i][0])].get_text().get_text()
        grid.get_celld()[(path[i][0])].get_text().set_text(player_text + '\nPlayer' + str(i))

        grid.get_celld()[(path[i][1])].set_facecolor(LIGHT_PURPLE)
        monster_text = grid.get_celld()[(path[i][1])].get_text().get_text()
        grid.get_celld()[(path[i][1])].get_text().set_text(monster_text + '\nMonster' + str(i))
        grid.get_celld()[(path[i][1])].get_text().set_color('black')

        if i > 0:
            if path[i][0] == path[i][1]:
                grid.get_celld()[(path[i][0])].set_facecolor(LIGHT_RED)
                grid.get_celld()[(path[i][0])].get_text().set_text('Player is dead')
            elif maze[path[i][0]] == 2:
                grid.get_celld()[(path[i][0])].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(path[i][0])].get_text().set_text('Player is out')
            else:
                grid.get_celld()[(path[i-1][0])].set_facecolor(col_map[maze[path[i-1][0]]])
            grid.get_celld()[(path[i-1][1])].set_facecolor(col_map[maze[path[i-1][1]]])
            if (maze[path[i-1][1]] == 1):
                grid.get_celld()[(path[i-1][1])].get_text().set_color('white')

        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1)