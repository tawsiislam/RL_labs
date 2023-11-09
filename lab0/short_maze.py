import numpy as np

class Maze:

    actions = {0: "stay",
               1: "left",
               2: "up",
               3: "right",
               4: "down"}
    reward_mat = [-100,-1,0] #impossible state, step, goal
    
    def __init__(self, maze_mat, weights=None, random_rewards=False):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze_mat # Matrix of the maze
        self.actions                  = self.__create_actions() #Callable function to get action
        self.states, self.map         = self.__create_states() #Callable function to get state
        self.n_actions                = len(self.actions)
        self.n_states                 = len(self.states)
        self.transition_probabilities = self.__transitions()
        self.rewards                  = self.__rewards(weights=weights,
                                                random_rewards=random_rewards)
        
    def __create_actions(self):
        actions = {0: (0,0),    #stay
               1: (-1,0),   #left (x_pos, y_pos)
               2: (0,-1), #up (we go up in matrix thus smaller index)
               3: (1,0),  #right
               4: (0,1)}   #down
        return actions
    
    def __create_states(self):
        states = dict()
        map = dict()
        s = 0
        for y_pos in range(self.maze.shape[0]):
            for x_pos in range(self.maze.shape[1]):
                if self.maze[y_pos][x_pos] != 1:
                    states[s] = (x_pos,y_pos)
                    map[(x_pos,y_pos)] = s
                    s+=1
        return states, map
    
    def __move(self, state, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        # Compute the future position given current (state, action)
        new_x_pos = self.states[state][0] + self.actions[action][0]
        new_y_pos = self.states[state][1] + self.actions[action][1]
        # Is the future position an impossible one ?
        hitting_maze_walls =  (new_y_pos == -1) or (new_y_pos == self.maze.shape[0]) or \
                              (new_x_pos == -1) or (new_x_pos == self.maze.shape[1]) or \
                              (self.maze[new_y_pos][new_x_pos] == 1)
        # Based on the impossiblity check return the next state.
        if hitting_maze_walls:
            return state    # Return current state/position
        else:
            return self.map[(new_x_pos, new_y_pos)] #Return the next state at next step
        
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
                transition_probabilities[next_s, s, a] = 1
        return transition_probabilities
    
    def __rewards(self, weights=None, random_rewards=None):
        rewards = np.zeros((self.n_states, self.n_actions))

        # If the rewards are not described by a weight matrix
        if weights is None:
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    next_s = self.__move(s,a)
                    # Reward for hitting a wall, move() returns current state instead of new
                    if s == next_s and a != 0: #if collision happened and we chose to move a!=0
                        rewards[s,a] = self.reward_mat[0]
                    # Reward for reaching the exit, current is returned because element in matrix was not 0
                    elif s == next_s and self.maze[self.states[next_s][1],self.states[next_s][0]] == 2:
                        rewards[s,a] = self.reward_mat[2]
                    # Reward for taking a step to an empty cell that is not the exit
                    else:
                        rewards[s,a] = self.reward_mat[1]

                    # If there exists trapped cells with probability 0.5, Trapped cells have negative values in maze matrix
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
        path = []
        
        if method == "DynPro":
            # Set the horizon which is the length of the policy
            # Append starting state in path list
            # Initialize current state with start and your time to check horizon reached
            
            # For each time step
                # Find the next state using move function that follows your policy
                # Policy should have been calculated from your dynamic program
                # After finding the next state your agent will go to after following policy,
                # append that to your path list
                # Update time and current state
            pass
        elif method == "ValIter":
            # Initilize current state, time and get next state
            # Append starting state and next state in path list
            
            # While current state is not equal to next state
                # Update current state and next state with move()
                # Append the next state into path list
            pass
        return path
    
def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
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
    """ Solves the shortest path problem using value iteration
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

def main():
    maze_mat = np.array([
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 2, 0]])
    start_pos = (0,0)
    


if "__name__" == "__main__":
    main()