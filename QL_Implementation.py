import Env_create as env
import numpy as np

# Number of APs
NUM_OF_AP = 9
# Number of Users
NUM_OF_USER = 10
# Number of Applications per User
NUM_OF_APP = 2
# Rate Requirement
R = [6, 3]  # Mbps
# Mean Packet Arrival Rate
MPAR = 0.1  # Mbps
# Learning rate alpha, discount factor beta, decay factor lambda, greedy policy factor epsilon
ALPHA = 0.8
BETA = 0.9
LAMBDA = 0.995
EPSILON = 0.5
# Number of states = (number of APs) ^ (number of apps)
NUM_OF_STATE = NUM_OF_AP^NUM_OF_APP
NUM_OF_ACTION = NUM_OF_AP^NUM_OF_APP

# PLOT DATA POINTS

# CREATE STATE MATRIXES
# State is a NUM_OF_USER*NUM_OF_APP matrix where state[user_index,app_index]=serving_ap_index
def initialize_state_matrix():
    state = np.matrix(np.zeros(shape=(NUM_OF_USER, NUM_OF_APP)))
    return state
def update_state(state,action):
    # check_drop    
    return state

# CREATE REWARD
# Return a reward array of each user
# state, action, achievable_rate là toàn bộ các state, action, achivalbe rate của user
# môi trường sẽ tính toán reward này và trả về cho các user nhận lấy reward tại
# vị trí cần tìm trong mảng reward
def reward(state, action, load, achievable_rate):
    reward = np.array(np.zeros(NUM_OF_USER))
    for k in range(NUM_OF_USER):
        c1 = 0
        c2 = 0
        for f in range(NUM_OF_APP):
            AP_request_index = state[k, f]
            achievable_rate_bk=achievable_rate[AP_request_index,k]
            drop=check_drop(action[k,f], load, achievable_rate_bk)[f]
            #if app f is not dropped then x_bkf=1    
            if ((not(drop)) and (achievable_rate_bk >= R[f])):
                c1 += achievable_rate_bk
            else:
                if((not(drop)) and (achievable_rate_bk < R[f])):
                    c2 += R[f]/achievable_rate_bk
                else:
                    c2+= R[f]/achievable_rate
                #Trong thuật toán này, hai trường hợp nhận hình phạt như nhau
        reward[k] = 0.8*c1-0.2*c2
    return reward


# CREATE MODEL
#Action is a array with size of NUM_OF_APP where action[f]=b means app f request AP b
#state is a array with size of NUM_OF_APP where state[f]=b means app f being served by AP b
def chose_action(state, Q_table):
    action=np.array(NUM_OF_APP)
    random_factor = np.random()
    EPSILON=EPSILON*LAMBDA
    if (random_factor < EPSILON):
        for f in range(NUM_OF_APP):
            action[f] = np.random.randint(0, NUM_OF_AP - 1)
        return action

    # Chose best action
    else:
        current_state_index=convert_to_index(state)
        max_Q_value = Q_table[current_state_index,0]
        expected_action_index = 0 
        for a in range(NUM_OF_ACTION):
            if(Q_table[current_state_index,a]>max_Q_value):
                max_Q_value = Q_table[current_state_index,a]
                expected_action_index = a
        action = convert_from_index(expected_action_index)
            
    return action


# Return a matrix of achievable rate between each user k and each AP b
def achievable_rate(h):
    r = np.matrix(np.zeros(shape=(NUM_OF_AP, NUM_OF_USER)))
    for b in range(NUM_OF_AP):
        for k in range(NUM_OF_USER):
            r[b, k] = env.r(h, b, k)
    return r

# Return a array of each AP's load
def AP_load(state, achievable_rate):
    load = np.array(np.zeros(NUM_OF_APP))
    for k in range(NUM_OF_USER):
        for f in range(NUM_OF_APP):
            load[state[k, f]] += MPAR/achievable_rate[state[k, f], k]
    return load

# Return a array of app dropped per user
# If Appilcation k requires a AP that will be overloaded if it serve app k then drop[k]=True
# action_of_user is a row in action matrix
# load is the array of each AP's load
# achievable_rate_bk is the value of achievable_rate[b,k]
def check_drop(action_of_user, load, achievable_rate_bk):
    drop = np.array(2)
    for k in range(NUM_OF_APP):
        load_for_serving = MPAR/achievable_rate_bk
        if (load[action_of_user[k]]+load_for_serving > 1):
            drop[k] = True
        else:
            drop[k] = False
    return drop

# Initialize Q
def initialize_Q():
    #shape=number of states * number of actions
    Q = np.matrix(np.zeros(shape=(NUM_OF_STATE, NUM_OF_ACTION)))
    return Q


def update_Q(state, action, reward,Q_table):
    # Find max Q value for state t+1
    next_action_row = Q_table[action] #state(t+1) = action(t)
    maxQ = next_action_row.max()
    Q_table[state,action] = Q_table[state,action] + ALPHA*(reward + BETA *maxQ - Q_table[state,action] )


# TRAINING
    # Read from old Q-tables
        
    # Train with new data
        # for frame in range(NUM_OF_FRAME):
            # user_positions = env.initialize_users_pos()
            # ap_positions = env.initialize_aps_pos()
            # h = env.initialize_users_h(user_positions, ap_positions)
            # r = achievable_rate(h)
            # state_of_all_user = initialize_state_matrix()
            
            # for k in range(NUM_OF_USER)
                # state = state_of_all_user[k]
                # action = chose_action(state)
                # load = AP_load(state_of_all_user, r)
                # reward = reward(state,action,load,r)
                # Q = update_Q(state, action, reward, Q_tables[k])
                # state = update_state(state, action)
                # state_of_all_user[k] = state

            # Write results to data files


# Get index of user's state in Q table
# state là 1 mảng gồm num_of_applications phần tử, mỗi phần tử chứa giá trị index của access point
def convert_to_index(state):
    index = 0
    for i in range(len(state)):
        index += pow(NUM_OF_AP, i) * state[i]
    return index

def convert_from_index(index):
    state = []
    k = 0
    while (k < NUM_OF_APP):
        state.append(index % NUM_OF_AP)
        index = int(index / NUM_OF_AP)
        k += 1
    return state

# state = list(map(int, input("Nhap list: ").split(" ")))
# print(state)
# i = int(input())
# state = convert_from_index(i)
# print(state)
