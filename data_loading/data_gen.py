import gc
import numpy as np
gc.enable()


from data_loading.convertdata import read_all_data, get_batch, get_test_data
import data_loading.data_utils


def genMotionTask(all_actions,shots,
                    one_hot    = False,
                    angles     = 54,
                    source_len = 50, 
                    target_len = 10,
                    num_labels = 15,
                    full_query = True,
                    training = True,
                    replace = True,
                    data_dir="./data/human36/h3.6m/dataset"):

    if one_hot:
        full_angles = angles+num_labels
    else:
        full_angles = angles
    
    
    full_data = {}

    action_stats = {}
    for action in all_actions:
        print("Loading data for action",action)

        out = read_all_data([action], source_len, target_len, data_dir,one_hot)
        train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use  = out
        if not training:
            action_stats[action] = (data_mean, data_std, dim_to_ignore)
        
        if not full_query:
            train_data = train_set | test_set

            if training:
                full_data[action] = get_batch(train_data, [action], shots*2, source_len, target_len, full_angles)

            else:
                full_data[action] = get_test_data([action], train_data, data_mean,
                        data_std, dim_to_ignore,one_hot, source_len, target_len, full_angles,batch_size=shots*2) # Not Working: TODO Unhardcode subject 5

        else:
            if training:
                sup_gen = get_batch(train_set, [action], shots, source_len, target_len, full_angles,replace=replace)
                que_gen = get_batch(test_set,  [action], len(list(test_set.keys())), source_len, target_len, full_angles)

            else:
                sup_gen = get_batch(train_set, [action], shots, source_len, target_len, full_angles)
                
                que_gen = get_test_data([action], test_set, data_mean,
                        data_std, dim_to_ignore,one_hot, source_len, target_len, full_angles,batch_size=len(list(test_set.keys())))

            full_data[action] = (sup_gen,que_gen)
    if not training:
        yield action_stats
    
    reuse_que = False
    while True:
        m_sup_x, m_sup_y, m_que_x, m_que_y = [], [], [], []

        if not training and not reuse_que:
            m_que_yeuler = []


        for gen in full_data.values():

            if full_query:

                sup_gen,que_gen = gen  #x, y, y_euler

                x, y = next(sup_gen)
                m_sup_x.append(x)
                m_sup_y.append(y)

                try:
                    out = next(que_gen)
                
                    if training:
                        que_x, que_y = out
                    else:
                        que_x, que_y, que_y_euler = out
                        m_que_yeuler.append(que_y_euler)

                    m_que_x.append(que_x)
                    m_que_y.append(que_y)

                except StopIteration:
                    reuse_que = True
                    continue

            else:

                out = next(gen)

                if training:
                    x, y = out
                else:
                    raise NotImplementedError("Not yet mate.")
                    x, y, y_euler = out
                    m_sup_yeuler.append(y_euler[:shots])
                    m_que_yeuler.append(y_euler[shots:])

                m_sup_x.append(x[:shots,:,:])
                m_sup_y.append(y[:shots,:,:])
                m_que_x.append(x[shots:,:,:])
                m_que_y.append(y[shots:,:,:])

        if not training:
            if reuse_que:
                m_que_x, m_que_y, m_que_yeuler = m_que_x_old, m_que_y_old, m_que_yeuler_old
            else:
                m_que_x_old, m_que_y_old, m_que_yeuler_old = m_que_x.copy(), m_que_y.copy(), m_que_yeuler.copy()

        if training:
            yield np.array(m_sup_x), np.array(m_sup_y), np.array(m_que_x), np.array(m_que_y), np.array(all_actions)
        else:
            yield np.array(m_sup_x), np.array(m_sup_y), np.array(m_que_x), np.array(m_que_y), np.array(m_que_yeuler), np.array(all_actions)


# Map from [0,53] angles to actual joints
joints = np.array([[0, 1, 2], #torso
[3, 4, 5],          #torso
[6, 7, 8],          #right_leg
[None,None,9],      #right_leg
[10, 11, 12],       #right_leg
[None,None,13],     #right_leg
[None,None,None],
[14, 15, 16],       #left_leg
[None,None,17],     #left_leg
[18, 19, 20],       #left_leg
[None,None,21],     #left_leg
[None,None,None], 
[22, 23, 24],       #torso
[25, 26, 27],       #torso
[28, 29, 30],       #torso
[31, 32, 33],       #torso
[None,None,None],
[34, 35, 36],       #left_arm
[37, 38, 39],       #left_arm
[None,None,40],     #left_arm
[41, 42, 43],       #left_arm
[None,None,None],
[None,None,None],
[None,None,None],
[None,None,None],
[44, 45, 46],       #right_arm
[47, 48, 49],       #right_arm
[None,None,50],     #right_arm
[51, 52, 53],       #right_arm
[None,None,None],
[None,None,None],
[None,None,None],
[None,None,None]])


train_sensors = [0, 1, 2, 3, 4, 5, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53] 
test_sensors  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]

def subgraphWrapper(gen, 
                    edges,
                    pad_data = False,
                    test=False,
                    max_nodes = 54,
                    mode=None):

    if test:
        yield next(gen)
    if max_nodes == -1:
    	max_nodes = 54

    while True:
        
        # Selecting random root node for new graph
        if mode is None:
            random_node = np.random.choice(range(len(np.unique(edges[0,:]))))
        elif mode == "train":
            random_node = np.random.choice(train_sensors)
        elif mode == "test":
            random_node = np.random.choice(test_sensors)
        else:
            raise ValueError("Mode needs to be one of [None,train,test]")

        root_joint = np.where(joints == random_node)

        nodes_contained = [random_node]
        not_contained = []

        def getNeighbors(node):

            neighbor_nodes = [edges[1,i] for i in range(len(edges[0])) if edges[0,i]==node]

            if mode =="train":
                neighbor_nodes = [i for i in neighbor_nodes if i in train_sensors]
            elif mode =="test":
                neighbor_nodes = [i for i in neighbor_nodes if i in test_sensors]

            return neighbor_nodes

        neighbors = getNeighbors(random_node)

        chosen_nodes = np.random.choice(neighbors,np.random.randint(1,len(neighbors)+1),replace=False)
        
        # Adding random subset of neighbors to new graph while removing the rest
        nodes_contained += list(chosen_nodes)
        not_contained   += list(np.setdiff1d(neighbors,chosen_nodes))

        visited = [random_node]
        toVisit = np.setdiff1d(nodes_contained,visited)

        # Iterate over every node in the graph that has not been visited and repeat above procedure
        while len(nodes_contained)<max_nodes and len(toVisit) != 0:

            for node in toVisit:

                visited.append(node)

                neighbors = getNeighbors(node)
                neighbors = list(np.setdiff1d(neighbors,nodes_contained))
                neighbors = list(np.setdiff1d(neighbors,not_contained))

                if len(neighbors) == 0:
                    continue

                chosen_nodes = np.random.choice(neighbors,np.random.randint(1,len(neighbors)+1),replace=False)
                nodes_contained += list(chosen_nodes)
                not_contained   += list(np.setdiff1d(neighbors,chosen_nodes))


            toVisit = np.setdiff1d(nodes_contained,visited)
            
        # Compute subset of edges from the full graph for the new graph
        new_edges = [edges[:,i] for i in range(len(edges[0])) if edges[0,i] in nodes_contained and edges[1,i] in nodes_contained]
        
        # Reindex edges to start with node 0 to num_nodes-1
        reidx_edges = []
        for e in new_edges:

            reidx_edges.append([np.where(np.array(nodes_contained)==e[0])[0][0],np.where(np.array(nodes_contained)==e[1])[0][0]])
        reidx_edges = np.transpose(reidx_edges)
        
        # Get full meta batch
        if test: 
            sup_x,sup_y, que_x,que_y, que_euler, action = next(gen)
        else:
            sup_x,sup_y,que_x,que_y,action = next(gen)

        sup_x = sup_x[:,:,:,nodes_contained]
        sup_y = sup_y[:,:,:,nodes_contained]
        que_x = que_x[:,:,:,nodes_contained]
        que_y = que_y[:,:,:,nodes_contained]

        
        # Slice the subset of nodes in the new graph
        if pad_data:
            padding = len(nodes_contained)

            sup_x = np.concatenate([sup_x, np.zeros(list(sup_x.shape)[:-1]+[54-padding])],axis=-1)
            sup_y = np.concatenate([sup_y, np.zeros(list(sup_y.shape)[:-1]+[54-padding])],axis=-1)
            que_x = np.concatenate([que_x, np.zeros(list(que_x.shape)[:-1]+[54-padding])],axis=-1)
            que_y = np.concatenate([que_y, np.zeros(list(que_y.shape)[:-1]+[54-padding])],axis=-1)
        
        e = np.tile(np.expand_dims(reidx_edges,0),[11,1,1])

        if test:
            if pad_data:
                yield (que_x,sup_x,sup_y,e,que_euler,nodes_contained,padding),que_y
            else:
                yield (que_x,sup_x,sup_y,e,que_euler,nodes_contained),que_y

        else:
            if pad_data:
                yield (que_x,sup_x,sup_y,e,padding),que_y
            else:
                yield (que_x,sup_x,sup_y,e),que_y          

# Compute real joint graph (not needed; remove before submission)
def realGraph(edges):
    
    real_edges = []
    for i in range(len(joints)):
        for j in range(len(joints)):
            
            if i==j:
                continue
            
            for e in np.transpose(edges):
                
                if e[0] in joints[i] and e[1] in joints[j]:
                    real_edges.append([i,j])
                    break
                    
                if e[1] in joints[i] and e[0] in joints[j]:
                    real_edges.append([i,j])
                    break
    return real_edges
