import tensorflow as tf
import gc
import sys
import os
import numpy as np
import datetime
import ast
gc.enable()
tf.__version__


from data_loading.convertdata import define_actions, evaluate_euler
#from convertdata import read_all_data, get_batch_srnn, define_actions, get_batch, get_srnn_gts
#import data_utils
from data_loading.data_gen import genMotionTask, subgraphWrapper

from MoHetNet.graph_het import TimeHetNet as GraphHetNet
from MoHetNet.time_het   import TimeHetNet

from args            import argument_parser
from save import saveMe



#parse arguments --------------------------------------------------------------------

args = argument_parser()
file_time = str(datetime.datetime.now()).replace(" ","_")
ft = "file_time"
if args.name is None:
    name = file_time
else:
    name = args.name+"_"+file_time

args.name = name

print("########## argument sheet ########################################")
for arg in vars(args):
    print (f"#{arg:>15}  :  {str(getattr(args, arg))} ")
print("##################################################################")
#------------------------------------------------------------------------------------
#Load our dataa generators

one_hot = False
angles  = 54
source_len = 50
target_len = 10
if one_hot:
    full_angles = angles+15
else:
    full_angles = angles

#split the actions for each set (training and testing)

train_actions = args.train_actions.split(",")
test_actions = ['discussion', 'eating', 'smoking', 'walking']

train_gen = genMotionTask(train_actions,5,
                    one_hot    = False,
                    angles     = 54,
                    source_len = 50, 
                    target_len = 10,
                    num_labels = 11,
                    full_query = True,
                    data_dir=args.data_dir)

test_gen = genMotionTask(test_actions,5,
                    one_hot    = False,
                    angles     = 54,
                    source_len = 50, 
                    target_len = 10,
                    num_labels = 5,
                    training   = False,
                    full_query = True,
                    data_dir=args.data_dir)


# load our base graph
edges = np.load(args.edges_dir)

#--------------------------------------------------------------------------
def preGen(gen):

    e = np.tile(np.expand_dims(edges,0),[11,1,1])

    while True: 
        sup_x,sup_y,que_x,que_y,action = next(gen)
        yield (que_x,sup_x,sup_y,e),que_y
        

# here we select the sampling method and encapsulate our generators
if args.sample_joints == "full": # tasks do not have the graph re-sampled. These are the tasks from our baselines.
    myTrainGen = preGen(train_gen)

elif args.sample_joints == "sub": # tasks have their graph resampled by sub-graphs
    myTrainGen = subgraphWrapper(train_gen,edges, max_nodes = args.max_joints,mode=args.sample_mode)
else:
    raise ValueError("Choose one of full/sub for sample_joints")
    

# a small hack to make the generators work with tf.dataset
def make_gen_callable(_gen):
        def gen():
            while True:
                (a,b,c,d),e = next(_gen)
                yield np.float32(a),np.float32(b),np.float32(c),d,np.float32(e)
        return gen
dataset = tf.data.Dataset.from_generator(make_gen_callable(myTrainGen),output_types=(tf.float32,tf.float32,tf.float32,tf.int64,tf.float32))
#myTestGen  = preGen(test_gen)

#we use this to run our experiment across multiple gpus
mirrored_strategy = tf.distribute.MirroredStrategy()

with mirrored_strategy.scope():
    #convert our dataset generator to a distributed generator
    dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
    
    #build our models inside the parallel scope
    if args.timehet_style == "classic":
        model = TimeHetNet(block  = ["conv","conv","gru","gru"], time=50,target_time=10, dims_inf = ast.literal_eval(args.dims_inf), dims_pred = ast.literal_eval(args.dims_pred))
    else:
        model = GraphHetNet(block = ["conv","conv","gru","gru"], time=50,target_time=10, dims_inf = ast.literal_eval(args.dims_inf), dims_pred = ast.literal_eval(args.dims_pred), joint_axis=2, dropout = args.dropout, l2_rate = args.l2_rate)

    #define losses and optmizers
    loss_object = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')


    # defining our train steps 
    def train_step(x):
        
        total_result=0.0
        
        #steps to be replicated across GPUs
        def replica_fn(x):
            with tf.GradientTape() as tape:
                a,b,c,d,y=x
                x = (a,b,c,d)
                predictions = model(x)
                loss = loss_object(y, predictions)
                
                #check if we are using regularization
                if args.l2_rate != 0:
                    reg_loss = loss + model.losses
                else:
                    reg_loss = loss
            gradients = tape.gradient(reg_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss
        
        #agregate the results from each device
        per_replica_result = mirrored_strategy.run(replica_fn, args=(x,))
        total_result += mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM,
                                         per_replica_result, axis=None)
        train_loss(total_result)

    #run our train loops
    EPOCHS = args.num_epochs
    hist = []
    da=iter(dist_dataset)
    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        test_loss.reset_states()

        for steps in range(args.steps):
            x = next(da)

            train_step(x)
        
        # print results, keep in mind that the paralell version agregates by sum. if you wish to have averaged,
        # update the loss_object and the mirrored_strategy.reduce to do so.
        
        hst=train_loss.result()
        hist.append(hst)
        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
        )
        gc.collect() #garbage collection

#save our model weights and result experiments
os.system(f"mkdir -p {os.path.join(args.save_dir,args.name)}")
np.save(f"{os.path.join(args.save_dir,args.name)}/lr_history.npy",hist)
print("saved history!")
model.save_weights(os.path.join(args.save_dir,args.name,args.name))
print("saved weights!")
saveMe(args,"trainMoHet",{"history":hist})
print("saved report!")
