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

from MoHetNet.graph_het  import TimeHetNet as GraphHetNet
from MoHetNet.time_het   import TimeHetNet

from args            import argument_parser
from save import saveMe

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
    #print(f"#{ft:>15}  :  {file_time} ")
print("##################################################################")

one_hot = False
angles  = 54
source_len = 50
target_len = 10
if one_hot:
    full_angles = angles+15
else:
    full_angles = angles

#split the actions for each set (training and testing)

test_actions = ['discussion', 'eating', 'smoking', 'walking']
train_actions = args.train_actions.split(",")
train_gen = genMotionTask(train_actions,5,
                    one_hot    = False,
                    angles     = 54,
                    source_len = 50, 
                    target_len = 10,
                    num_labels = len(train_actions),
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

#--------------------------------------------------------------------------

#load our models 

if args.timehet_style == "classic":
    model = TimeHetNet(block  = ["conv","conv","gru","gru"], time=50,target_time=10, dims_inf = ast.literal_eval(args.dims_inf), dims_pred = ast.literal_eval(args.dims_pred))
else:
    model = GraphHetNet(block = ["conv","conv","gru","gru"], time=50,target_time=10, dims_inf = ast.literal_eval(args.dims_inf), dims_pred = ast.literal_eval(args.dims_pred), joint_axis=2, dropout = args.dropout, l2_rate = args.l2_rate)

#--------------------------------------------------------------------------
#define our losses

loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')

#define our train loop
# we create two of them since with the full tasks we can actually use the tf.function decorator
if args.sample_joints == "full":
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss = loss_object(y, predictions)

            if args.l2_rate != 0:
                reg_loss = loss + model.losses
            else:
                reg_loss = loss

        gradients = tape.gradient(reg_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
else:

    def train_step(x, y):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = model(x)
            loss = loss_object(y, predictions)
            if args.l2_rate != 0:
                reg_loss = loss + model.losses
            else:
                reg_loss = loss
        gradients = tape.gradient(reg_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)

@tf.function
def test_step(x, y):
    predictions = model(x)
    t_loss = loss_object(y, predictions)

    test_loss(t_loss)

#define our epochs loop --------------------------------------------------------
EPOCHS = args.num_epochs
hist = []
for epoch in range(EPOCHS):
    
    # reset the losses
    train_loss.reset_states()
    test_loss.reset_states()

    # loop
    for steps in range(args.steps):
        x,y = next(myTrainGen)
        train_step(x,y)

    hst=train_loss.result()
    hist.append(hst)
    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
    )
    gc.collect()  #collect the garbage

#save results
os.system(f"mkdir -p {os.path.join(args.save_dir,args.name)}")
np.save(f"{os.path.join(args.save_dir,args.name)}/lr_history.npy",hist)
model.save_weights(os.path.join(args.save_dir,args.name,args.name))
saveMe(args,"trainMoHet",{"history":hist})
