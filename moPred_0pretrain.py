### Pretrains the two encodes and the decoder of MoPredNet as shown in their algorithm

import tensorflow as tf
import gc
import sys
import os
import numpy as np
gc.enable()
tf.__version__
print(os.getcwd())
from data_loading.data_gen import genMotionTask, subgraphWrapper
from MoPredNet.moPredNet import MoPredNet
from MoPredNet.utils import weights2vec, vec2weights
from data_loading.convertdata import define_actions

one_hot = False
angles  = 54
source_len = 50
target_len = 10
if one_hot:
    full_angles = angles+15
else:
    full_angles = angles

import datetime
from args            import argument_parser

args = argument_parser()
file_time = str(datetime.datetime.now()).replace(" ","_")
ft = "file_time"
if args.name is None:
    name = file_time
else:
    name = args.name+"_"+file_time

args.name = name
edges = np.load(args.edges_dir)

all_actions = define_actions("all")

train_actions = [ "directions",
                 "greeting", 
                 "phoning", 
                 "posing", 
                 "purchases", 
                 "sitting", 
                 "sittingdown", 
                 "takingphoto", 
                 "waiting", 
                 "walkingdog", 
                 "walkingtogether"]

test_actions = ['discussion', 'eating', 'smoking', 'walking']

def preGen(fewGen):
    
    while True:
        
        sup_x,sup_y,que_x,que_y,action = next(fewGen)
        sup_x = np.transpose(sup_x,[1,2,3,0]).astype(np.float32)
        sup_y = sup_y[0].astype(np.float32)
        
        yield sup_x,sup_y

def wrapWrapper(wrapGen):

    while True:
        (que_x,sup_x,sup_y,e,padding),que_y = next(wrapGen)
        
def metaPreGen(fewGen,meta = True,test=False, wrapper=False):
    
    while True:
        if wrapper:
            if test:
                (que_x,sup_x,sup_y,e,que_euler,nodes_contained,padding),que_y = next(fewGen)
            else:
                (que_x,sup_x,sup_y,e,padding),que_y = next(fewGen)

        else:
            if test:
                sup_x, sup_y,que_x,que_y, euler, action = next(fewGen)
            else:
                sup_x,sup_y,que_x,que_y,action = next(fewGen)

        sup_x = np.expand_dims(sup_x,axis=-1).astype(np.float32)
        sup_y = sup_y.astype(np.float32)
        
        que_x = np.expand_dims(que_x,axis=-1).astype(np.float32)
        que_y = que_y.astype(np.float32)
        
        if not meta:
            sup_x = np.reshape(sup_x,[-1,sup_x.shape[2],sup_x.shape[3],sup_x.shape[4]])
            
            sup_y = np.reshape(sup_y,[-1,sup_y.shape[2],sup_y.shape[3]])
            
            yield sup_x,sup_y
            
        else:
            yield sup_x,sup_y,que_x,que_y,action


train_gen = genMotionTask(train_actions,12,
                    one_hot    = False,
                    angles     = 54,
                    source_len = 50, 
                    target_len = 10,
                    num_labels = 11,
                    full_query = True,
                    replace = True,
                    data_dir=args.data_dir)

test_gen = genMotionTask(test_actions,5,
                    one_hot    = False,
                    angles     = 54,
                    source_len = 50, 
                    target_len = 10,
                    num_labels = 5,
                    training   = False,
                    full_query = True,
                data_dir = args.data_dir)


if args.sample_joints == "full":
    action_stats = next(test_gen)
    pre      = metaPreGen(train_gen,meta=False)
    pre_test = metaPreGen(test_gen,meta=False,test=True)
    padded = False

elif args.sample_joints == "sub":
    train_gen = subgraphWrapper(train_gen, edges, pad_data=True)

    pre      = metaPreGen(train_gen,meta=False, wrapper=True)
    pre_test = None
    padded = True

else:
    raise ValueError("Choose one of full/sub for sample_joints")
    


model = MoPredNet(target_length=10,drop=args.dropout, def_conv=args.mo_encoder)
_ = model(np.random.random([12,50,54,1]).astype(np.float32))

loss_object = tf.keras.losses.MeanSquaredError()
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer = opt,
              loss = loss_object)

try:
    hist = model.fit(x=pre,steps_per_epoch=50,epochs=500, validation_data=pre_test, validation_steps=1)
except:
    print("Interrupt saving")
    os.system(f"mkdir -p {os.path.join(args.save_dir,args.name)}")
    model.save_weights(os.path.join(args.save_dir,args.name,args.name))

    raise KeyboardInterrupt

os.system(f"mkdir -p {os.path.join(args.save_dir,args.name)}")
np.save(f"{os.path.join(args.save_dir,args.name)}/lr_history.npy",hist)
model.save_weights(os.path.join(args.save_dir,args.name,args.name))
saveMe(args,"trainMoPred",{"history":hist})


