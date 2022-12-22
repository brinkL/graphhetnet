## Runs the second part of the MoPredNet algorithm to generate the external memory that stores decoder weights for every meta-train action

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
from args import argument_parser

args = argument_parser()
file_time = str(datetime.datetime.now()).replace(" ","_")
ft = "file_time"
if args.name is None:
    name = file_time
else:
    name = args.name+"_"+file_time

args.name = name
edges = np.load(args.edges_dir)

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
        
def metaPreGen(fewGen,meta = True, wrapper=False, test=False):
    
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

sample_joints = "sub"

train_gen = genMotionTask(train_actions,12,
                    one_hot    = False,
                    angles     = 54,
                    source_len = 50, 
                    target_len = 10,
                    num_labels = 11,
                    full_query = True,
                    replace = True,
                    data_dir = args.data_dir)
    
pre = metaPreGen(train_gen,meta=False)

model = MoPredNet(target_length=10,drop=0., def_conv=False)
loss_object = tf.keras.losses.MeanSquaredError()
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.enc_a.trainable = False
model.enc_m.trainable = False
model.dec.layers[0].trainable = False


model.build(input_shape=(12,50,54,1))

model.enc_a.trainable = False
model.enc_m.trainable = False
model.dec.layers[0].trainable = False

SAVED_WEIGHTS = "" # Path to the saved weights after pretraining MoPredNet

model.load_weights(SAVED_WEIGHTS)
model.compile(optimizer = opt,
              loss = loss_object)

memory = []

v = []
for i in range(100):
    x,y = next(pre)
    v.append(model.enc_a(x,training=False))
v = np.array(v)
v = np.mean(v,axis=(0,1))

shapes = [i.shape for i in model.dec.trainable_weights]
print(shapes)
theta = weights2vec(model.dec.trainable_weights)[0]
memory.append([v,theta])


init_weights = model.get_weights()

for action in train_actions:
    print(f"Generating decoder weights for action {action}")
    for var in opt.variables():
        var.assign(tf.zeros_like(var))
    
    model.set_weights(init_weights)
    
    train_gen = genMotionTask([action],12,
                        one_hot    = False,
                        angles     = 54,
                        source_len = 50, 
                        target_len = 10,
                        num_labels = 11,
                        full_query = True,
                        replace = False,
                        data_dir = args.data_dir)
    
    if sample_joints == "full":
        pre      = metaPreGen(train_gen,meta=False)
        padded = False

    elif sample_joints == "sub":
        train_gen = subgraphWrapper(train_gen, edges, pad_data=True)

        pre      = metaPreGen(train_gen,meta=False, wrapper=True)
        padded = True

    
    hist = model.fit(x=pre,steps_per_epoch=50,epochs=100)
    
    v = []
    for i in range(100):
        x,y = next(pre)
        v.append(model.enc_a(x))
    v = np.array(v)
    v = np.mean(v,axis=(0,1))
    
    theta = weights2vec(model.dec.trainable_weights)[0]
    memory.append([v,theta])

os.system(f"mkdir -p {os.path.join(args.save_dir,args.name)}")
np.save(f"{os.path.join(args.save_dir,args.name)}/memory.npy",memory)