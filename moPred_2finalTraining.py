import tensorflow as tf
import gc
import sys
import os
import numpy as np
import time
gc.enable()
tf.__version__

from data_loading.convertdata import define_actions, evaluate_euler
from data_loading.data_gen import genMotionTask, subgraphWrapper
from MoPredNet.moPredNet import MoPredNet
from MoPredNet.models import FCNet
from MoPredNet.utils import weights2vec, vec2weights
one_hot = False
angles  = 54
source_len = 50
target_len = 10
if one_hot:
    full_angles = angles+15
else:
    full_angles = angles

full = False

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

argmax = True
sample_joints = "sub"

PATH_TO_MEMORY = "" # Path to the external memory of decoder weights saved in .npy
memory = np.load(PATH_TO_MEMORY, allow_pickle=True)

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



def preGen(fewGen,wrapper=False):
    while True:
        
        if wrapper:
            (que_x,sup_x,sup_y,e,padding),que_y = next(fewGen)
        else:
            sup_x,sup_y,que_x,que_y,action = next(fewGen)

        sup_x = np.expand_dims(sup_x,axis=-1).astype(np.float32)
        sup_y = sup_y.astype(np.float32)
        
        que_x = np.expand_dims(que_x,axis=-1).astype(np.float32)
        que_y = que_y.astype(np.float32)

        yield sup_x,sup_y,que_x,que_y


opt_dec = tf.keras.optimizers.Adam(learning_rate=0.0001)
opt_pgm = tf.keras.optimizers.Adam(learning_rate=0.00001)

pg_model = FCNet(layer_sizes=[512,72630],activation="relu")

v_m = np.array([v_m[0] for v_m in memory])
w_m = np.array([w_m[1] for w_m in memory])

@tf.function
def execute(sup_x,model,pgm, full=False):

    # Generate embedding for task
    v_novel = model.enc_a(sup_x)
    v_novel = tf.reduce_mean(v_novel,0)
    
    # Generate decoder weights via PGM
    normalize_vm = tf.nn.l2_normalize(v_m,1)        
    normalize_vn = tf.nn.l2_normalize(tf.tile(tf.expand_dims(v_novel,0),[12,1]),1)
    h_probs = tf.nn.softmax(tf.reduce_sum(tf.multiply(normalize_vm,normalize_vn),axis=-1))

    if argmax:
        h = tf.slice(w_m,[tf.argmax(h_probs),0],[1,-1])[0]
    else:
        h = tf.matmul(tf.expand_dims(h_probs,0),w_m)[0]

    w_pred = pgm(tf.expand_dims(v_novel,0))[0]

    if full:
        w_pred = tf.concat([tf.zeros([tf.shape(h)[0]-tf.shape(w_pred)[0]]),w_pred],0)
    
    theta_gen = h + w_pred
    
    theta_set = vec2weights(tf.expand_dims(theta_gen,0),shapes)

    # Compute forecaste
    if full:
        _ = model.dec.trainable_variables[0].assign(theta_set[0])
        _ = model.dec.trainable_variables[1].assign(theta_set[1])
        _ = model.dec.trainable_variables[2].assign(theta_set[2])
        _ = model.dec.trainable_variables[3].assign(theta_set[3])
        _ = model.dec.trainable_variables[4].assign(theta_set[4])
        _ = model.dec.trainable_variables[5].assign(theta_set[5])
    
    else:  
        _ = model.dec.trainable_variables[0].assign(theta_set[0])
        _ = model.dec.trainable_variables[1].assign(theta_set[1])
        _ = model.dec.trainable_variables[2].assign(theta_set[2])
        _ = model.dec.trainable_variables[3].assign(theta_set[3])
        
    pred = model(sup_x)
    
    return pred, tf.expand_dims(v_novel,0), h

@tf.function
def train(sup_x,sup_y,model,pgm,full=False):
    
    with tf.GradientTape() as tape:
        
        predictions,v_nov,h = execute(sup_x,model,pgm, full=full)
        loss = loss_object(sup_y, predictions)


    gradients = tape.gradient(loss, model.dec.trainable_variables)
    gradients = [tf.clip_by_norm(g, 2) for g in gradients]

    opt_dec.apply_gradients(zip(gradients, model.dec.trainable_variables))

    theta_new = weights2vec(model.dec.trainable_variables)
        
    with tf.GradientTape() as tape2:    
        
        w_pred = pg_model(v_nov)
        if full:
            w_pred = tf.concat([tf.zeros([tf.shape(h)[0]-tf.shape(w_pred)[0]]),w_pred],0)
        w_pred = w_pred + h

        theta_loss = loss_object(w_pred , theta_new)
        
    theta_gradients = tape2.gradient(theta_loss, pgm.trainable_variables)
    opt_pgm.apply_gradients(zip(theta_gradients, pgm.trainable_variables))
        
    return loss, theta_loss
edges = np.load(args.edges_dir)
train_gen = genMotionTask(train_actions,5,
                    one_hot    = False,
                    angles     = 54,
                    source_len = 50, 
                    target_len = 10,
                    num_labels = 11,
                    full_query = True,
                    data_dir = args.data_dir)



model = MoPredNet(target_length=10,drop=0., def_conv=False)
loss_object = tf.keras.losses.MeanSquaredError()

# sub
SAVED_WEIGHTS = "" # Path to the saved weights after pretraining MoPredNet

model.load_weights(SAVED_WEIGHTS)

model.enc_a.trainable = False
model.enc_m.trainable = False
model.dec.layers[0].trainable = False


model.build(input_shape=(12,50,54,1))



model.compile(loss = loss_object)
pg_model.compile(loss = loss_object)
shapes = [i.shape for i in model.dec.trainable_weights]
print("shapes",shapes)


if sample_joints == "full":
    pre_gen = preGen(train_gen)
    padded = False

elif sample_joints == "sub":
    pre_gen = subgraphWrapper(train_gen, edges, pad_data=True)
    pre_gen = preGen(pre_gen,wrapper=True)
    padded = True

sup_x,sup_y,que_x,que_y = next(pre_gen)

hist = []
times = []
for i in range(1000):

    sup_x,sup_y,que_x,que_y = next(pre_gen)

    l1s,l2s = [],[]
    start = time.time()
    for mb in range(11):
        
        l1,l2 = train(sup_x[mb],sup_y[mb],model,pg_model,full=full)
        
        l1s.append(l1.numpy())
        l2s.append(l2.numpy())
    times.append(time.time()-start)

    print("Epoch",i+1,"Pred Loss:", str(np.mean(l1s,axis=0)).ljust(10), "Weight loss:", str(np.mean(l2s,axis=0)*1e8).ljust(10))
    hist.append([l1s,l2s])


hist = np.array(hist)

os.system(f"mkdir -p {os.path.join(args.save_dir,args.name)}")
pg_model.save_weights(os.path.join(args.save_dir,args.name,args.name,"pgm_new"))
np.save(f"{os.path.join(args.save_dir,args.name)}/lr_history.npy",hist)