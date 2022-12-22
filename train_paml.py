import tensorflow as tf
import gc
import sys
import os
import numpy as np
import datetime
gc.enable()
tf.__version__

from data_loading.convertdata import define_actions
#from convertdata import read_all_data, get_batch_srnn, define_actions, get_batch, get_srnn_gts
#import data_utils
from data_loading.data_gen import genMotionTask, subgraphWrapper

from PAML.maml_motion import MAML
from PAML.models import ResGRUCell, EncoderModel, DecoderModel
from args            import argument_parser
from save import saveMe

one_hot = False
angles  = 54
source_len = 50
target_len = 10
if one_hot:
    full_angles = angles+15
else:
    full_angles = angles

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

padded = True

#all_actions = define_actions("all")
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

train_gen = genMotionTask(train_actions, 5,
                    one_hot    = False,
                    angles     = 54,
                    source_len = 50, 
                    target_len = 10,
                    num_labels = 15,
                    full_query = True,
                    data_dir=args.data_dir)

test_gen = genMotionTask(test_actions, 5,
                    one_hot    = False,
                    angles     = 54,
                    source_len = 50, 
                    target_len = 10,
                    num_labels = 15,
                    full_query = True,
                    data_dir=args.data_dir)

edges = np.load(args.edges_dir)



if args.sample_joints == "full":
    train_gen = preGen(train_gen)
    padded = False

elif args.sample_joints == "sub":
    train_gen = subgraphWrapper(train_gen, edges, pad_data=True)
else:
    raise ValueError("Choose one of full/sub for sample_joints")

if padded:
    (que_x,sup_x,sup_y,_,pad),que_y = next(train_gen)
    #sup_x,sup_y,que_x,que_y,action,padding  = next(train_gen)
    print("Real feats:",pad)
else:
    (que_x,sup_x,sup_y,_),que_y = next(train_gen)

print("Data shapes:",sup_x.shape,sup_y.shape,que_x.shape,que_y.shape)

gru = tf.keras.layers.RNN(ResGRUCell(num_units=1024,num_outputs=full_angles,layer=1), return_sequences=True, return_state=True)

enc = EncoderModel(gru)
dec = DecoderModel(gru)

inner_loss = tf.keras.losses.MeanSquaredError(reduction="none")
outer_loss = tf.keras.losses.MeanSquaredError()

myMaml = MAML(enc, dec, inner_loss=inner_loss, outer_loss=outer_loss)

x_enc, x_dec = sup_x[0][:,:-1,:] , sup_x[0][:,-1:,:]
cur_s = enc(x_enc)
cur_x , cur_s = dec(x_dec, cur_s)

outer_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
inner_optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)

losses = []
last_loss = []

for i in range(args.num_epochs):

    if padded:
        (que_x,sup_x,sup_y,_,padding),que_y = next(train_gen)
    else:
        (que_x,sup_x,sup_y,_),que_y = next(train_gen)
        padding = None


    mean_loss, r_loss, _ = myMaml.train_on_batch(que_x,sup_x,sup_y,que_y,padding, inner_optimizer, inner_step=3, target_len=target_len,outer_optimizer=outer_optimizer)
    losses.append([mean_loss,r_loss])#
    last_loss.append(mean_loss)

    if i%args.steps==0:
        print(i,np.mean(last_loss))
        last_loss = []
        
losses = np.array(losses)

os.system(f"mkdir -p {os.path.join(args.save_dir,args.name)}")
enc.save_weights(os.path.join(args.save_dir,args.name,args.name,"enc"))
dec.save_weights(os.path.join(args.save_dir,args.name,args.name,"dec"))
np.save(f"{os.path.join(args.save_dir,args.name)}/history.npy",losses)
saveMe(args,"TrainMamlMotion",{"meanloss_-_rloss":losses})
