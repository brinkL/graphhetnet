#this code was adapted from: https://github.com/Runist/MAML-keras/blob/master/net.py

import tensorflow as tf
import numpy as np

## MAML with option for weight look ahead

class MAML:
    def __init__(self, enc, dec, inner_loss, outer_loss, look_ahead=None,final_weights=None):


        self.meta_enc  = enc
        self.meta_dec  = dec

        if final_weights is None:
            if look_ahead is not None:
                print("Warning!: H_model given but no weight targets. Proceed regular MAML")
            self.look_ahead = None
        else:
            if look_ahead is None:
                print("Warning!: Weight targets given but no H_model. Proceed regular MAML")

            else:

                # Compute vector for weights targets
                self.final_vec = {}
                for k,v in final_weights.items():
                    self.final_vec[k] = weights2vec(v)

                # get weight shapes from one target
                self.shapes = []
                for w in list(final_weights.values())[0]:
                    self.shapes.append(w.shape)

            self.look_ahead = look_ahead
        
        self.inner_loss = inner_loss
        self.outer_loss = outer_loss


    def train_on_batch(self, que_x,sup_x,sup_y,que_y,padding, inner_optimizer, inner_step, target_len, lb=0.1, outer_optimizer=None):

        batch_loss = []
        batch_loss_real = []
        batch_loss_h    = []
        task_weights = []
        weight_losses = []

        padded = padding is not None

        meta_weights = self.meta_dec.get_weights()

        meta_support_x, meta_support_y, meta_query_x, meta_query_y, padding = (sup_x,sup_y,que_x,que_y,padding)

        
        for support_x, support_y in zip(meta_support_x, meta_support_y):#, meta_classes): took out meta_classes for weight forecast

            self.meta_dec.set_weights(meta_weights)

            for _ in range(inner_step):
                with tf.GradientTape() as tape:
 
                    pred = execute(support_x,self.meta_enc,self.meta_dec,target_len)

                    loss = (support_y - pred)**2
                    loss = tf.slice(loss,[0,0,0],[-1,-1,padding])
                    #loss = self.inner_loss(support_y, logits)

                    loss = tf.reduce_mean(loss)

                grads = tape.gradient(loss, self.meta_dec.trainable_variables)
                gradients = [tf.clip_by_norm(g, 5) for g in grads]
                inner_optimizer.apply_gradients(zip(gradients, self.meta_dec.trainable_variables))

            updated_weights = self.meta_dec.get_weights()
            
            if self.look_ahead is not None:
                updated_weights = self.look_ahead(weights2vec(updated_weights))
                weight_loss     = tf.sqrt(tf.reduce_sum((updated_weights - self.final_vec[tuple(task_classes)])**2))
                updated_weights = vec2weights(updated_weights,self.shapes)
                weight_losses.append(weight_loss)
                
            task_weights.append(updated_weights)
            
        predictions = []

        with tf.GradientTape() as tape:
            for i, (query_x, query_y) in enumerate(zip(meta_query_x, meta_query_y)):

                self.meta_dec.set_weights(task_weights[i])

                pred = execute(query_x,self.meta_enc,self.meta_dec,target_len)
                predictions.append(pred)

                loss = (query_y - pred)**2
                loss = tf.slice(loss,[0,0,0],[-1,-1,padding])
                #loss = self.outer_loss(query_y, pred)
                
                batch_loss_real.append(loss)
                if self.look_ahead is not None:
                    h_loss = 0.5*lb*weight_losses[i]
                    batch_loss_h.append(h_loss)
                    loss += h_loss
                    
                loss = tf.reduce_mean(loss)
                batch_loss.append(loss)

            mean_loss = tf.reduce_mean(batch_loss)

        self.meta_dec.set_weights(meta_weights)
        if outer_optimizer:
            grads = tape.gradient(mean_loss, self.meta_dec.trainable_variables)
            #print("Out grads:",np.mean([np.mean(g) for g in grads]))
            outer_optimizer.apply_gradients(zip(grads, self.meta_dec.trainable_variables))

            return mean_loss, tf.reduce_mean(batch_loss_real), tf.reduce_mean(batch_loss_h)

        else:
            return predictions, mean_loss, tf.reduce_mean(batch_loss_real), tf.reduce_mean(batch_loss_h)

# Transform tensorflow weight list to single vector
def weights2vec(weights):
    weight_vec = []
    for w in weights:
        weight_vec.append(np.reshape(w,[1,-1]))
    weight_vec = np.concatenate(weight_vec,axis=1)
    return weight_vec

# Transform single weight vector to tensorflow weight list given a list of weight shapes
def vec2weights(weights,shapes):
    
    cur = 0
    orig_weights = []
    for s in shapes:
        cur_w = weights[:,cur:cur+np.prod(s)]
        cur_w = np.reshape(cur_w,s)
        cur+=np.prod(s)
        orig_weights.append(cur_w)
    return orig_weights

@tf.function
def execute(x,enc,dec,size=10,channels=69):
    x_enc, x_dec = x[:,:-1,:] , x[:,-1:,:]
    cur_s = enc(x_enc)
    cur_x , cur_s = dec(x_dec, cur_s)
    y_hat = cur_x
    # print(y_hat)
    for i in np.arange(start=1, stop=size):
        cur_x , cur_s = dec(cur_x, cur_s)
        y_hat = tf.concat([y_hat,cur_x],axis=1)       
    return y_hat