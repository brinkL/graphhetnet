import tensorflow as tf
from MoPredNet.models import DSTCEncoder, FCNet, Masking

class MoPredNet(tf.keras.Model):

    def __init__(self, drop=0.5, target_length=50, recent_k=10, masking=True):
        super(MoPredNet, self).__init__()
        
        self.enc_a           = DSTCEncoder(drop=drop)
        self.enc_m           = DSTCEncoder(drop=drop)
        
        self.dec           = FCNet(drop=drop, layer_sizes = [512,128,54], activation="lrelu")
        self.mask          = Masking()
        self.target_length = target_length

        self.recent_k = recent_k
        self.masking = masking


    def callOld(self, x, training = False, debug=False):
        
        l     = tf.shape(x)[1]
        j     = tf.shape(x)[2]

        # Pad input to f_m to length "source+target" to have fixed length input 
        diff  = self.target_length
        zeros = tf.zeros([tf.shape(x)[0], diff ,tf.shape(x)[2],tf.shape(x)[3]])
        x_pad = tf.concat([x,zeros],axis=1)
        v_ms = []
        # Compute embeddings v_0 and v_m based on observations and masked forecasts
        v_0   = self.enc_a(x, training=training)
        v_m   = self.enc_m(x_pad, training=training)
        v_ms.append(x_pad)
        # Generate first forecasted pose x_t+1 and reshape it to format (BxTxSxC)
        dec_in = tf.concat([v_0,v_m],axis=-1)
        new_pose  = self.dec(dec_in, training=training)
        new_pose  = tf.reshape(new_pose,[-1,1,j,1])

        # Set x_new to be observations + previous poses [x,xhat]
        x_new = x

        # Forecast remaining poses x_t+k
        for k in range(1,self.target_length):
            
            # Compute masked recent poses
            mask_out = self.mask(x_new,new_pose)
            x_new = tf.concat([x_new,new_pose],axis=1)
            
            # Pad input to f_m to length "source+target" to have fixed length input 
            diff  = self.target_length-k
            zeros = tf.zeros([tf.shape(x)[0], diff ,tf.shape(x)[2],tf.shape(x)[3]])
            x_pad = tf.concat([mask_out,zeros],axis=1)

            # Compute updated embeddings for v_m based on observations and masked forecasts
            v_m   = self.enc_m(x_pad, training=training)
            dec_in = tf.concat([v_0,v_m],axis=-1)
            v_ms.append(x_pad)
            # Generate next forecasted pose x_t+k and reshape it to format (BxTxSxC)
            new_pose  = self.dec(dec_in, training=training)
            new_pose  = tf.reshape(new_pose,[-1,1,j,1])
        
        x_new = tf.concat([x_new,new_pose],axis=1)
        #[5, 60, 54, 1])
        out = tf.slice(x_new,[0,l,0,0],[-1,self.target_length,-1,-1])
        out = tf.squeeze(out,axis=-1)
        if debug:
            return out,v_ms
        return out


    def call(self, x, training = False, debug=False):
        
        l     = tf.shape(x)[1]
        j     = tf.shape(x)[2]

        # Pad input to f_m to length "source+target" to have fixed length input 
        diff  = self.target_length

        x_pad = tf.slice(x,[0,l-self.recent_k,0,0],[-1,-1,-1,-1])

        if debug:
            v_ms =  []
            masks = []

        # Compute embeddings v_0 and v_m based on observations and masked forecasts
        v_0   = self.enc_a(x, training=training)
        v_m   = self.enc_m(x_pad, training=training)
        if debug:
            v_ms.append(x_pad)

        # Generate first forecasted pose x_t+1 and reshape it to format (BxTxSxC)
        dec_in = tf.concat([v_0,v_m],axis=-1)
        new_pose  = self.dec(dec_in, training=training)
        new_pose  = tf.reshape(new_pose,[-1,1,j,1])

        # Set x_new to be observations + previous poses [x,xhat]
        x_new = x

        # Forecast remaining poses x_t+k
        for k in range(1,self.target_length):
            
            # Compute masked recent poses
            if self.masking:
                x_pad = self.mask(x_pad,new_pose,debug=debug)
            else:
                x_pad = tf.concat([x_pad,new_pose],axis=1)

            if debug and self.masking:
                x_pad, mask = x_pad
                masks.append(mask)

            x_new = tf.concat([x_new,new_pose],axis=1)
            
            # Pad input to f_m to length "source+target" to have fixed length input 
            diff  = self.target_length-k

            x_pad = tf.slice(x_pad,[0,1,0,0],[-1,-1,-1,-1])

            # Compute updated embeddings for v_m based on observations and masked forecasts
            v_m   = self.enc_m(x_pad, training=training)
            dec_in = tf.concat([v_0,v_m],axis=-1)
            if debug:
                v_ms.append(x_pad)

            # Generate next forecasted pose x_t+k and reshape it to format (BxTxSxC)
            new_pose  = self.dec(dec_in, training=training)
            new_pose  = tf.reshape(new_pose,[-1,1,j,1])
        
        x_new = tf.concat([x_new,new_pose],axis=1)
        #[5, 60, 54, 1])
        out = tf.slice(x_new,[0,l,0,0],[-1,self.target_length,-1,-1])
        out = tf.squeeze(out,axis=-1)
        if debug:
            return out,v_ms,masks
        return out