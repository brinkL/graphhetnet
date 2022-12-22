#Code adapted from TF2 Version of https://github.com/kastnerkyle/deform-conv/blob/126ebcc283a4325c474332fa170f57d52a59e34d/deform_conv/deform_conv.py#L8

import tensorflow as tf
import numpy as np

class ConvOffset2D(tf.keras.layers.Conv2D):
    """ConvOffset2D"""

    def __init__(self, filters, init_normal_stddev=0.01, time_axis=None, **kwargs):
        """Init"""
        
        self.time_axis = time_axis
        self.filters = filters
        
        # ReLU to make temporal offsets negative only
        if time_axis is not None:
            self.relu    = tf.keras.layers.ReLU()
            
        super(ConvOffset2D, self).__init__(
            self.filters * 2, (3, 3), padding='same', use_bias=False,
            # TODO gradients are near zero if init is zeros
            #kernel_initializer='zeros',
            # kernel_initializer=RandomNormal(0, init_normal_stddev),
            **kwargs
        )

    def call(self, x, debug=False):
        if debug:
            all_out = []

        x_shape = tf.shape(x)
        offsets = super(ConvOffset2D, self).call(x)
        offsets = offsets*20.
        if debug:
            all_out.append(offsets)
        # Offset (b, h, w, 2c) -> (b*c, h, w, 2)
        offsets = tf.transpose(offsets, [0, 3, 1, 2])
        offsets = tf.reshape(offsets, (-1, x_shape[1], x_shape[2], 2))
        if debug:
            all_out.append(offsets)
        # Set temporal offsets to [-inf,0] via relu
        if self.time_axis is not None:
            offsets_time = tf.slice(offsets,[0,0,0,self.time_axis],[-1,-1,-1,1])
            offsets_time = -self.relu(offsets_time)
            offsets_time = tf.zeros(tf.shape(offsets_time))
            
            if self.time_axis == 0:
                offsets = tf.concat([offsets_time,tf.slice(offsets,[0,0,0,1],[-1,-1,-1,1])],axis=-1)
            else:
                offsets = tf.concat([tf.slice(offsets,[0,0,0,1],[-1,-1,-1,1]),offsets_time],axis=-1)
        if debug:
            all_out.append(offsets)
        # Input (b, h, w, c) -> (b*c, h, w)
        x = tf.transpose(x, [0, 3, 1, 2])
        x = tf.reshape(x, (-1, x_shape[1], x_shape[2]))
        if debug:
            all_out.append(x)
        # Requires that channels of x is = filters
        x_offset = tf_batch_map_offsets_v2(x, offsets,debug=debug)
        if debug:
            x_offset,allout = x_offset
            all_out.append(x_offset)
        # Output (b*c, h, w) -> (b, h, w, c)
        x_offset = tf.reshape(
            x_offset, (-1, x_shape[3], x_shape[1], x_shape[2])
        )
        x_offset = tf.transpose(x_offset, [0, 2, 3, 1])

        if debug:
            all_out.append(x_offset)
            return x_offset, all_out,allout


        return x_offset

def tf_map_coordinates(input, coords, order=1):
    """Tensorflow verion of scipy.ndimage.map_coordinates
    Note that coords is transposed and only 2D is supported
    Parameters
    ----------
    input : tf.Tensor. shape = (s, s)
    coords : tf.Tensor. shape = (n_points, 2)
    """

    assert order == 1

    coords_lt = tf.cast(tf.floor(coords), 'int32')
    coords_rb = tf.cast(tf.math.ceil(coords), 'int32')
    coords_lb = tf.stack([coords_lt[:, 0], coords_rb[:, 1]], axis=1)
    coords_rt = tf.stack([coords_rb[:, 0], coords_lt[:, 1]], axis=1)

    vals_lt = tf.gather_nd(input, coords_lt)
    vals_rb = tf.gather_nd(input, coords_rb)
    vals_lb = tf.gather_nd(input, coords_lb)
    vals_rt = tf.gather_nd(input, coords_rt)

    coords_offset_lt = coords - tf.cast(coords_lt, 'float32')
    vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[:, 0]
    vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[:, 0]
    mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[:, 1]

    return mapped_vals

def tf_batch_map_coordinates(input, coords, order=1):
    """Batch version of tf_map_coordinates
    Only supports 2D feature maps
    Parameters
    ----------
    input : tf.Tensor. shape = (b, s, s)
    coords : tf.Tensor. shape = (b, n_points, 2)
    """

    input_shape = tf.shape(input)
    batch_size = input_shape[0]
    input_size = input_shape[1]
    n_coords = tf.shape(coords)[1]

    coords = tf.clip_by_value(coords, 0, tf.cast(input_size, 'float32') - 1)
    coords_lt = tf.cast(tf.math.floor(coords), 'int32')
    coords_rb = tf.cast(tf.math.ceil(coords), 'int32')
    coords_lb = tf.stack([coords_lt[..., 0], coords_rb[..., 1]], axis=-1)
    coords_rt = tf.stack([coords_rb[..., 0], coords_lt[..., 1]], axis=-1)

    idx = tf.repeat(tf.range(batch_size), n_coords)

    def _get_vals_by_coords(input, coords):
        indices = tf.stack([
            idx, tf.reshape(coords[..., 0],[-1]), tf.reshape(coords[..., 1],[-1])
        ], axis=-1)
        vals = tf.gather_nd(input, indices)
        vals = tf.reshape(vals, (batch_size, n_coords))
        return vals

    vals_lt = _get_vals_by_coords(input, coords_lt)
    vals_rb = _get_vals_by_coords(input, coords_rb)
    vals_lb = _get_vals_by_coords(input, coords_lb)
    vals_rt = _get_vals_by_coords(input, coords_rt)

    coords_offset_lt = coords - tf.cast(coords_lt, 'float32')
    vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[..., 0]
    vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[..., 0]
    mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[..., 1]

    return mapped_vals

# allow for non square inputs
def tf_batch_map_coordinates_v2(input, coords, order=1):
    """Batch version of tf_map_coordinates
    Only supports 2D feature maps
    Parameters
    ----------
    input : tf.Tensor. shape = (b, s1, s2)
    coords : tf.Tensor. shape = (b, n_points, 2)
    """

    input_shape = tf.shape(input)
    batch_size = input_shape[0]
    input_size1 = input_shape[1]
    input_size2 = input_shape[2]
    n_coords = tf.shape(coords)[1]

    # updated to allow for input with w!=h
    coords0 = tf.slice(coords,[0,0,0],[-1,-1,1])
    coords1 = tf.slice(coords,[0,0,1],[-1,-1,1])
    coords0 = tf.clip_by_value(coords0, 0, tf.cast(input_size1, 'float32') - 1)
    coords1 = tf.clip_by_value(coords1, 0, tf.cast(input_size2, 'float32') - 1)
    coords = tf.concat([coords0,coords1],axis=-1)
    
    coords_lt = tf.cast(tf.math.floor(coords), 'int32')
    coords_rb = tf.cast(tf.math.ceil(coords), 'int32')
    coords_lb = tf.stack([coords_lt[..., 0], coords_rb[..., 1]], axis=-1)
    coords_rt = tf.stack([coords_rb[..., 0], coords_lt[..., 1]], axis=-1)

    idx = tf.repeat(tf.range(batch_size), n_coords)

    def _get_vals_by_coords(input, coords):
        indices = tf.stack([
            idx, tf.reshape(coords[..., 0],[-1]), tf.reshape(coords[..., 1],[-1])
        ], axis=-1)
        vals = tf.gather_nd(input, indices)
        vals = tf.reshape(vals, (batch_size, n_coords))
        return vals

    vals_lt = _get_vals_by_coords(input, coords_lt)
    vals_rb = _get_vals_by_coords(input, coords_rb)
    vals_lb = _get_vals_by_coords(input, coords_lb)
    vals_rt = _get_vals_by_coords(input, coords_rt)

    coords_offset_lt = coords - tf.cast(coords_lt, 'float32')
    vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[..., 0]
    vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[..., 0]
    mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[..., 1]

    return mapped_vals

def tf_repeat_2d(a, repeats):
    """Tensorflow version of np.repeat for 2D"""

    assert len(tf.shape(a)) == 2
    a = tf.expand_dims(a, 0)
    a = tf.tile(a, [repeats, 1, 1])
    return a

def tf_batch_map_offsets(input, offsets, order=1):
    """Batch map offsets into input
    Parameters
    ---------
    input : tf.Tensor. shape = (b, s, s)
    offsets: tf.Tensor. shape = (b, s, s, 2)
    """

    input_shape = tf.shape(input)
    batch_size = input_shape[0]
    input_size = input_shape[1]

    offsets = tf.reshape(offsets, (batch_size, -1, 2))
    grid = tf.meshgrid(
        tf.range(input_size), tf.range(input_size), indexing='ij'
    )
    grid = tf.stack(grid, axis=-1)
    grid = tf.cast(grid, 'float32')
    grid = tf.reshape(grid, (-1, 2))
    grid = tf_repeat_2d(grid, batch_size)
    coords = offsets + grid

    mapped_vals = tf_batch_map_coordinates(input, coords)
    return mapped_vals

# allow for non square inputs
def tf_batch_map_offsets_v2(input, offsets, order=1,debug=False):
    """Batch map offsets into input
    Parameters
    ---------
    input : tf.Tensor. shape = (b, s1, s2)
    offsets: tf.Tensor. shape = (b, s1, s2, 2)
    """
    if debug:
        allout = []
    input_shape = tf.shape(input)
    batch_size = input_shape[0]
    input_size1 = input_shape[1]
    input_size2 = input_shape[2]

    offsets = tf.reshape(offsets, (batch_size, -1, 2))
    if debug:
        allout.append(offsets)
    grid = tf.meshgrid(
        tf.range(input_size1), tf.range(input_size2), indexing='ij'
    )
    if debug:
        allout.append(grid)
    grid = tf.stack(grid, axis=-1)
    if debug:
        allout.append(grid)
    grid = tf.cast(grid, 'float32')
    grid = tf.reshape(grid, (-1, 2))
    grid = tf_repeat_2d(grid, batch_size)
    if debug:
        allout.append(grid)
    coords = offsets + grid
    if debug:
        allout.append(coords)
    mapped_vals = tf_batch_map_coordinates_v2(input, coords)
    if debug:
        return mapped_vals, allout
    return mapped_vals


# Transform tensorflow weight list to single vector
def weights2vec(weights):
    weight_vec = []
    for w in weights:
        weight_vec.append(tf.reshape(w,[1,-1]))
    weight_vec = tf.concat(weight_vec,axis=1)
    return weight_vec

# Transform single weight vector to tensorflow weight list given a list of weight shapes
def vec2weights(weights,shapes):
    
    cur = 0
    orig_weights = []
    for s in shapes:
        cur_w = weights[:,cur:cur+tf.reduce_prod(s)]
        cur_w = tf.reshape(cur_w,s)
        cur+=tf.reduce_prod(s)
        orig_weights.append(cur_w)
    return orig_weights