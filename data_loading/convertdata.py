# adapted from https://github.com/una-dinosauria/human-motion-prediction/
import data_loading.data_utils as data_utils
import numpy as np
import tensorflow as tf

# this code was adapted from https://github.com/una-dinosauria/human-motion-prediction/
# all the credits will be given to the author



# train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = read_all_data(
#     actions, FLAGS.seq_length_in, FLAGS.seq_length_out, FLAGS.data_dir, not FLAGS.omit_one_hot )

# encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch( train_set, not FLAGS.omit_one_hot )

def read_all_data( actions, seq_length_in, seq_length_out, data_dir, one_hot ):
    """
    Loads data for training/testing and normalizes it.
    Args
    actions: list of strings (actions) to load
    seq_length_in: number of frames to use in the burn-in sequence
    seq_length_out: number of frames to use in the output sequence
    data_dir: directory to load the data from
    one_hot: whether to use one-hot encoding per action
    Returns
    train_set: dictionary with normalized training data
    test_set: dictionary with test data
    data_mean: d-long vector with the mean of the training data
    data_std: d-long vector with the standard dev of the training data
    dim_to_ignore: dimensions that are not used becaused stdev is too small
    dim_to_use: dimensions that we are actually using in the model
    """

    # === Read training data ===
    # print ("Reading training data (seq_len_in: {0}, seq_len_out {1}).".format(
    #        seq_length_in, seq_length_out))

    train_subject_ids = [1,6,7,8,9,11]
    test_subject_ids = [5]

    train_set, complete_train = data_utils.load_data( data_dir, train_subject_ids, actions, one_hot )
    test_set,  complete_test  = data_utils.load_data( data_dir, test_subject_ids,  actions, one_hot )

    # Compute normalization stats
    data_mean, data_std, dim_to_ignore, dim_to_use = data_utils.normalization_stats(complete_train)

    # Normalize -- subtract mean, divide by stdev
    train_set = data_utils.normalize_data( train_set, data_mean, data_std, dim_to_use, actions, one_hot )
    test_set  = data_utils.normalize_data( test_set,  data_mean, data_std, dim_to_use, actions, one_hot )
    # print("done reading data.")

    return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use

def get_batch(data, actions, batch_size, source_seq_len, target_seq_len, input_size, replace=True):
    """Get a random batch of data from the specified bucket, prepare for step.
    Args
      data: a list of sequences of size n-by-d to fit the model to.
      actions: a list of the actions we are using
    Returns
      The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
      the constructed batches have the proper format to call step(...) later.
    """
    while True:
            # Select entries at random
        all_keys    = list(data.keys())
        chosen_keys = np.random.choice( len(all_keys), batch_size, replace=replace )

        # How many frames in total do we need?
        total_frames = source_seq_len + target_seq_len

        encoder_inputs  = np.zeros((batch_size, source_seq_len,   input_size), dtype=float)
        decoder_inputs  = np.zeros((batch_size, target_seq_len,   input_size), dtype=float)
        decoder_outputs = np.zeros((batch_size, target_seq_len,   input_size), dtype=float)

        for i in range( batch_size ):

            the_key = all_keys[ chosen_keys[i] ]

            # Get the number of frames
            n, _ = data[ the_key ].shape

            # Sample somewherein the middle
            idx = np.random.randint( 16, n-total_frames )

            # Select the data around the sampled points
            data_sel = data[ the_key ][idx:idx+total_frames ,:]

            # Add the data
            encoder_inputs[i,:,0:input_size]  = data_sel[0:source_seq_len, :]
            # decoder_inputs[i,:,0:input_size]  = data_sel[source_seq_len-1:source_seq_len+target_seq_len-1, :]
            decoder_outputs[i,:,0:input_size] = data_sel[source_seq_len:, 0:input_size]

        # yield encoder_inputs, decoder_inputs, decoder_outputs
        yield encoder_inputs, decoder_outputs

def get_batch_srnn(data, action, source_seq_len, target_seq_len, input_size, batch_size = 8):
    """
    Get a random batch of data from the specified bucket, prepare for step.
    Args
      data: dictionary with k:v, k=((subject, action, subsequence, 'even')),
        v=nxd matrix with a sequence of poses
      action: the action to load data from
    Returns
      The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
      the constructed batches have the proper format to call step(...) later.
    """

    actions = ["directions", "discussion", "eating", "greeting", "phoning",
              "posing", "purchases", "sitting", "sittingdown", "smoking",
              "takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]

    if not action in actions:
        raise ValueError("Unrecognized action {0}".format(action))

    frames = {}
    frames[ action ] = find_indices_srnn( data, action )

    #batch_size = 8 # we always evaluate 8 seeds
    subject    = 5 # we always evaluate on subject 5

    seeds = [( action, (i%2)+1, frames[action][i] ) for i in range(batch_size)]

    encoder_inputs  = np.zeros( (batch_size, source_seq_len, input_size), dtype=float ) ## changed -1 here
    decoder_inputs  = np.zeros( (batch_size, target_seq_len, input_size), dtype=float )
    decoder_outputs = np.zeros( (batch_size, target_seq_len, input_size), dtype=float )

    # Compute the number of frames needed
    total_frames = source_seq_len + target_seq_len

    # Reproducing SRNN's sequence subsequence selection as done in
    # https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L343
    for i in range( batch_size ):

        _, subsequence, idx = seeds[i]
        idx = idx + 50

        data_sel = data[ (subject, action, subsequence, 'even') ]

        data_sel = data_sel[(idx-source_seq_len):(idx+target_seq_len) ,:]

        encoder_inputs[i, :, :]  = data_sel[0:source_seq_len, :] ## changed -1 here
        #decoder_inputs[i, :, :]  = data_sel[source_seq_len-1:(source_seq_len+target_seq_len-1), :]
        decoder_outputs[i, :, :] = data_sel[source_seq_len:, :]


    return encoder_inputs, decoder_outputs

def find_indices_srnn(data, action ):
    """
    Find the same action indices as in SRNN.
    See https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
    """

    # Used a fixed dummy seed, following
    # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
    SEED = 1234567890
    rng = np.random.RandomState( SEED )

    subject = 5
    subaction1 = 1
    subaction2 = 2

    T1 = data[ (subject, action, subaction1, 'even') ].shape[0]
    T2 = data[ (subject, action, subaction2, 'even') ].shape[0]
    prefix, suffix = 50, 100

    idx = []
    idx.append( rng.randint( 16,T1-prefix-suffix ))
    idx.append( rng.randint( 16,T2-prefix-suffix ))
    idx.append( rng.randint( 16,T1-prefix-suffix ))
    idx.append( rng.randint( 16,T2-prefix-suffix ))
    idx.append( rng.randint( 16,T1-prefix-suffix ))
    idx.append( rng.randint( 16,T2-prefix-suffix ))
    idx.append( rng.randint( 16,T1-prefix-suffix ))
    idx.append( rng.randint( 16,T2-prefix-suffix ))
    return idx
        


def define_actions( action ):
    """
    Define the list of actions we are using.
    Args
        action: String with the passed action. Could be "all"
    Returns
        actions: List of strings of actions
    Raises
        ValueError if the action is not included in H3.6M
    """

    actions = ["walking", "eating", "smoking", "discussion",  "directions",
              "greeting", "phoning", "posing", "purchases", "sitting",
              "sittingdown", "takingphoto", "waiting", "walkingdog",
              "walkingtogether"]

    if action in actions:
        return [action]

    if action == "all":
        return actions

    if action == "all_srnn":
        return ["walking", "eating", "smoking", "discussion"]

    raise( ValueError, "Unrecognized action: %d" % action )


def get_srnn_gts( actions, test_set, data_mean, data_std, dim_to_ignore, one_hot, source_seq_len, target_seq_len, input_size, to_euler=True, ):
    """
    Get the ground truths for srnn's sequences, and convert to Euler angles.
    (the error is always computed in Euler angles).
    Args
    actions: a list of actions to get ground truths for.
    model: training model we are using (we only use the "get_batch" method).
    test_set: dictionary with normalized training data.
    data_mean: d-long vector with the mean of the training data.
    data_std: d-long vector with the standard deviation of the training data.
    dim_to_ignore: dimensions that we are not using to train/predict.
    one_hot: whether the data comes with one-hot encoding indicating action.
    to_euler: whether to convert the angles to Euler format or keep thm in exponential map
    Returns
    srnn_gts_euler: a dictionary where the keys are actions, and the values
      are the ground_truth, denormalized expected outputs of srnns's seeds.
    """
    srnn_gts_euler = {}

    for action in actions:

        srnn_gt_euler = []
        _, srnn_expmap = get_batch_srnn( test_set, action, source_seq_len, target_seq_len, input_size)

        # expmap -> rotmat -> euler
        for i in np.arange( srnn_expmap.shape[0] ):
            denormed = data_utils.unNormalizeData(srnn_expmap[i,:,:], data_mean, data_std, dim_to_ignore, actions, one_hot )

            if to_euler:
                for j in np.arange( denormed.shape[0] ):
                    for k in np.arange(3,97,3):
                        denormed[j,k:k+3] = data_utils.rotmat2euler( data_utils.expmap2rotmat( denormed[j,k:k+3] ))

            srnn_gt_euler.append( denormed );

        # Put back in the dictionary
        srnn_gts_euler[action] = srnn_gt_euler

    return srnn_gts_euler


def get_test_data( actions, test_set, data_mean, data_std, dim_to_ignore, one_hot, source_seq_len, target_seq_len, input_size, to_euler=True, batch_size=8):
    """
    Get the ground truths for srnn's sequences, and convert to Euler angles.
    (the error is always computed in Euler angles).
    Args
    actions: a list of actions to get ground truths for.
    model: training model we are using (we only use the "get_batch" method).
    test_set: dictionary with normalized training data.
    data_mean: d-long vector with the mean of the training data.
    data_std: d-long vector with the standard deviation of the training data.
    dim_to_ignore: dimensions that we are not using to train/predict.
    one_hot: whether the data comes with one-hot encoding indicating action.
    to_euler: whether to convert the angles to Euler format or keep thm in exponential map
    Returns
    srnn_gts_euler: a dictionary where the keys are actions, and the values
      are the ground_truth, denormalized expected outputs of srnns's seeds.
    """
    srnn_gts_euler = {}

    for action in actions:

        srnn_gt_euler = []
        x, srnn_expmap = get_batch_srnn( test_set, action, source_seq_len, target_seq_len, input_size, batch_size=batch_size)

        # expmap -> rotmat -> euler
        for i in np.arange( srnn_expmap.shape[0] ):
            denormed = data_utils.unNormalizeData(srnn_expmap[i,:,:], data_mean, data_std, dim_to_ignore, actions, one_hot )

            if to_euler:
                for j in np.arange( denormed.shape[0] ):
                    for k in np.arange(3,97,3):
                        denormed[j,k:k+3] = data_utils.rotmat2euler( data_utils.expmap2rotmat( denormed[j,k:k+3] ))

            srnn_gt_euler.append( denormed );

        # Put back in the dictionary

        yield (x, srnn_expmap, srnn_gt_euler)

# Given a prediction in exp map and a label in euler, computes mean angular error
def evaluate_euler(y_pred, y_true_euler, data_mean, data_std, dim_to_ignore, actions, one_hot):
    srnn_pred_expmap = tf.transpose(y_pred,[1,0,2])
    srnn_pred_expmap = data_utils.revert_output_format( srnn_pred_expmap,
                data_mean, data_std, dim_to_ignore, actions, one_hot)
    mean_errors = np.zeros( (len(srnn_pred_expmap), srnn_pred_expmap[0].shape[0]) )

    # Compute loss

    for i in range(len(y_pred)):

        eulerchannels_pred = srnn_pred_expmap[i]

        # Convert from exponential map to Euler angles
        for j in np.arange( eulerchannels_pred.shape[0] ):
              for k in np.arange(3,97,3):
                eulerchannels_pred[j,k:k+3] = data_utils.rotmat2euler(data_utils.expmap2rotmat( eulerchannels_pred[j,k:k+3] ))

        gt_i = y_true_euler[i]
        gt_i[:,0:6] = 0
        idx_to_use = np.where( np.std( gt_i, 0 ) > 1e-4 )[0]
        euc_error = np.power( gt_i[:,idx_to_use] - eulerchannels_pred[:,idx_to_use], 2)
        euc_error = np.sum(euc_error, 1)
        euc_error = np.sqrt( euc_error )
        mean_errors[i,:] = euc_error
        
    return mean_errors