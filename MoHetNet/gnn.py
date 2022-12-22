#adapted from https://github.com/keras-team/keras-io/blob/master/examples/graph/gnn_citations.py

import tensorflow.keras.layers as layers
import tensorflow as tf

class GraphConvLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        aggregation_type="mean",
        combination_type="concat",
        combination_net = None,
        prepare_net = None,
        normalize = False,
        node_axis = 2,
        *args,
        **kwargs,
    ):
        super(GraphConvLayer, self).__init__(*args, **kwargs)

        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.normalize = normalize
        self.node_axis = 2

        if prepare_net is None:
            raise ValueError("Need to pass network for combination_net")
        else:
            self.ffn_prepare = prepare_net
        
        if combination_net is None:
            raise ValueError("Need to pass network for combination_net")
        else:
            self.update_fn = combination_net

    def prepare(self, node_repesentations, weights=None):
        # node_repesentations shape is [num_edges, embedding_dim].
        messages = self.ffn_prepare(node_repesentations)
        if weights is not None:
            messages = messages * tf.expand_dims(weights, -1)
        return messages

    def aggregate(self, node_indices, neighbour_messages, node_repesentations):
        # node_indices shape is [num_edges].
        # neighbour_messages shape: [num_edges, representation_dim].
        # node_repesentations shape is [num_nodes, representation_dim]
        
        # transpose axes with node in the beginning 
        trans, trans_in = getTrans(len(neighbour_messages.shape),self.node_axis)
        
        neighbour_messages = tf.transpose(neighbour_messages,trans)
        node_repesentations = tf.transpose(node_repesentations,trans)

        num_nodes = node_repesentations.shape[0]
        if self.aggregation_type == "sum":
            aggregated_message = tf.math.unsorted_segment_sum(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        elif self.aggregation_type == "mean":
            aggregated_message = tf.math.unsorted_segment_mean(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        elif self.aggregation_type == "max":
            aggregated_message = tf.math.unsorted_segment_max(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        else:
            raise ValueError(f"Invalid aggregation type: {self.aggregation_type}.")
            
        aggregated_message = tf.transpose(aggregated_message,trans_in)

        return aggregated_message

    def update(self, node_repesentations, aggregated_messages):
        # node_repesentations shape is [num_nodes, representation_dim].
        # aggregated_messages shape is [num_nodes, representation_dim].
        if self.combination_type == "gru":
            # Create a sequence of two elements for the GRU layer.
            h = tf.stack([node_repesentations, aggregated_messages], axis=-2)
        elif self.combination_type == "concat":
            # Concatenate the node_repesentations and aggregated_messages.
            
            h = tf.concat([node_repesentations, aggregated_messages], axis=-1)
        elif self.combination_type == "add":
            # Add node_repesentations and aggregated_messages.
            h = node_repesentations + aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}.")

        # Apply the processing function.
        node_embeddings = self.update_fn(h)
        if self.combination_type == "gru":
            node_embeddings = tf.unstack(node_embeddings, axis=1)[-1]

        if self.normalize:
            node_embeddings = tf.nn.l2_normalize(node_embeddings, axis=-1)
        return node_embeddings

    def call(self, node_repesentations, edges):
        """Process the inputs to produce the node_embeddings.

        inputs: a tuple of three elements: node_repesentations, edges, edge_weights.
        Returns: node_embeddings of shape [num_nodes, representation_dim].
        """
        edge_weights = None
        # Get node_indices (source) and neighbour_indices (target) from edges.
        node_indices, neighbour_indices = edges[0], edges[1]

        # neighbour_repesentations shape is [num_edges, representation_dim].
        neighbour_repesentations = tf.gather(node_repesentations, neighbour_indices,axis=self.node_axis)

        # Prepare the messages of the neighbours.
        neighbour_messages = self.prepare(neighbour_repesentations, edge_weights)

        # Aggregate the neighbour messages.
        aggregated_messages = self.aggregate(
            node_indices, neighbour_messages, node_repesentations
        )
        # Update the node embedding with the neighbour messages.
        return self.update(node_repesentations, aggregated_messages)
    

# computes the transpose vector for shifting an axis to the front and inverse it again
def getTrans(length, axis):
    oList = list(range(length))
    t  = [oList[axis]] + oList[:axis] + oList[axis+1:]
    
    ti = oList[1:axis+1] + [0] + oList[axis+1:]
    
    return t,ti