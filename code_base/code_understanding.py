import numpy as np
import tensorflow as tf
import os


class DNC:
    def __init__(self, input_size, output_size, seq_len, num_words=256, word_size=64, num_heads=4):
        # define data
        # input data - [[1 0] [0 1] [0 0] [0 0]]
        #  it will be a (4,2) ?
        self.input_size = input_size  # X
        # output data [[0 0] [0 0] [1 0] [0 1]]
        self.output_size = output_size  # Y

        #  MATRIX MEMORY DEF

        # define read + write vector size
        # this will define also the memory matrix
        # 10
        self.num_words = num_words  # N
        # 4 characters
        self.word_size = word_size  # W

        # READ AND WRITE HEADS

        # define number of read+write heads
        # we could have multiple, but just 1 for simplicity
        # see it has an operation
        # it is how many times we read and write form the memort for each feedforward (time step?)
        self.num_heads = num_heads  # R

        # INTERFACE VECTOR

        #  the interface vector is the other output in addition to the normal output of the network
        #  it defines how the controller will interact with the memory bank at the next time step
        # size of output vector from controller that defines interactions with memory matrix
        self.interface_size = (num_heads * word_size) + (3 * word_size) + (5 * num_heads) + 3
        #  those magic numbers are hyperparams

        # SIZE OF THE NN

        # the actual size of the neural network input after flatenning and
        # concatenating the input vector with the previously read vctors from memory (word_size)
        self.nn_input_size = (num_heads * word_size) + input_size
        # size of output(it will include also the vector of the interface)
        self.nn_output_size = output_size + self.interface_size

        #  why???
        # gaussian normal distribution for both outputs
        self.nn_out = tf.truncated_normal([1, self.output_size], stddev=0.1)
        self.interface_vec = tf.truncated_normal([1, self.interface_size], stddev=0.1)

        #  MEMORY MATRIX FOR WORDS

        # Create memory matrix
        self.mem_mat = tf.zeros([num_words, word_size])  # N*W

        # THE ASIDE MATRIX FOR SUPPORT READ AND WRITE OP

        # other variables (those concerning the other matrix: that is the one which takes into account the
        #  sequence of writings)
        # The usage vector records which locations have been used so far,
        self.usage_vec = tf.fill([num_words, 1], 1e-6)  # N*1
        # a temporal link matrix records the order in which locations were written;
        # (defined using the usage_vector)
        self.link_mat = tf.zeros([num_words, num_words])  # N*N
        # represents degrees to which last location was written to
        self.precedence_weight = tf.zeros([num_words, 1])  # N*1

        # WEIGHTS FOR THE WRITE AND READ HEADS

        # remind we have one head but we have weights for that head. They are used to define the degree of
        #  reading and writing. Read and write are just matrix moltiplication, we can tune how much we are doing
        #  this. Similar to a learning rate. Also those weights are differentiated
        # Read and write head weight variables
        self.read_weights = tf.fill([num_words, num_heads], 1e-6)  # N*R
        self.write_weights = tf.fill([num_words, 1], 1e-6)  # N*1
        self.read_vecs = tf.fill([num_heads, word_size], 1e-6)  # R*W

        ###NETWORK VARIABLES###

        # PLACEHOLDERS

        # gateways into the computation graph for input output pairs
        # it's like initializing the inputs and outputs
        self.i_data = tf.placeholder(tf.float32, [seq_len * 2, self.input_size], name='input_node')
        self.o_data = tf.placeholder(tf.float32, [seq_len * 2, self.output_size], name='output_node')

        # NETWORK LAYER WEIGHTS AND BIAS

        # 2 layer feedforwarded network
        #  NOTE: it's using 32 neurons for the hidden layers
        self.W1 = tf.Variable(tf.truncated_normal([self.nn_input_size, 32], stddev=0.1), \
                              name='layer1_weights', dtype=tf.float32)
        self.b1 = tf.Variable(tf.zeros([32]), \
                              name='layer1_bias', dtype=tf.float32)
        self.W2 = tf.Variable(tf.truncated_normal([32, self.nn_output_size], stddev=0.1), \
                              name='layer2_weights', dtype=tf.float32)
        self.b2 = tf.Variable(tf.zeros([self.nn_output_size]), \
                              name='layer2_bias', dtype=tf.float32)

        ###DNC OUTPUT WEIGHTS

        #  we have also weights for the output due to everything can be differentiated
        # NOTE: those are the weights form the second hidden layer to the output
        self.nn_out_weights = tf.Variable(tf.truncated_normal([self.nn_output_size, self.output_size], stddev=0.1),
                                          name='net_output_weights')
        self.interface_weights = tf.Variable(
            tf.truncated_normal([self.nn_output_size, self.interface_size], stddev=0.1), name='interface_weights')

        #  this for sure is related with the memory
        self.read_vecs_out_weight = tf.Variable(
            tf.truncated_normal([self.num_heads * self.word_size, self.output_size], stddev=0.1),
            name='read_vector_weights')

    #################################################
    # 3 attention mechanisms for read/writes to memory
    #################################################

    #### FIRST DIFFERENTIABLE ATTENTION ####

    # a key vector emitted by the controller is compared to the
    # content of each location in memory according to a -similarity measure-
    # The similarity scores determine a weighting that can be used by the read heads
    # for associative recall1 or by the write head to modify an existing vector in memory.
    def content_lookup(self, key, string):
        # The l2 norm of a vector is the square root of the sum of the
        # absolute values squared
        # applied to the memory
        norm_mem = tf.nn.l2_normalize(self.mem_mat, 1)  # N*W
        # applied to the key (emitted from the NN)
        norm_key = tf.nn.l2_normalize(key, 0)  # 1*W for write or R*W for read

        # get similarity measure between both vectors, transpose before multiplicaiton
        ##(N*W,W*1)->N*1 for write
        # (N*W,W*R)->N*R for read
        sim = tf.matmul(norm_mem, norm_key, transpose_b=True)
        # str is 1*1 or 1*R

        # returns similarity measure
        return tf.nn.softmax(sim * string, 0)  # N*1 or N*R

    #### SECOND DIFFERENTIABLE ATTENTION ####

    # retreives the writing allocation weighting based on the usage free list
    # The ‘usage’ of each location is represented as a number between 0 and 1,
    # and a weighting that picks out unused locations is delivered to the write head.

    # independent of the size and contents of the memory, meaning that
    # DNCs can be trained to solve a task using one size of memory and later

    # NOTE: upgraded to a larger memory without retraining
    def allocation_weighting(self):
        # sorted usage - the usage vector sorted ascendingly
        # the original indices of the sorted usage vector
        sorted_usage_vec, free_list = tf.nn.top_k(-1 * self.usage_vec, k=self.num_words)
        sorted_usage_vec *= -1
        # cumulative product
        cumprod = tf.cumprod(sorted_usage_vec, axis=0, exclusive=True)
        unorder = (1 - sorted_usage_vec) * cumprod

        alloc_weights = tf.zeros([self.num_words])
        I = tf.constant(np.identity(self.num_words, dtype=np.float32))

        # for each usage vec
        for pos, idx in enumerate(tf.unstack(free_list[0])):
            # flatten
            m = tf.squeeze(tf.slice(I, [idx, 0], [1, -1]))
            # add to weight matrix
            alloc_weights += m * unorder[0, pos]
        # the allocation weighting for each row in memory
        # define how best we want to allocate memory, so it's a matrix
        return tf.reshape(alloc_weights, [self.num_words, 1])

    # >>>>> STEP DEFINITION <<<<< #

    # at every time step the controller receives input vector from dataset and emits output vector.
    # it also recieves a set of read vectors from the memory matrix at the previous time step via
    # the read heads. then it emits an interface vector that defines its interactions with the memory
    # at the current time step
    def step_m(self, x):
        ### INPUT RESHAPE

        # the input is not just data, it's also the read from the memory at the previous step.
        #  the NN is a Feedforward one but in general is active recursevly due to the memory.
        #  We are not feeding the previous state of the network, but previous red data
        input = tf.concat(
            [
                x,  #  the dataset one
                tf.reshape(
                    self.read_vecs,  #  we concatenate the previous read
                    [1, (self.num_heads * self.word_size)]  # adapted to the size (1, heads*words)
                )
            ],
            1  # axis
        )

        ### FORWARD PASS

        # forward propagation
        # activation layer 1h
        l1_out = tf.matmul(input, self.W1) + self.b1
        #  output layer 1h
        l1_act = tf.nn.tanh(l1_out)
        # activation layer 2h
        l2_out = tf.matmul(l1_act, self.W2) + self.b2
        # output layer 2h
        l2_act = tf.nn.tanh(l2_out)

        # output vector
        self.nn_out = tf.matmul(l2_act, self.nn_out_weights)  # (1*eta+Y, eta+Y*Y)->(1*Y)

        # interaction vector - how to interact with memory
        self.interface_vec = tf.matmul(l2_act, self.interface_weights)  # (1*eta+Y, eta+Y*eta)->(1*eta)

        ### MEMORY MATRIX INTERACTION

        # the idea is to split the interface vector according to some strategy (take a big array and split
        #  it's more like dealing with PCB design, where encoding bits and concatenating them)
        partition = tf.constant(
            [[0] * (self.num_heads * self.word_size) + [1] * (self.num_heads) + [2] * (self.word_size) + [3] + \
             [4] * (self.word_size) + [5] * (self.word_size) + \
             [6] * (self.num_heads) + [7] + [8] + [9] * (self.num_heads * 3)], dtype=tf.int32)

        # convert interface vector into a set of read write vectors
        # using tf.dynamic_partitions(Partitions interface_vec into 10 tensors using indices from partition)
        (
            read_keys,
            read_str,
            write_key,
            write_str,
            erase_vec,
            write_vec,
            free_gates,
            alloc_gate,
            write_gate,
            read_modes
        ) = tf.dynamic_partition(
            self.interface_vec,
            partition,
            10
        )

        # MEMORY

        # read vectors (content address)
        read_keys = tf.reshape(read_keys, [self.num_heads, self.word_size])  # R*W
        #  helps to initialize the read
        # what does it do ?
        read_str = 1 + \
                   tf.nn.softplus(
                       tf.expand_dims(read_str, 0)
                   )  # 1*R

        # write vectors (content address)
        write_key = tf.expand_dims(write_key, 0)  # 1*W
        #  helps to initialize the write
        # what does it do ?
        write_str = 1 + \
                    tf.nn.softplus(
                        tf.expand_dims(write_str, 0)
                    )  # 1*1
        # overwrite
        erase_vec = tf.nn.sigmoid(tf.expand_dims(erase_vec, 0))  # 1*W
        write_vec = tf.expand_dims(write_vec, 0)  # 1*W

        # GATES INITIALIZATIONS

        # gate in the sense of define the degree of the operations (like LSTM or GRU networks)
        # this gates value, which are weights, are differentiables

        # the degree to which locations at read heads will be freed
        free_gates = tf.nn.sigmoid(tf.expand_dims(free_gates, 0))  # 1*R

        # the fraction of writing that is being allocated in a new location
        # dynamic memory allocation (not static)
        alloc_gate = tf.nn.sigmoid(alloc_gate)  # 1

        # the amount of information to be written to memory
        write_gate = tf.nn.sigmoid(write_gate)  # 1

        # READ MODES

        # the softmax distribution between the three read modes (backward, forward, lookup)
        # The read heads can use gates called read modes to switch between content lookup
        # using a read key and reading out locations either forwards or backwards
        # in the order they were written.
        read_modes = tf.nn.softmax(tf.reshape(read_modes, [3, self.num_heads]))  # 3*R

        ### WRITE FIRST

        # ALLOCATION DECISION

        # used to calculate usage vector, what's available to write to?
        retention_vec = tf.reduce_prod(1 - free_gates * self.read_weights, reduction_indices=1)
        # used to dynamically allocate memory
        self.usage_vec = (self.usage_vec + self.write_weights - self.usage_vec * self.write_weights) * retention_vec

        ##retreives the writing allocation weighting
        alloc_weights = self.allocation_weighting()  # N*1
        # where to write to??
        write_lookup_weights = self.content_lookup(write_key, write_str)  # N*1
        # define our write weights now that we know how much space to allocate for them and where to write to
        self.write_weights = write_gate * (alloc_gate * alloc_weights + (1 - alloc_gate) * write_lookup_weights)

        # write erase, then write to memory!
        self.mem_mat = self.mem_mat * (1 - tf.matmul(self.write_weights, erase_vec)) + \
                       tf.matmul(self.write_weights, write_vec)

        ### READING THEN

        # As well as writing, the controller can read from multiple locations in memory.
        # Memory can be searched based on the content of each location, or the associative
        # temporal links can be followed forward and backward to recall information written
        # in sequence or in reverse. (3rd attention mechanism)

        # updates and returns the temporal link matrix for the latest write
        # given the precedence vector and the link matrix from previous step
        nnweight_vec = tf.matmul(self.write_weights, tf.ones([1, self.num_words]))  # N*N
        self.link_mat = (1 - nnweight_vec - tf.transpose(nnweight_vec)) * self.link_mat + \
                        tf.matmul(self.write_weights, self.precedence_weight, transpose_b=True)
        self.link_mat *= tf.ones([self.num_words, self.num_words]) - tf.constant(
            np.identity(self.num_words, dtype=np.float32))

        # update of the precedence weights
        self.precedence_weight = (1 - tf.reduce_sum(self.write_weights, reduction_indices=0)) * \
                                 self.precedence_weight + self.write_weights

        #### THIRD DIFFERENTIABLE ATTENTION ####

        # 3 modes - forward, backward, content lookup
        forw_w = read_modes[2] * tf.matmul(self.link_mat, self.read_weights)  # (N*N,N*R)->N*R
        look_w = read_modes[1] * self.content_lookup(read_keys, read_str)  # N*R
        back_w = read_modes[0] * tf.matmul(self.link_mat, self.read_weights, transpose_a=True)  # N*R

        # use them to intiialize read weights
        self.read_weights = back_w + look_w + forw_w  # N*R

        # create read vectors by applying read weights to memory matrix
        self.read_vecs = tf.transpose(tf.matmul(self.mem_mat, self.read_weights, transpose_a=True))  # (W*N,N*R)^T->R*W

        # FINAL READ

        # multiply them together
        read_vec_mut = tf.matmul(tf.reshape(self.read_vecs, [1, self.num_heads * self.word_size]),
                                 self.read_vecs_out_weight)  # (1*RW, RW*Y)-> (1*Y)

        ### NN OUTPUT (ALL)

        # return output + read vecs product
        return self.nn_out + read_vec_mut


# output list of numbers (one hot encoded) by running the step function
def run(self):
    big_out = []
    for t, seq in enumerate(tf.unstack(self.i_data, axis=0)):
        seq = tf.expand_dims(seq, 0)
        y = self.step_m(seq)
        big_out.append(y)
    return tf.stack(big_out, axis=0)
