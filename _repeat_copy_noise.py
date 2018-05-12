import numpy as np
import sonnet as snt
import tensorflow as tf
import collections
import _repeat_copy
import random

DatasetNoisedTensors = collections.namedtuple('DatasetNoisedTensors', ('observations',
                                                                       'target',
                                                                       'target_noise',
                                                                       'mask',
                                                                       'distortion'))


def custom_bit_string_readable(data, batch_size, model_output=None, whole_batch=False,
                               with_noise=False, with_distortion=False):
    """Produce a human readable representation of the sequences in data.

    Args:
      data: data to be visualised
      batch_size: size of batch
      model_output: optional model output tensor to visualize alongside data.
      whole_batch: whether to visualise the whole batch. Only a random sample of the
          batch will be visualized
      with_noise: decide to visualize or not the noised batch
      with_distortion: decide to print or not the distortion value computed as cosine
          similarity between the real target and the noised one

    Returns:
      A string used to visualise the data batch
    """

    def _readable(datum):
        return '+' + ' '.join(['-' if x == 0 else '%d' % x for x in datum]) + '+'

    obs_batch = data.observations
    targ_batch = data.target
    targ_noise_batch = data.target_noise
    dist_batch = data.distortion
    iterate_over = range(batch_size) if whole_batch else [random.randint(0, batch_size - 1)]

    batch_strings = []
    for batch_index in iterate_over:
        obs = obs_batch[:, batch_index, :]
        targ = targ_batch[:, batch_index, :]
        targ_n = targ_noise_batch[:, batch_index, :]
        dist = dist_batch[batch_index]

        obs_channels = range(obs.shape[1])
        targ_channels = range(targ.shape[1])
        obs_channel_strings = [_readable(obs[:, i]) for i in obs_channels]
        targ_channel_strings = [_readable(targ[:, i]) for i in targ_channels]
        targ_noise_channel_strings = [_readable(targ_n[:, i]) for i in targ_channels]

        readable_obs = 'Observations:\n' + '\n'.join(obs_channel_strings)
        readable_targ = 'Targs:\n' + '\n'.join(targ_channel_strings)
        readable_targ_n = 'Target_noise: \n' + '\n'.join(targ_noise_channel_strings)
        readable_dist = 'Distortion {}\n'.format(dist)

        if with_noise and with_distortion:
            strings = [readable_obs, readable_targ, readable_targ_n, readable_dist]
        elif with_distortion and not with_distortion:
            strings = [readable_obs, readable_targ, readable_dist]
        elif with_noise and not with_distortion:
            strings = [readable_obs, readable_targ, readable_targ_n]
        else:
            strings = [readable_obs, readable_targ]

        if model_output is not None:
            output = model_output[:, batch_index, :]
            output_strings = [_readable(output[:, i]) for i in targ_channels]
            strings.append('Model Output:\n' + '\n'.join(output_strings))

        batch_strings.append('\n\n'.join(strings))

    return '\n' + '\n\n\n\n'.join(batch_strings)


class NoisedRepeatCopy(_repeat_copy.RepeatCopy):

    def __init__(self, num_bits=6, batch_size=1, min_length=1, max_length=1, min_repeats=1, max_repeats=2, 
                 norm_max=10, log_prob_in_bits=False, time_average_cost=False, 
                 name='repeat_copy', noise_level=None):
        
        super(NoisedRepeatCopy, self).__init__(num_bits, batch_size, min_length, max_length, min_repeats, 
                                               max_repeats, norm_max, log_prob_in_bits, time_average_cost, name)

        self.noise_level = noise_level

    def _build(self):
        min_length, max_length = self._min_length, self._max_length
        min_reps, max_reps = self._min_repeats, self._max_repeats
        num_bits = self.num_bits
        batch_size = self.batch_size

        full_obs_size = num_bits + 2
        full_targ_size = num_bits + 1
        start_end_flag_idx = full_obs_size - 2
        num_repeats_channel_idx = full_obs_size - 1

        sub_seq_length_batch = tf.random_uniform(
            [batch_size], minval=min_length, maxval=max_length + 1, dtype=tf.int32)
        num_repeats_batch = tf.random_uniform(
            [batch_size], minval=min_reps, maxval=max_reps + 1, dtype=tf.int32)

        total_length_batch = sub_seq_length_batch * (num_repeats_batch + 1) + 3
        max_length_batch = tf.reduce_max(total_length_batch)
        residual_length_batch = max_length_batch - total_length_batch

        obs_batch_shape = [max_length_batch, batch_size, full_obs_size]
        targ_batch_shape = [max_length_batch, batch_size, full_targ_size]
        mask_batch_trans_shape = [batch_size, max_length_batch]

        obs_tensors = []
        targ_tensors = []
        # include also the noised version
        targ_noise_tensors = []
        mask_tensors = []
        # distortion
        distortion = []

        ### MODIFIED FROM HERE ######

        for batch_index in range(batch_size):
            sub_seq_len = sub_seq_length_batch[batch_index]
            num_reps = num_repeats_batch[batch_index]

            obs_pattern_shape = [sub_seq_len, num_bits]
            obs_pattern = tf.cast(
                tf.random_uniform(
                    obs_pattern_shape, minval=0, maxval=2, dtype=tf.int32),
                tf.float32)

            targ_pattern_shape = [sub_seq_len * num_reps, num_bits]
            flat_obs_pattern = tf.reshape(obs_pattern, [-1])
            flat_targ_pattern = tf.tile(flat_obs_pattern, tf.stack([num_reps]))

            # perturbation of the target
            def addNoise(x):
                #print('Added Noise'.format(batch_index))
                val = np.random.randint(1, 100)

                if (self.noise_level is not None) and (val > self.noise_level):
                    #print('Noise applied!', val)

                    def f1():
                        return tf.add(x, 1)

                    def f2():
                        return tf.add(x, -1)

                    output = tf.cond(tf.equal(x, tf.constant(1, dtype=tf.float32)), true_fn=f2, false_fn=f1)
                    return output   
                else:
                    return x

            # create a noised version of the target
            noised_flat_targ_pattern = tf.map_fn(addNoise, flat_targ_pattern)

            ### EVALUATE THE DISTORTION
            # compute how many bits are changed as a distortion metric
            distortion.append(
                tf.reduce_sum(
                    tf.square(
                        tf.subtract(
                            noised_flat_targ_pattern, flat_targ_pattern
                        )
                    )
                )
            )

            # reshape the tensors (both the original and the noised one)
            targ_pattern = tf.reshape(flat_targ_pattern, targ_pattern_shape)
            target_noise_pattern = tf.reshape(noised_flat_targ_pattern, targ_pattern_shape)

            ##### END MODIFICATIONS

            obs_flag_channel_pad = tf.zeros([sub_seq_len, 2])
            obs_start_flag = tf.one_hot(
                [start_end_flag_idx], full_obs_size, on_value=1., off_value=0.)
            num_reps_flag = tf.one_hot(
                [num_repeats_channel_idx],
                full_obs_size,
                on_value=self._normalise(tf.cast(num_reps, tf.float32)),
                off_value=0.)

            # note the concatenation dimensions.
            obs = tf.concat([obs_pattern, obs_flag_channel_pad], 1)
            obs = tf.concat([obs_start_flag, obs], 0)
            obs = tf.concat([obs, num_reps_flag], 0)

            # Now do the same for the targ_pattern (it only has one extra channel).
            targ_flag_channel_pad = tf.zeros([sub_seq_len * num_reps, 1])
            targ_end_flag = tf.one_hot(
                [start_end_flag_idx], full_targ_size, on_value=1., off_value=0.)
            targ = tf.concat([targ_pattern, targ_flag_channel_pad], 1)
            targ = tf.concat([targ, targ_end_flag], 0)

            ### INCLUDE THE NOISED ONE
            targ_noise = tf.concat([target_noise_pattern, targ_flag_channel_pad], 1)
            targ_noise = tf.concat([targ_noise, targ_end_flag], 0)
            ###

            # This aligns them s.t. the target begins as soon as the obs ends.
            obs_end_pad = tf.zeros([sub_seq_len * num_reps + 1, full_obs_size])
            targ_start_pad = tf.zeros([sub_seq_len + 2, full_targ_size])

            # The mask is zero during the obs and one during the targ.
            mask_off = tf.zeros([sub_seq_len + 2])
            mask_on = tf.ones([sub_seq_len * num_reps + 1])

            obs = tf.concat([obs, obs_end_pad], 0)
            targ = tf.concat([targ_start_pad, targ], 0)

            ### INCLUDE THE NOISED ONE
            targ_noise = tf.concat([targ_start_pad, targ_noise], 0)
            ###

            mask = tf.concat([mask_off, mask_on], 0)

            obs_tensors.append(obs)
            targ_tensors.append(targ)

            ### INCLUDE THE NOISED ONE
            targ_noise_tensors.append(targ_noise)
            ###

            mask_tensors.append(mask)

        # End the loop over batch index.
        # Compute how much zero padding is needed to make tensors sequences
        # the same length for all batch elements.
        residual_obs_pad = [
            tf.zeros([residual_length_batch[i], full_obs_size])
            for i in range(batch_size)
        ]
        residual_targ_pad = [
            tf.zeros([residual_length_batch[i], full_targ_size])
            for i in range(batch_size)
        ]
        residual_mask_pad = [
            tf.zeros([residual_length_batch[i]]) for i in range(batch_size)
        ]

        # Concatenate the pad to each batch element.
        obs_tensors = [
            tf.concat([o, p], 0) for o, p in zip(obs_tensors, residual_obs_pad)
        ]
        targ_tensors = [
            tf.concat([t, p], 0) for t, p in zip(targ_tensors, residual_targ_pad)
        ]

        ### INCLUDE THE NOISED ONE
        targ_noise_tensors = [
            tf.concat([t, p], 0) for t, p in zip(targ_noise_tensors, residual_targ_pad)
        ]
        ###

        mask_tensors = [
            tf.concat([m, p], 0) for m, p in zip(mask_tensors, residual_mask_pad)
        ]

        # Concatenate each batch element into a single tensor.
        obs = tf.reshape(tf.concat(obs_tensors, 1), obs_batch_shape)
        targ = tf.reshape(tf.concat(targ_tensors, 1), targ_batch_shape)

        ### INCLUDE THE NOISED ONE
        targ_noise = tf.reshape(tf.concat(targ_noise_tensors, 1), targ_batch_shape)
        ###

        mask = tf.transpose(
            tf.reshape(tf.concat(mask_tensors, 0), mask_batch_trans_shape))
        # return the collection including the noised one
        return DatasetNoisedTensors(obs, targ, targ_noise, mask, distortion)

    def to_human_readable(self, data, model_output=None, whole_batch=False, with_distortion=False, 
                          with_noise=False):
        obs = data.observations
        # it has to denormalize the value associated with the channel for the number of repetitions
        # that is why it takes the last row
        unnormalised_num_reps_flag = self._unnormalise(obs[:, :, -1:]).round()
        # rebuild the original one with the unormalized values
        obs = np.concatenate([obs[:, :, :-1], unnormalised_num_reps_flag], axis=2)
        data = data._replace(observations=obs)
        return custom_bit_string_readable(data, self.batch_size, model_output, whole_batch, with_noise, 
                                          with_distortion)
