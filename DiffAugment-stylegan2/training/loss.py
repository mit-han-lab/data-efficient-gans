# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Loss functions."""

import tensorflow as tf
from dnnlib.tflib.autosummary import autosummary
from DiffAugment_tf import DiffAugment


def ns_DiffAugment_r1(G, D, training_set, minibatch_size, reals, gamma=10, policy='', **kwargs):
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fakes = G.get_output_for(latents, labels, is_training=True)

    real_scores = D.get_output_for(DiffAugment(reals, policy=policy, channels_first=True), is_training=True)
    fake_scores = D.get_output_for(DiffAugment(fakes, policy=policy, channels_first=True), is_training=True)
    real_scores = autosummary('Loss/scores/real', real_scores)
    fake_scores = autosummary('Loss/scores/fake', fake_scores)

    G_loss = tf.nn.softplus(-fake_scores)
    G_loss = autosummary('Loss/G_loss', G_loss)
    D_loss = tf.nn.softplus(fake_scores) + tf.nn.softplus(-real_scores)
    D_loss = autosummary('Loss/D_loss', D_loss)

    with tf.name_scope('GradientPenalty'):
        real_grads = tf.gradients(tf.reduce_sum(real_scores), [reals])[0]
        gradient_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1, 2, 3])
        gradient_penalty = autosummary('Loss/gradient_penalty', gradient_penalty)
        D_reg = gradient_penalty * (gamma * 0.5)
    return G_loss, D_loss, D_reg


def ns_r1_DiffAugment(G, D, training_set, minibatch_size, reals, gamma=10, policy='', **kwargs):
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fakes = G.get_output_for(latents, labels, is_training=True)

    reals = DiffAugment(reals, policy=policy, channels_first=True)
    fakes = DiffAugment(fakes, policy=policy, channels_first=True)
    real_scores = D.get_output_for(reals, is_training=True)
    fake_scores = D.get_output_for(fakes, is_training=True)
    real_scores = autosummary('Loss/scores/real', real_scores)
    fake_scores = autosummary('Loss/scores/fake', fake_scores)

    G_loss = tf.nn.softplus(-fake_scores)
    G_loss = autosummary('Loss/G_loss', G_loss)
    D_loss = tf.nn.softplus(fake_scores) + tf.nn.softplus(-real_scores)
    D_loss = autosummary('Loss/D_loss', D_loss)

    with tf.name_scope('GradientPenalty'):
        real_grads = tf.gradients(tf.reduce_sum(real_scores), [reals])[0]
        gradient_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1, 2, 3])
        gradient_penalty = autosummary('Loss/gradient_penalty', gradient_penalty)
        D_reg = gradient_penalty * (gamma * 0.5)
    return G_loss, D_loss, D_reg

# ----------------------------------------------------------------------------
