from deepmusic.moduleloader import ModuleLoader
from deepmusic.keyboardcell import KeyboardCell
import deepmusic.songstruct as music
import tqdm
import numpy as np
import tensorflow as tf
print(tf.__version__)
'''
using helper classes from https://github.com/Conchylicultor/MusicGenerator/tree/master/deepmusic
'''

def build_network(self):
  #graph creation/tensorflow session creation
  input_dim = ModuleLoader.batch_builders.get_module().get_input_dim()
  
  #notes
  with tf.name('placehoder_inputs'):
    self.inputs = [
      tf.placeholder(
          tf.float32, 
          [self.batch_size, input_dim], #data size
          name = 'input'
      )
    ]
  #88 key moment, target values
  with tf.name_scope('placeholder_targets'):
    self.targets = [
      tf.placeholder(
          tf.int32, #either 0 or 1
          [self.batch_size],
          name = 'target'
      )
    ]
  #hidden state
  with tf.name_scope('placeholder_use_prev'):
    self.use_prev = [
      tf.placeholder(
          tf.bool,
          [],
          name = 'use_prev'
      )
    ]
  #pog network moment we gon win
  self.loop_processing = ModuleLoader.loop_processings.build_module(self.args) #manual loop :(
  #feed output back into loop
  def loop_rnn(prev, i):
    next_input = self.loop_processing(prev)
    return tf.cond(self.prev[i], lambda: next_input, lambda: self.inputs[i])#tensorflow conditional object, conditional statement, returns 0/1

  self.outputs, self.final_state = tf.nn.seq2seq.rnn_decoder(
      decoder_inputs = self.inputs,
      initial_state = None,
      cell = KeyboardCell,
      loop_function = loop_rnn
  )
  
  #TRAINING POG
  loss_function = tf.nn.seq2seq.sequence_loss(
      self.outputs,
      self.targets,
      softmax_loss_function = tf.nn.softmax.cross_entropy_with_logits,
      average_across_timesteps = True,
      average_across_batch = True
  )

  #minimize loss
  opt = tf.train.AdamOptimizer(
      learning_rate = self.current_learning_rate,
      beta1 = 0.9,
      beta2 = 0.999,
      epsilon = 1e-08
  )

  self.opt_op = opt.minimize(loss_function)