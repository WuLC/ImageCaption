# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train the model."""

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"] = '1' # use the second GPU

import tensorflow as tf

import configuration
import show_and_tell_model


train_img_dir = '../data/aichallenge/TFRecordFile/'
input_file_pattern = train_img_dir + 'train-?????-of-00795'
inception_checkpoint_file = '../pretrained_models/inception_v3.ckpt'
train_dir = '../aichallenge_model_inception/train/'
word_embedding_file = '../aichallenge_model_inception/word_vector.npy'
train_inception = False
num_steps = 1500000

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern", input_file_pattern,
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("inception_checkpoint_file", inception_checkpoint_file,
                       "Path to a pretrained inception_v3 model.")
tf.flags.DEFINE_string("train_dir", train_dir,
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_string("word_embedding_file", word_embedding_file,
                       "word embedding of words in the vocabulary obtained by word2vec")
tf.flags.DEFINE_boolean("train_inception", train_inception,
                        "Whether to train inception submodel variables.")
tf.flags.DEFINE_integer("number_of_steps", num_steps, "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 50,
                        "Frequency at which loss and global step are logged.")

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
  assert FLAGS.input_file_pattern, "--input_file_pattern is required"
  assert FLAGS.train_dir, "--train_dir is required"

  model_config = configuration.ModelConfig()
  model_config.input_file_pattern = FLAGS.input_file_pattern
  model_config.inception_checkpoint_file = FLAGS.inception_checkpoint_file
  model_config.word_embedding_file = FLAGS.word_embedding_file
  training_config = configuration.TrainingConfig()

  # Create training directory.
  train_dir = FLAGS.train_dir
  if not tf.gfile.IsDirectory(train_dir):
    tf.logging.info("Creating training directory: %s", train_dir)
    tf.gfile.MakeDirs(train_dir)

  # Build the TensorFlow graph.
  g = tf.Graph()
  with g.as_default():
    # Build the model, train from scratch
    model = show_and_tell_model.ShowAndTellModel(
        model_config, mode="train", train_inception=FLAGS.train_inception)
    model.build()
    
    # Set up the learning rate.
    learning_rate_decay_fn = None
    if FLAGS.train_inception:
      learning_rate = tf.constant(training_config.train_inception_learning_rate)
    else:
      learning_rate = tf.constant(training_config.initial_learning_rate)
      if training_config.learning_rate_decay_factor > 0:
        num_batches_per_epoch = (training_config.num_examples_per_epoch /
                                 model_config.batch_size)
        decay_steps = int(num_batches_per_epoch *
                          training_config.num_epochs_per_decay)

        def _learning_rate_decay_fn(learning_rate, global_step):
          return tf.train.exponential_decay(
              learning_rate,
              global_step,
              decay_steps=decay_steps,
              decay_rate=training_config.learning_rate_decay_factor,
              staircase=True)

        learning_rate_decay_fn = _learning_rate_decay_fn

    # Set up the training ops.
    train_op = tf.contrib.layers.optimize_loss(
        loss=model.total_loss,
        global_step=model.global_step,
        learning_rate=learning_rate,
        optimizer=training_config.optimizer,
        clip_gradients=training_config.clip_gradients,
        learning_rate_decay_fn=learning_rate_decay_fn)

    # Set up the Saver for saving and restoring model checkpoints.
    saver = tf.train.Saver(max_to_keep=training_config.max_checkpoints_to_keep)

  # Run training.
  tf.contrib.slim.learning.train(
      train_op,
      train_dir,
      log_every_n_steps=FLAGS.log_every_n_steps,
      graph=g,
      global_step=model.global_step,
      number_of_steps=FLAGS.number_of_steps,
      init_fn=model.init_fn,
      saver=saver)


if __name__ == "__main__":
  tf.app.run()
