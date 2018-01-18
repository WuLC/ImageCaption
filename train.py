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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # use the second GPU

import tensorflow as tf

import configuration
import show_and_tell_model


train_img_dir = '../data/aichallenge/TFRecordFile/'
input_file_pattern = train_img_dir + 'train-?????-of-00795'
inception_checkpoint_file = '../pretrained_models/inception_v3.ckpt'
vgg19_checkpoint_file = '../pretrained_models/vgg_19.ckpt'
inception_trained_models_dir = '../aichallenge_model_inception/train/'
vgg_trained_models_dir = '../aichallenge_model_vgg/train/'
# trained_models_dir = '../aichallenge_model_inception_with_custom_embedding/train/'
word_embedding_file = '../data/aichallenge/word_embedding/word_embedding.npy'
cnn_model = 'VGG19' # 'InceptionV3'
custom_word_embedding = False
train_cnn_model = False
num_steps = 2000000

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern", input_file_pattern,
                       "File pattern of sharded TFRecord input files.")             
tf.flags.DEFINE_string("cnn_model", cnn_model,
                       "choose which cnn model to use for image embedding, (currently InceptionV3 and VGG19 are available)")
tf.flags.DEFINE_boolean("train_cnn_model", train_cnn_model,
                        "Whether to train cnn submodel variables or not")
tf.flags.DEFINE_string("inception_checkpoint_file", inception_checkpoint_file,
                       "Path to a pretrained inception_v3 model.")
tf.flags.DEFINE_string("vgg19_checkpoint_file", vgg19_checkpoint_file,
                       "Path to a pretrained vgg19 model.")
tf.flags.DEFINE_boolean("custom_word_embedding", custom_word_embedding,
                       "Whether to use the word embedding file")
tf.flags.DEFINE_string("word_embedding_file", word_embedding_file,
                       "path of word embedding of words in the vocabulary trained with word2vec")
tf.flags.DEFINE_integer("number_of_steps", num_steps,
                        "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 50,
                        "Frequency at which loss and global step are logged.")

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
    model_config = configuration.ModelConfig()
    model_config.input_file_pattern = FLAGS.input_file_pattern
    model_config.inception_checkpoint_file = FLAGS.inception_checkpoint_file
    model_config.vgg19_checkpoint_file = FLAGS.vgg19_checkpoint_file
    model_config.word_embedding_file = FLAGS.word_embedding_file
    training_config = configuration.TrainingConfig()

    
    if cnn_model == 'InceptionV3':
        trained_models_dir = inception_trained_models_dir
    elif cnn_model == 'VGG19':
        trained_models_dir = vgg_trained_models_dir
    else:
        print('Unknown cnn model {0}'.format(cnn_model))
        exit(0)
    if not tf.gfile.IsDirectory(trained_models_dir):
        tf.logging.info("Creating training directory: %s", trained_models_dir)
        tf.gfile.MakeDirs(trained_models_dir)

    # Build the TensorFlow graph.
    g = tf.Graph()
    with g.as_default():
        # Build the model, train from scratch
        model = show_and_tell_model.ShowAndTellModel(
            model_config,
            mode="train",
            cnn_model = FLAGS.cnn_model,
            train_cnn_model=FLAGS.train_cnn_model,
            custom_word_embedding=FLAGS.custom_word_embedding)
        model.build()

        # Set up the learning rate.
        learning_rate_decay_fn = None
        if FLAGS.train_cnn_model:
            learning_rate = tf.constant(
                training_config.train_inception_learning_rate)
        else:
            learning_rate = tf.constant(training_config.initial_learning_rate)
            if training_config.learning_rate_decay_factor > 0:
                num_batches_per_epoch = (training_config.num_examples_per_epoch / model_config.batch_size)
                decay_steps = int(num_batches_per_epoch * training_config.num_epochs_per_decay)

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
        trained_models_dir,
        log_every_n_steps=FLAGS.log_every_n_steps,
        graph=g,
        global_step=model.global_step,
        number_of_steps=FLAGS.number_of_steps,
        init_fn=model.init_fn,
        saver=saver)


if __name__ == "__main__":
    tf.app.run()
