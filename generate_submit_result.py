# -*- coding: utf-8 -*-
# @Author: lc
# @Date:   2017-09-17 00:22:17
# @Last Modified by:   WuLC
# @Last Modified time: 2017-09-30 22:10:20

import sys
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"] = '0' # decide to use CPU or GPU

import json
import tensorflow as tf

import configuration
import inference_wrapper
from inference_utils import caption_generator
from inference_utils import vocabulary

checkpoint_path = '../aichallenge_model_inception_with_custom_embedding/train/'
vocab_file = '../data/aichallenge/TFRecordFile/word_counts.txt'
test_img_dir = '../data/aichallenge/test20170923png/'
submit_json_dir = '../data/aichallenge/submit/'

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("checkpoint_path", checkpoint_path, 
                       "Model checkpoint file or directory containing a model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", vocab_file, "Text file containing the vocabulary.")
tf.flags.DEFINE_string("test_img_dir", test_img_dir, 
                       "directory containing images for test")
tf.flags.DEFINE_string("submit_json_dir", submit_json_dir, 
                       "directory containing json file for the result of the test images")


def main(_):
    start_time = time.time()
    # Create the vocabulary.
    vocab = vocabulary.Vocabulary(FLAGS.vocab_file)
    # remove Thumbs.db from files
    filenames = [f for f in os.listdir(FLAGS.test_img_dir) if f.endswith('png')]
    print('There are totally {0} images.....'.format(len(filenames)))

    # with embedding:  result_model.ckpt-487261.json
    checkpoint_file = FLAGS.checkpoint_path + 'model.ckpt-487261' #'model.ckpt-1188726'
    submit_json_file = '{0}submit_{1}_custom_embedding.json'.format(FLAGS.submit_json_dir, checkpoint_file.split('/')[-1])
  

    g = tf.Graph()
    with g.as_default():
        model = inference_wrapper.InferenceWrapper()
        restore_fn = model.build_graph_from_config(configuration.ModelConfig(), checkpoint_file)
    g.finalize()

    with tf.Session(graph=g) as sess:
        # Load the model from checkpoint.
        restore_fn(sess)

        # Prepare the caption generator. Here we are implicitly using the default
        # beam search parameters. See caption_generator.py for a description of the
        # available beam search parameters.
        count, result = 0, []
        generator = caption_generator.CaptionGenerator(model, vocab)
        for filename in filenames:
            count += 1
            with open(FLAGS.test_img_dir + filename, "rb") as f:
                image = f.read()
            captions = generator.beam_search(sess, image)
            sentence = [vocab.id_to_word(w) for w in captions[0].sentence[1:-1]]
            sentence = ''.join(sentence)
            image_id = filename.split('.')[0]
            result.append({'caption': sentence, 'image_id':image_id})
            if count % 500 == 0:
                print('finish generating caption for {0} images'.format(count))
        print('finish totally {0} images'.format(count))
        with open(submit_json_file, encoding = 'utf8', mode = 'w') as f:
            json.dump(result, f, ensure_ascii = False)
        print('time consuming: {0}s'.format(time.time() - start_time))


if __name__ == "__main__":
    tf.app.run()


