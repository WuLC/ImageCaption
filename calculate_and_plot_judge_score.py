# -*- coding: utf-8 -*-
# @Author: lc
# @Date:   2017-09-23 21:31:59
# @Last Modified by:   LC
# @Last Modified time: 2017-09-30 16:58:53

# get scores for result json files in a directory and plot the scores 
# requires python2

import os
import logging

import numpy as np
import matplotlib.pyplot as plt

from caption_eval.run_evaluations import compute_m1

log_file = './logs/prediction_json_scores.log'
logging.basicConfig(level=logging.INFO, filename = log_file, filemode="a+", format="%(asctime)-15s %(levelname)-8s  %(message)s")


def get_batch_final_score(reference_file, result_dir):
    files = sorted(os.listdir(result_dir))
    print('totally {0} result json files'.format(len(files)))
    final_scores = []
    for f in files:
        score = compute_m1(result_dir + f, reference_file)
        final_score = (score['Bleu_4'] + score['CIDEr'] + score['METEOR'] + score['ROUGE_L'])/4.0
        final_scores.append(final_score)
        logging.info('{0} steps, final_score {1:.5f} {2:.5f} {3:.5f} {4:.5f} {5:.5f}'.format(
                     f.split('-')[1].rstrip('.json'),
                     final_score,
                     score['Bleu_4'],
                     score['CIDEr'],
                     score['METEOR'],
                     score['ROUGE_L']))
    np.savetxt('../data/aichallenge/scores/scores.dat', np.array(final_scores))
    # draw and save image
    fig = plt.figure()
    plt.plot(final_scores)
    fig.savefig('../data/aichallenge/scores/scores.png')

    print('max score {0} was achieved by file: {1}'.format(
          max(final_scores),
          files[final_scores.index(max(final_scores))]))


if __name__ == '__main__':
    reference_file = '../data/aichallenge/annotations/captions_7500test.json'
    result_dir = '../data/aichallenge/result/'
    get_batch_final_score(reference_file, result_dir)
    """
    predict_result = result_dir + 'result_model.ckpt-1202581.json'
    score = get_final_score(predict_result, reference_file)
    print(score)
    """
