# -*- coding: utf-8 -*-
# @Author: lc
# @Date:   2017-09-23 21:31:59
# @Last Modified by:   WuLC
# @Last Modified time: 2017-09-24 22:13:29

# get scores for result json files in a directory and plot the scores 
# requires python2

import os

import numpy as np
import matplotlib.pyplot as plt

from caption_eval.run_evaluations import get_judge_score

def get_batch_judge_score(reference_file, result_dir):
    files = sorted(os.listdir(result_dir))
    scores = [get_judge_score(result_dir + f, reference_file) for f in files]

    np.array(scores).dump('../data/aichallenge/scores/scores.npy')
    # draw and save image
    fig = plt.figure()
    plt.plot(scores)
    fig.savefig('../data/aichallenge/scores/scores.png')

    print('max score was achieved by file: {0}'.format(files[scores.index(max(scores))]))


if __name__ == '__main__':
    reference_file = '../data/aichallenge/annotations/captions_7500test.json'
    result_dir = '../data/aichallenge/result/'
    get_batch_judge_score(reference_file, result_dir)