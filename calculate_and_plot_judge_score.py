# -*- coding: utf-8 -*-
# @Author: lc
# @Date:   2017-09-23 21:31:59
# @Last Modified by:   lc
# @Last Modified time: 2017-09-23 21:41:17

# get scores for result json files in a directory and plot the scores 
# requires python2

import os

from caption_eval.run_evaluations import get_judge_score

def get_batch_judge_score(reference_file, result_dir):
    indices, scores = [], []
    for filename in sorted(os.listdir(result_dir)):
        num = filename.lstrip('result').rstrip('.json')
        score = get_judge_score(result_dir + filename, reference_file)
        indices.append(indices)
        scores.append(score)

    # draw and save image
    fig = plt.figure()
    plt.plot(indices, scores)
    fig.savefig('../data/scores.png')


if __name__ == '__main__':
    reference_file = '../data/aichallenge/annotations/captions_7500test.json'
    result_dir = '../data/aichallenge/result/'
    get_batch_judge_score(reference_file, result_dir)