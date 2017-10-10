# encoding: utf-8
# Copyright 2017 challenger.ai
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
"""Evaluation utility for image Chinese captioning task."""
# __author__ = 'ZhengHe'
# python2.7
# python run_evaluations.py --submit=your_result_json_file --ref=reference_json_file

import sys
import argparse

reload(sys)
sys.setdefaultencoding('utf8')
from coco_caption.pycxtools.coco import COCO
from coco_caption.pycxevalcap.eval import COCOEvalCap


def compute_m1(json_predictions_file, reference_file):
    """Compute m1_score"""
    m1_score = {}
    m1_score['error'] = 0

    coco = COCO(reference_file)
    print('=======loading reference_file successfully')
    coco_res = coco.loadRes(json_predictions_file)
    print('=======loading json_prediction_file successfully')
    # create coco_eval object.
    coco_eval = COCOEvalCap(coco, coco_res)

    # evaluate results
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print '%s: %.3f'%(metric, score)
        m1_score[metric] = score
    return m1_score


def get_judge_score(json_predictions_file, reference_file):
    scores = compute_m1(json_predictions_file, reference_file)
    #judge_score = (scores['Bleu_4'] + scores['METEOR'] + scores['ROUGE_L'] + scores['CIDEr'])/4.0
    #print("judging score: {0}".format(judge_score))
    #return judge_score


if __name__ == "__main__":
    json_predictions_file = '../../data/aichallenge/result/result_model.ckpt-1257469.json'
    reference_file = '../../data/aichallenge/annotations/captions_7500test.json'
    get_judge_score(json_predictions_file, reference_file)
