# -*- coding: utf-8 -*-
# @Author: lc
# @Date:   2017-09-17 11:31:14
# @Last Modified by:   WuLC
# @Last Modified time: 2017-09-19 15:50:06

# extract images for testing from validation set
# account for 5% of the validation images
# need to run with python2

import sys
import hashlib
import jieba
import json
from shutil import copyfile


def extract_test_data(val_img_dir, val_caption_file, test_img_dir, test_caption_file):
    test_data = {}
    info = { "contributor": "He Zheng", "description": "CaptionEval", "url": "https://github.com/AIChallenger/AI_Challenger.git", "version": "1", "year": 2017}
    licenses = [{ "url": "https://challenger.ai"}]
    type_ = "captions"
    test_data['annotations'] = []
    test_data['images'] = []
    test_data['info'] = info
    test_data['licenses'] = licenses
    test_data['type'] = type_
    count = 0
    train_percentage = 0.95 # percentage extracted from validation set as training set
    with open(val_caption_file, 'r') as f:
        data = json.load(f)
    for item in data[int(len(data) * train_percentage):]:
        filename = item['image_id'].split('.')[0]
        # copy validation images to another dir
        copyfile(val_img_dir + filename + '.png', test_img_dir + filename + '.png')
        captions = item['caption']
        image_id = int(int(hashlib.sha256(filename.encode('utf8')).hexdigest(), 16) % sys.maxint)
        for c in captions:
            count += 1
            annotation = {'caption': ' '.join(list(jieba.cut(c))), 'id': count, 'image_id': image_id}
            image = {'file_name': filename, 'id': image_id}
            test_data['annotations'].append(annotation)
            test_data['images'].append(image)
        if count % 500 == 0:
            print('===========finish {0}'.format(count))
    print('total {0} records'.format(count))
    with open(test_caption_file, 'w') as f:
        json.dump(test_data, f)

if __name__ == '__main__':
    val_img_dir = '../data/aichallenge/val20170911png/'
    test_img_dir = '../data/aichallenge/test1500png/'
    val_caption_file = '../data/aichallenge/annotations/captions_val20170911.json'
    test_caption_file = '../data/aichallenge/annotations/captions_7500test.json'
    extract_test_data(val_img_dir, val_caption_file, test_img_dir, test_caption_file)