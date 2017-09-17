# -*- coding: utf-8 -*-
# @Author: lc
# @Date:   2017-09-17 11:31:14
# @Last Modified by:   lc
# @Last Modified time: 2017-09-17 11:30:35


def extract_test_data(val_img_dir, val_caption_file, test_img_dir, test_caption_file):
    train_percentage = 0.95 # percentage extracted from validation set as training set
    test_data = {'annotations':[], 'images':[]}
    count = 0
    with open(val_caption_file, 'r') as f:
        data = json.load(f)
    for item in data[int(len(data) * train_percentage):]:
        filename = item['image_id'].split('.')[0]
        # copy validation images to another dir
        copyfile(val_img_dir + filename + '.png', test_img_dir + filename + '.png')
        captions = item['caption']
        image_id = int(int(hashlib.sha256(filename.encode('utf8')).hexdigest(), 16) % sys.maxsize)
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
    test_img_dir = '../data/aichallenge/test7500png/'
    val_caption_file = '../data/aichallenge/annotations/captions_val20170911.json'
    test_caption_file = '../data/aichallenge/annotations/captions_7500test.json'
    extract_test_data(val_img_dir, val_caption_file, test_img_dir, test_caption_file)