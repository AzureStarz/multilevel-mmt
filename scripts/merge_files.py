import os
import numpy as np

from os.path import join, abspath, dirname

root = join(dirname(dirname(abspath(__file__))), 'bishe')
files = join(root, 'region')

def main():
    all_features = []
    splits=['val', 'test_2016_flickr', 'test_2017_flickr', 'test_2017_mscoco']
    # splits=['train', 'val', 'test_2016_flickr', 'test_2017_flickr', 'test_2017_mscoco']
    for i in range(11):
        path = join(files, 'bbox_train_features_faster_rcnn_R_101_' + ('%02d' % i) + '.npy')
        print('loading {0}/10...'.format(i))
        features = np.load(path)
        all_features.extend(features.tolist())
    for split in splits:
        path = join(files, 'bbox_{0}_features_faster_rcnn_R_101_00.npy'.format(split))
        print('loading 1/1...')
        features = np.load(path)
        all_features.extend(features.tolist())
    all_features = np.array(all_features)
    print(all_features.shape)
    np.save(join(files, 'bbox_all_features_faster_rcnn_R_101.npy'), all_features)


if __name__ == '__main__':
    main()