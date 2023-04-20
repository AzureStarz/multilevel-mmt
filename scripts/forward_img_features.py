import os
import cv2
import sys
import glob
import json
import torch
import argparse
import numpy as np
from os.path import abspath, dirname, join

root = join(dirname(dirname(abspath(__file__))), 'bishe')
detectron_path = join(root, 'detectron2')
model_weights_path = join(detectron_path, 'model_weights')
region_files = join(root, 'region')

sys.path.insert(0, join(root, 'detectron2'))

def load_orig_images(images_root, split):
    if split == 'test':
        split = 'test_2016_flickr'
    elif split == 'test1':
        split = 'test_2017_flickr'
    elif split == 'test2':
        split = 'test_2017_mscoco'
    parent_path = abspath(join(images_root, ".."))
    images_root = join(images_root, split)
    images_split_root = join(parent_path, 'image_splits')
    index = join(images_split_root, '{split}.txt'.format(split=split))
    if not exists(index):
        raise(RuntimeError("{0}.txt does not exist in {1}".format(split, images_split_root)))

    image_files = []
    with open(index, 'r') as f:
        for fname in f:
            fname = join(images_root, fname.strip())
            assert exists(fname), "{} does not exist.".format(fname)
            image_files.append(str(fname))
    return image_files

def read_image(fnames):
    imgs = []
    print('{} loading images'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    for fname in fnames:
        with open(fname, 'rb') as f:
            img = Image.open(f).convert('RGB')
            imgs.append(img)
    print('{} END of loading images'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    return imgs

def main():
    args = get_parser().parse_args()
    cfg = setup_cfg(args)

    model = build_model(cfg)
    model.eval()
    weights = os.path.join(model_weights_path, 'model_final_a2914c.pkl')
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(weights)

    aug = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )
    input_format = cfg.INPUT.FORMAT

    results = []

    def hook_fn_forward(module, input, output):
        feature = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1))
        results.append(feature[0, :, 0, 0].cpu().numpy().tolist())
    
    model.roi_heads.res5[2].conv3.register_forward_hook(hook_fn_forward)
    
    splits=['train', 'val', 'test_2016_flickr', 'test_2017_flickr', 'test_2017_mscoco']
    # splits=['val', 'test_2016_flickr', 'test_2017_flickr', 'test_2017_mscoco']
    
    for split in splits:
        images_path = join(root, 'images/{split}'.format(split=split))
        with open(join(region_files, 'bbox_{split}.json'.format(split=split)), 'r') as f:
            bboxes = json.load(f)

        with torch.no_grad():
            chunk_size = 10000
            for chunk_id in range(len(bboxes) // chunk_size + 1):
                begin = chunk_id * chunk_size
                end = min((chunk_id + 1) * chunk_size, len(bboxes))
                results = []
                for idx in range(begin, end):
                    bbox = bboxes[idx]
                    print('{0}/{1}'.format(idx, len(bboxes)))
                    image_path = join(images_path, bbox['image'])
                    original_image = read_image(image_path, format='BGR')
                    pred_boxes = Boxes(torch.tensor(bbox['bbox']).unsqueeze(0))
                    pred_classes = torch.zeros(pred_boxes.tensor.size(0), dtype=torch.long)

                    if input_format == 'RGB':
                        original_image = original_image[:, :, ::-1]
                    height, width = original_image.shape[:2]
                    image = aug.get_transform(original_image).apply_image(original_image)
                    image = torch.as_tensor(image.astype('float32').transpose(2, 0, 1))

                    inputs = {'image': image, 'height': height, 'width': width}
                    detected_instances = Instances((height, width), pred_boxes=pred_boxes, pred_classes=pred_classes)
                    predictions = model.inference([inputs], detected_instances=[detected_instances])[0]
        
                results = np.array(results)
                filename = 'bbox_{split}_features_faster_rcnn_R_101_'.format(split=split) + '%02d' % chunk_id + '.npy'
                np.save(join(region_files, filename), results)


if __name__ == '__main__':
    main()