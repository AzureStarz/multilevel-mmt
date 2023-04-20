import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from os.path import join, abspath, dirname, exists
from torch.autograd import Variable
from visual_grounding.model.grounding_model import *
from visual_grounding.utils.utils import *
from visual_grounding.utils.transforms import letterbox
# from transformers import BertTokenizer
from pytorch_pretrained_bert import BertTokenizer

root = join(dirname(dirname(abspath(__file__))), 'bishe')
vg = join(root, 'visual_grounding')
vis_path = join(root, 'region/vis')


def main():
    if not exists(vis_path):
        os.mkdir(vis_path)
    
    cudnn.benchmark = False
    cudnn.deterministic = True
    
    anchor_imsize = 416
    anchors = '29,26,  55,58,  137,71,  82,121,  124,205,  204,132,  209,263,  369,169,  352,294'
    anchors = [float(x) for x in anchors.split(',')]
    anchors_full = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)][::-1]

    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenizer = BertTokenizer.from_pretrained('./bert_weight', do_lower_case=True)

    model = grounding_model(corpus=None, light=False, emb_size=512, coordmap=True, bert_model='bert-base-uncased', dataset='flickr')
    model = torch.nn.DataParallel(model).cuda()

    pretrained_file = join(vg, 'saved_models/bert_flickr_model.pth.tar')
    pretrained_dict = torch.load(pretrained_file)['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()

    splits=['train', 'val', 'test_2016_flickr', 'test_2017_flickr', 'test_2017_mscoco']
    # splits=['val', 'test_2016_flickr', 'test_2017_flickr', 'test_2017_mscoco']
    # splits=['test_2017_mscoco']
    total_results =[]
    for split in splits:
        # dataset_path = join(root, 'dataset/multi30k')
        images_path = join(root, 'images/{split}'.format(split=split))
        image_splits = join(root, 'image_splits/{split}.txt'.format(split=split))

        with open(image_splits, 'r') as f:
            image_list = list(map(str.strip, f.readlines()))
        with open(join(root, 'phrase/np_{split}.json'.format(split=split))) as f:
            nps_list = json.load(f)

        results = []

        with torch.no_grad():
            for idx in range(len(image_list)):
                print('image: {0}/{1}'.format(idx, len(image_list)))
                image_path = join(images_path, image_list[idx])
                nps = nps_list[idx]['nps']
                # print(image_path)
                img = cv2.imread(image_path)
                imsize = 256
                if (img.shape[-1] > 1):
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img = np.stack([img] * 3)
                img, _, ratio, dw, dh = letterbox(img, None, imsize)
                img = input_transform(img).unsqueeze(0).cuda()
                dw = torch.tensor(dw, dtype=torch.float32).unsqueeze(0)
                dh = torch.tensor(dh, dtype=torch.float32).unsqueeze(0)
                ratio = torch.tensor(ratio, dtype=torch.float32).unsqueeze(0)

                subdir_path = join(vis_path, '{split}_%05d'.format(split=split) % idx)
                if not exists(subdir_path):
                    os.mkdir(subdir_path)

                for text in nps:
                    query_len = 128
                    tokens_tmp = tokenizer.tokenize(text['phrase'].lower())
                    if len(tokens_tmp) > query_len - 2:
                        tokens_tmp = tokens_tmp[:query_len - 2]
                    tokens = []
                    input_type_ids = []
                    tokens.append('[CLS]')
                    input_type_ids.append(0)
                    for token in tokens_tmp:
                        tokens.append(token)
                        input_type_ids.append(0)
                    tokens.append('[SEP]')
                    input_type_ids.append(0)
                    input_ids = tokenizer.convert_tokens_to_ids(tokens)
                    input_mask = [1] * len(input_ids)
                    while len(input_ids) < query_len:
                        input_ids.append(0)
                        input_mask.append(0)
                        input_type_ids.append(0)
                    input_ids = torch.tensor(input_ids, dtype=torch.int64).unsqueeze(0)
                    input_mask = torch.tensor(input_mask, dtype=torch.int64).unsqueeze(0)

                    input_ids = input_ids.cuda()
                    input_mask = input_mask.cuda()
                    pred_anchor = model(img, input_ids, input_mask)

                    for i in range(len(pred_anchor)):
                        pred_anchor[i] = pred_anchor[i].view(
                            pred_anchor[i].size(0), 3, 5, pred_anchor[i].size(2), pred_anchor[i].size(3)
                        )

                    pred_conf_list = []
                    for i in range(len(pred_anchor)):
                        pred_conf_list.append(pred_anchor[i][:, :, 4, :, :].contiguous().view(1, -1))
                    pred_conf = torch.cat(pred_conf_list, dim=1)
                    max_conf, max_loc = torch.max(pred_conf, dim=1)
                    if max_loc[0] < 3 * (imsize // 32) ** 2:
                        best_scale = 0
                    elif max_loc[0] < 3 * (imsize // 32) ** 2 + 3 * (imsize // 16) ** 2:
                        best_scale = 1
                    else:
                        best_scale = 2
                    grid, grid_size = imsize // (32 // (2 ** best_scale)), 32 // (2 ** best_scale)
                    anchor_idxs = [x + 3 * best_scale for x in range(0, 3)]
                    anchors = [anchors_full[i] for i in anchor_idxs]
                    scaled_anchors = [(x[0] / (anchor_imsize / grid), x[1] / (anchor_imsize / grid)) for x in anchors]
                    pred_conf = pred_conf_list[best_scale].view(1, 3, grid, grid).data.cpu().numpy()
                    max_conf_ii = max_conf.data.cpu().numpy()
                    best_n, gj, gi = np.where(pred_conf[0, :, :, :] == max_conf_ii[0])
                    best_n, gj, gi = int(best_n[0]), int(gj[0]), int(gi[0])

                    pred_bbox = torch.zeros(1, 4)
                    pred_bbox[0, 0] = F.sigmoid(pred_anchor[best_scale][0, best_n, 0, gj, gi]) + gi
                    pred_bbox[0, 1] = F.sigmoid(pred_anchor[best_scale][0, best_n, 1, gj, gi]) + gj
                    pred_bbox[0, 2] = torch.exp(pred_anchor[best_scale][0, best_n, 2, gj, gi]) * scaled_anchors[best_n][0]
                    pred_bbox[0, 3] = torch.exp(pred_anchor[best_scale][0, best_n, 3, gj, gi]) * scaled_anchors[best_n][1]
                    pred_bbox[0, :] = pred_bbox[0, :] * grid_size
                    pred_bbox = xywh2xyxy(pred_bbox)
                    pred_bbox[:, 0], pred_bbox[:, 2] = (pred_bbox[:, 0] - dw[0]) / ratio, (pred_bbox[:, 2] - dw[0]) / ratio
                    pred_bbox[:, 1], pred_bbox[:, 3] = (pred_bbox[:, 1] - dh[0]) / ratio, (pred_bbox[:, 3] - dh[0]) / ratio

                    top, bottom = round(float(dh) - 0.1), imsize - round(float(dh) + 0.1)
                    left, right = round(float(dw) - 0.1), imsize - round(float(dw) + 0.1)
                    img_np = img[0, :, top:bottom, left:right].data.cpu().numpy().transpose(1, 2, 0)

                    ratio = float(ratio)
                    new_shape = (round(img_np.shape[1] / ratio), round(img_np.shape[0] / ratio))
                    img_np = cv2.resize(img_np, new_shape, interpolation=cv2.INTER_CUBIC)
                    img_np = torch.from_numpy(img_np.transpose(2, 0, 1)).cuda().unsqueeze(0)

                    pred_bbox[:, :2], pred_bbox[:, 2], pred_bbox[:, 3] = \
                        torch.clamp(pred_bbox[:, :2], min=0), torch.clamp(pred_bbox[:, 2], max=img_np.shape[3]), torch.clamp(pred_bbox[:, 3], max=img_np.shape[2])

                    result = {
                        'image': image_list[idx],
                        'phrase': text['phrase'],
                        'head': text['head'],
                        'bbox': pred_bbox.squeeze(0).detach().numpy().tolist()
                    }
                    results.append(result)
                    total_results.append(result)

                    tmp_img = cv2.imread(image_path)
                    tmp_img = cv2.rectangle(tmp_img, (int(pred_bbox[0, 0]), int(pred_bbox[0, 1])), (int(pred_bbox[0, 2]), int(pred_bbox[0, 3])), (0, 255, 0), 2)
                    cv2.imwrite(join(subdir_path, text['phrase'] + '.jpg'), tmp_img)
        with open(join(root, 'region/bbox_{split}.json'.format(split=split)), 'w') as f:
            json.dump(results, f, indent=4)

    with open(join(root, 'region/bbox.json'), 'w') as f:
        json.dump(total_results, f, indent=4)


if __name__ == '__main__':
    main()