import json

splits = ['train', 'val', 'test_2016_flickr', 'test_2017_flickr', 'test_2017_mscoco']

results = []

for split in splits:
    
    with open('/userhome/zhanghb/bishe/region/bbox_{split}.json'.format(split=split)) as f:
        tmp = json.load(f)
        results.extend(tmp)

with open('/userhome/zhanghb/bishe/region/bbox.json', 'w') as f:
        json.dump(results, f, indent=4)