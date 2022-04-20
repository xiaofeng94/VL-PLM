# Psuedo Label Generation

This document describes how to generate psuedo labels for open vocabulary/zero-shot object detection, which is introduced in the paper [Exploiting unlabeled data with vision and language models for object detection]().
First of all, download the [COCO dataset ](https://cocodataset.org/#home), and put it in the `datasets/` directory.


## Evaluate PLs on COCO Validation Set

**Step 1**: Generate `inst_val2017_novel.json` which contains annotations of novel categories only. 
We remove images without any novel annotation, and there shold be 2064 images left.
```
cd ./tools
python split_coco_for_OVD.py '../datasets/coco/annotations/instances_val2017.json' './inst_val2017_novel.json'
```

**Step 2**: Download the pretrained two-stage class agnostic proposal generator [weight]() and put it as `./tools/mask_rcnn_R_50_FPN_1x_base_num1.pth`. 
Generate `test_PL.json` which contains PLs. You may set `--thres` to other values to control the number of PLs.
```
python gen_PLs.py '../configs/mask_rcnn_R_50_FPN_1x_base_num1.yaml' './mask_rcnn_R_50_FPN_1x_base_num1.pth' --gt_json ./inst_val2017_novel.json --pl_save_file ./test_PL.json --thres 0.8 
```

**Step 3**: Evaluate PLs
```
python offline_eval_onCOCO.py ./inst_val2017_novel.json ./test_PL.json
```


## Genearte PLs on COCO Training Set

To accelerate PL generation, we divide this process into two stages. 
In the first stage, we get CLIP scores for top 100 region proposals with multiple processes.
In the second stage, we generate PLs with pre-computed CLIP scores.

**Stage 1**: Run the following commands at the same time
```
cd ./tools
CUDA_VISIBLE_DEVICES=0 python get_CLIP_scores_for_PLs.py '../configs/mask_rcnn_R_50_FPN_1x_base_num1.yaml' './mask_rcnn_R_50_FPN_1x_base_num1.pth' --gt_json ../datasets/coco/annotations/instances_train2017.json --save_dir ./CLIP_scores_for_PLs --start 0 --end 30000
CUDA_VISIBLE_DEVICES=1 python get_CLIP_scores_for_PLs.py '../configs/mask_rcnn_R_50_FPN_1x_base_num1.yaml' './mask_rcnn_R_50_FPN_1x_base_num1.pth' --gt_json ../datasets/coco/annotations/instances_train2017.json --save_dir ./CLIP_scores_for_PLs --start 30000 --end 60000
CUDA_VISIBLE_DEVICES=2 python get_CLIP_scores_for_PLs.py '../configs/mask_rcnn_R_50_FPN_1x_base_num1.yaml' './mask_rcnn_R_50_FPN_1x_base_num1.pth' --gt_json ../datasets/coco/annotations/instances_train2017.json --save_dir ./CLIP_scores_for_PLs --start 60000 --end 90000
CUDA_VISIBLE_DEVICES=3 python get_CLIP_scores_for_PLs.py '../configs/mask_rcnn_R_50_FPN_1x_base_num1.yaml' './mask_rcnn_R_50_FPN_1x_base_num1.pth' --gt_json ../datasets/coco/annotations/instances_train2017.json --save_dir ./CLIP_scores_for_PLs --start 90000
```






