# Psuedo Label Generation

This document describes how to generate psuedo labels for open vocabulary/zero-shot object detection, which is introduced in the paper [Exploiting unlabeled data with vision and language models for object detection]().

**Prerequsite**: 
- Download the [COCO dataset](https://cocodataset.org/#home), and put it in the `datasets/` directory. 
- Download the pretrained class agnostic proposal generator [weight](https://drive.google.com/file/d/1ZDPrPGd5eyR62BZhjpHdZdOYPePVfPeO/view?usp=sharing) and put it as `./tools/mask_rcnn_R_50_FPN_1x_base_num1.pth`.
- Install [Pytorch]()(>=1.7), [Detectron2](), [CLIP](https://github.com/openai/CLIP), [COCO API](https://github.com/cocodataset/cocoapi), OpenCV, and tqdm.

## Evaluate PLs on COCO Validation Set

**Step 1**: Generate `inst_val2017_novel.json` which contains annotations of novel categories only. 
We remove images without any novel annotation, and there shold be 2064 images left.
```
cd ./tools
python split_coco_for_OVD.py '../datasets/coco/annotations/instances_val2017.json' './inst_val2017_novel.json'
```

**Step 2**: Generate `test_PL.json` which contains PLs. You may set `--thres` to other values to control the number of PLs.
```
python gen_PLs.py '../configs/mask_rcnn_R_50_FPN_1x_base_num1.yaml' './mask_rcnn_R_50_FPN_1x_base_num1.pth' --gt_json ./inst_val2017_novel.json --pl_save_file ./test_PL.json --thres 0.8 
```

**Step 3**: Evaluate PLs.
```
python offline_eval_onCOCO.py ./inst_val2017_novel.json ./test_PL.json
```
You may see the result as follows
```
Total PL boxes num: 10033, avg num: 4.95
loading annotations into memory...
Done (t=0.05s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.09s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=4.02s).
Accumulating evaluation results...
DONE (t=0.66s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.156
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.255
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.160
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.103
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.193
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.212
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.162
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.291
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.297
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.138
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.328
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.427
 ```

**Step 4 (optional)**: add text embedding into the json file for the evaluation of open vocabulary detectors. 
The postfix `_txtEmb` will be added in the input file name and results in the final file `test_PL_txtEmb.json`.
This will be used in the training.
```
python add_textEmb_cocojson.py ./test_PL.json
```


## Genearte PLs on COCO Training Set

To accelerate PL generation, we divide this process into two stages. 
In the first stage, we get CLIP scores for top 100 region proposals with multiple processes.
In the second stage, we generate PLs with pre-computed CLIP scores.
Additionally, to avoid data leakage, we should generate PLs for all images in the training set instead of those with novel objects.

**Stage 1**: Open the directory `tools/`.
```
cd ./tools
```
Run the following commands in different temrinals. The pre-computed CLIP scores will be stored in `--save_dir ./CLIP_scores_for_PLs`.
```
CUDA_VISIBLE_DEVICES=0 python get_CLIP_scores_for_PLs.py '../configs/mask_rcnn_R_50_FPN_1x_base_num1.yaml' './mask_rcnn_R_50_FPN_1x_base_num1.pth' --gt_json ../datasets/coco/annotations/instances_train2017.json --save_dir ./CLIP_scores_for_PLs --start 0 --end 30000
CUDA_VISIBLE_DEVICES=1 python get_CLIP_scores_for_PLs.py '../configs/mask_rcnn_R_50_FPN_1x_base_num1.yaml' './mask_rcnn_R_50_FPN_1x_base_num1.pth' --gt_json ../datasets/coco/annotations/instances_train2017.json --save_dir ./CLIP_scores_for_PLs --start 30000 --end 60000
CUDA_VISIBLE_DEVICES=2 python get_CLIP_scores_for_PLs.py '../configs/mask_rcnn_R_50_FPN_1x_base_num1.yaml' './mask_rcnn_R_50_FPN_1x_base_num1.pth' --gt_json ../datasets/coco/annotations/instances_train2017.json --save_dir ./CLIP_scores_for_PLs --start 60000 --end 90000
CUDA_VISIBLE_DEVICES=3 python get_CLIP_scores_for_PLs.py '../configs/mask_rcnn_R_50_FPN_1x_base_num1.yaml' './mask_rcnn_R_50_FPN_1x_base_num1.pth' --gt_json ../datasets/coco/annotations/instances_train2017.json --save_dir ./CLIP_scores_for_PLs --start 90000
```

**Stage 2**: Generate PL using pre-computed CLIP scores.
```
python gen_PLs_from_CLIP_scores.py --gt_json ../datasets/coco/annotations/instances_train2017.json --clip_score_dir ./CLIP_scores_for_PLs --pl_save_file ../datasets/coco/annotations/inst_train2017_basePL.json
```

**Finally**: Add text embedding into the json file for training or evaluation. The postfix `_txtEmb` will be added in the input file name and results in the final file `inst_train2017_basePL_txtEmb.json`. You may use `test_PL_txtEmb.json` in the last section as validation in the training.
```
python add_textEmb_cocojson.py ../datasets/coco/annotations/inst_train2017_basePL.json
```



