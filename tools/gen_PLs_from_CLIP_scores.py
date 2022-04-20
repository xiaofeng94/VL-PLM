import json
import os, glob, argparse

import numpy as np
import torch, torchvision

from pycocotools.coco import COCO

from utils import COCO_BASE_CatName as baseCatNames
from utils import COCO_NOVEL_CatName as novelCatNames
from utils import  detection_postprocessing


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate PLs with pre-computed CLIP scores.')
    parser.add_argument('--gt_json', type=str, default='../datasets/coco/annotations/instances_train2017.json', help='gt coco json file. We only annotations of base categories')
    parser.add_argument('--clip_score_dir', type=str, default='./CLIP_scores_for_PLs', help='dir to save intermediate CLIP scores')
    parser.add_argument('--pl_save_file', type=str, default='../datasets/coco/annotations/inst_train2017_basePL.json',
                        help='PL coco json file')

    parser.add_argument('--lamda', type=float, default=0.5, help='the weight of RPN scores in RPN and CLIP score fusion. CLIP score weight: 1 - lamda (default: 0.5)')
    parser.add_argument('--thres', type=float, default=0.8, help='threshold for score fusion (default: 0.8)')
    parser.add_argument('--nms_thres', type=float, default=0.6, help='threshold for NMS (default: 0.6)')

    args = parser.parse_args()

    ###################################p
    orig_json_file = args.gt_json
    rec_json_dir = args.clip_score_dir
    save_json_file = args.pl_save_file

    mean_thres = args.thres
    lamda = args.lamda
    nms_iou_thres = args.nms_thres

    ###################################

    coco = COCO(orig_json_file)

    ## category names
    baseCatIds = coco.getCatIds(catNms=baseCatNames)
    novelCatIds = coco.getCatIds(catNms=novelCatNames)
    print('baseCatIds:', len(baseCatIds), baseCatIds)
    print('novelCatIds:', len(novelCatIds), novelCatIds)

    jsonFileList = sorted(glob.glob(os.path.join(rec_json_dir, '*.json')))
    print('jsonFile num: ', len(jsonFileList))

    # load annotations for base categories
    data = json.load(open(orig_json_file, 'r'))

    new_annotations = list()

    for anno in data["annotations"]:
        if anno['category_id'] in baseCatIds:
            new_annotations.append(anno)
    print('original annotations: %d' % (len(new_annotations),))

    # load pseudo annotations for other categories
    new_anno_count = 0
    new_image_count = 0
    for jsonFile in jsonFileList:
        print('-- %s' % jsonFile)

        CLIPScoreData = json.load(open(jsonFile, 'r'))

        imgId_list = CLIPScoreData['img_ids_list']
        bbox_all_list = CLIPScoreData['bbox_all_list']
        rpn_score_all_list = CLIPScoreData['rpn_score_all_list']
        clip_score_all_list = CLIPScoreData['clip_score_all_list']
        clip_catIDs_all_list = CLIPScoreData['clip_catIDs_all_list']

        new_image_count += len(imgId_list)

        for iidx, curImgIdx in enumerate(imgId_list):
            curBoxList = bbox_all_list[iidx]
            curRPNScoreList = rpn_score_all_list[iidx]
            curCLIPScoreList = clip_score_all_list[iidx]
            curCatIDsList = clip_catIDs_all_list[iidx]

            # curCLIPScoreList in the form of [[top1 score for box 1, top2 score for box 1, ...], [...], ...]
            # we simply take the top1 score for each box
            curCLIPScoreList = [x[0] for x in curCLIPScoreList]
            curPredCOCOIdList = [x[0] for x in curCatIDsList]

            # merge CLIP and RPN scores
            curScoreList = list()
            for b_idx, box in enumerate(curBoxList):
                rpnScore = curRPNScoreList[b_idx]
                clipScores = curCLIPScoreList[b_idx]

                curScore = rpnScore * lamda + clipScores * (1 - lamda)
                curScoreList.append(curScore)

            # thresholding & NMS
            ensure_boxlist, ensure_mScorelist, clip_catIDs_list = detection_postprocessing(curBoxList, curScoreList, curPredCOCOIdList,
                                                                                    thres=mean_thres, pre_cls_nms_thres=nms_iou_thres)

            # convert PLs into COCO format
            for bidx in range(len(ensure_boxlist)):
                currAnno = {'segmentation': [[]], 'area': 0, 'iscrowd': 0, 'image_id': curImgIdx}

                curBox = ensure_boxlist[bidx]
                x0, y0, x1, y1 = curBox             # xyxy
                box = [x0, y0, x1 - x0, y1 - y0]    # xywh

                curConf = ensure_mScorelist[bidx]
                catId_top1 = clip_catIDs_list[bidx]

                currAnno['bbox'] = box
                currAnno['category_id'] = catId_top1
                currAnno['id'] = -(new_anno_count + 1)
                currAnno['confidence'] = curConf

                currAnno['thing_isBase'] = False
                currAnno['thing_isNovel'] = True if (catId_top1 in novelCatIds) else False

                new_annotations.append(currAnno)
                new_anno_count += 1

        print('curr annotations: %d' % (len(new_annotations),))

    data['annotations'] = new_annotations

    print('save_json_file: ', save_json_file)
    with open(save_json_file, 'w') as outfile:
        json.dump(data, outfile)

    print('annotations: ', len(data['annotations']))
    print('new annotations: %d, avg: %.2f (%d)' % (new_anno_count, new_anno_count / new_image_count, new_image_count))
    print('Done!')







