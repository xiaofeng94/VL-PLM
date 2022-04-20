import os, argparse
import json
from tqdm import tqdm

import numpy as np
import cv2
from PIL import Image, ImageDraw

import torch, torchvision

import clip

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
from detectron2.structures import Boxes

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils import COCO_BASE_CatName as baseCatNames
from utils import COCO_NOVEL_CatName as novelCatNames
from utils import get_coco_ids_by_order, detections2json
from utils import multiple_templates, build_text_embedding, scale_box
from utils import get_region_proposal, get_CLIP_pred_for_proposals, detection_postprocessing




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate PLs directly')
    parser.add_argument('RPN_config', type=str, help='config file for proposal network')
    parser.add_argument('RPN_weight_file', type=str, help='weight file for proposal network')

    parser.add_argument('--gt_json', type=str, default='./instances_val2017_novel.json',
                        help='GT coco json file. We only annotations of base categories')
    parser.add_argument('--pl_save_file', type=str, default='./inst_val2017_PLOnly.json',
                        help='PL coco json file to save')

    parser.add_argument('--coco_root', type=str, default='../datasets/coco', help='coco root dir')
    parser.add_argument('--roi_num', type=int, default=10, help='the number we repeat roi head')
    parser.add_argument('--clip_model', type=str, default='ViT-B/32', help='clip model type')
    parser.add_argument('--lamda', type=float, default=0.5, help='the weight of RPN scores in RPN and CLIP score fusion. CLIP score weight: 1 - lamda (default: 0.5)')
    parser.add_argument('--thres', type=float, default=0.8, help='threshold for score fusion (default: 0.8)')
    parser.add_argument('--nms_thres', type=float, default=0.6, help='threshold for NMS (default: 0.6)')

    parser.add_argument('--with_base', action='store_true')

    args = parser.parse_args()

    #############################################
    pp_topK = 100                           # only select topk proposals
    box_scalelist = [1, 1.5]

    roiBoxRepeat_num = args.roi_num
    clip_model_type = args.clip_model

    lamda = args.lamda                      # CLIP weight = 1 - lamda
    thres = args.thres
    pre_class_iou_thres = args.nms_thres

    orig_COCOJson_file = args.gt_json
    coco_root = args.coco_root
    save_json_file = args.pl_save_file

    # proposal network config & weights
    config_file = args.RPN_config
    weight_file = args.RPN_weight_file

    # novel categories
    usedCatNames = novelCatNames
    textEmbCatNames = usedCatNames                      # Novel
    # textEmbCatNames = usedCatNames + ['background']     # Novel + BG

    #############################################

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(1)

    ### load proposal network
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weight_file
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
    cfg.MODEL.RPN.NMS_THRESH = 0.3
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0  # set threshold for this model.  = 0.0 if all top100 boxes of RoI head are needed,
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 1.0
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000

    maskRCNN = build_model(cfg)
    DetectionCheckpointer(maskRCNN).load(cfg.MODEL.WEIGHTS)
    maskRCNN.eval()

    DataAug = T.ResizeShortestEdge(
                [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
            )

    ### load COCO
    # initialize COCO api for instance annotations
    coco = COCO(orig_COCOJson_file)

    imgIdsList = sorted(coco.getImgIds())
    print('total image num: ', len(imgIdsList))

    baseCatIds = coco.getCatIds(catNms=baseCatNames)
    print('base cat ids:', len(baseCatIds), baseCatIds)
    usedCatIds_inOrder = np.array(get_coco_ids_by_order(coco, usedCatNames))
    print('novel cat ids:', len(usedCatIds_inOrder), usedCatIds_inOrder)

    # load annotations for base categories
    data = json.load(open(orig_COCOJson_file, 'r'))

    new_annotations = list()

    if args.with_base:
        for anno in data["annotations"]:
            if anno['category_id'] in baseCatIds:
                new_annotations.append(anno)
        print('original annotations: %d' % (len(new_annotations),))

    ### load CLIP model
    print(clip.available_models())
    CLIPModel, preprocess = clip.load(clip_model_type, device=device)

    text_embed = build_text_embedding(CLIPModel, textEmbCatNames, multiple_templates)
    print('text_embed: ', text_embed.shape)

    ### Scoring
    new_anno_count = 0
    new_image_count = 0

    for img_id in tqdm(imgIdsList):
        new_image_count += 1

        # coco info
        imgInfo = coco.loadImgs(img_id)[0]
        cocoURLSplits = imgInfo['coco_url'].split('/')
        filePath = os.path.join(coco_root, cocoURLSplits[-2], imgInfo['file_name'])
        cvImg = cv2.imread(filePath)  # BGR

        # get region proposals
        proposal_boxes, pp_scores = get_region_proposal(cvImg, maskRCNN, DataAug=DataAug, roihead_num=roiBoxRepeat_num, topK_box=pp_topK)
        # get CLIP scores
        curBoxList, curRPNScoreList, curCLIPScoreList, curPredCOCOIdList = get_CLIP_pred_for_proposals(cvImg, proposal_boxes, pp_scores,
                                                                            CLIPModel, preprocess, text_embed, usedCatIds_inOrder,
                                                                            box_scalelist=box_scalelist, device=device)

        # curCLIPScoreList in the form of [[top1 score for box 1, top2 score for box 1, ...], [...], ...]
        # we simply take the top1 score for each box
        curCLIPScoreList = [x[0] for x in curCLIPScoreList]
        curPredCOCOIdList = [x[0] for x in curPredCOCOIdList]

        # merge CLIP and RPN scores
        curScoreList = list()
        for b_idx, box in enumerate(curBoxList):
            rpnScore = curRPNScoreList[b_idx]
            clipScores = curCLIPScoreList[b_idx]

            curScore = rpnScore * lamda + clipScores * (1 - lamda)
            curScoreList.append(curScore)

        # thresholding & NMS
        curBoxList, curScoreList, curPredCOCOIdList = detection_postprocessing(curBoxList, curScoreList, curPredCOCOIdList,
                                                                                thres=thres, pre_cls_nms_thres=pre_class_iou_thres)

        # convert PLs into COCO format
        for bidx in range(len(curBoxList)):
            currAnno = {'segmentation': [[]], 'area': 0, 'iscrowd': 0, 'image_id': img_id}

            curBox = curBoxList[bidx]
            x0, y0, x1, y1 = curBox  # xyxy
            box = [x0, y0, x1 - x0, y1 - y0]  # xywh

            curConf = curScoreList[bidx]
            catId_top1 = curPredCOCOIdList[bidx]

            currAnno['bbox'] = box
            currAnno['category_id'] = catId_top1
            currAnno['id'] = -(new_anno_count + 1)
            currAnno['confidence'] = curConf

            currAnno['thing_isBase'] = False
            currAnno['thing_isNovel'] = True if (catId_top1 in usedCatIds_inOrder) else False

            new_annotations.append(currAnno)
            new_anno_count += 1

    # save base gt + novel PL
    data['annotations'] = new_annotations

    print('save_json_file: ', save_json_file)
    with open(save_json_file, 'w') as outfile:
        json.dump(data, outfile)

    print('annotations: ', len(data['annotations']))
    print('new annotations: %d, avg: %.2f (%d)' % (new_anno_count, new_anno_count / new_image_count, new_image_count))
    print('Done!')


