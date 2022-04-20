import os, argparse
import json
from tqdm import tqdm, trange

import numpy as np
import cv2
from PIL import Image, ImageDraw

import torch, torchvision
from  torchvision import transforms

import clip

import detectron2
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
from utils import get_region_proposal, get_CLIP_pred_for_proposals


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-compute CLIP scores for PL generation.')
    parser.add_argument('RPN_config', type=str, help='config file for proposal network')
    parser.add_argument('RPN_weight_file', type=str, help='weight file for proposal network')

    parser.add_argument('--gt_json', type=str, default='../datasets/coco/annotations/instances_train2017.json',
                        help='GT coco json file. We only annotations of base categories')
    parser.add_argument('--save_dir', type=str, default='./CLIP_scores_for_PLs',
                        help='PL coco json file to save')

    parser.add_argument('--coco_root', type=str, default='../datasets/coco', help='coco root dir')
    parser.add_argument('--roi_num', type=int, default=10, help='start index')
    parser.add_argument('--clip_model', type=str, default='ViT-B/32', help='clip model type')

    parser.add_argument('--start', type=int, default=0, help='start image index')
    parser.add_argument('--end', type=int, default=None, help='end image index')

    args = parser.parse_args()

    ##############################
    pp_topK = 100  # only select topk proposals
    box_scalelist = [1, 1.5]
    topK_clip_scores = 1 # save topK_clip_scores clip predictions

    roiBoxRepeat_num = args.roi_num
    clip_model_type = args.clip_model

    imge_start_idx = args.start
    img_end_idx = args.end

    # coco json
    orig_COCOJson_file = args.gt_json
    coco_root = args.coco_root
    # save results in the folder <rec_save_root> as the name <rec_json_name>
    rec_save_root = args.save_dir
    rec_json_name = 'CLIP_scores'

    # proposal network config & weights
    config_file = args.RPN_config
    weight_file = args.RPN_weight_file

    # novel categories
    usedCatNames = novelCatNames
    textEmbCatNames = usedCatNames                        # Novel
    # textEmbCatNames = usedCatNames + ['background']     # Novel + BG

    ##############################

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(1)

    if not os.path.exists(rec_save_root):
        os.makedirs(rec_save_root)

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

    usedCatIds_inOrder = np.array(get_coco_ids_by_order(coco, usedCatNames))
    print('used cat ids:', len(usedCatIds_inOrder), usedCatIds_inOrder)

    # load CLIP model
    print(clip.available_models())
    CLIPModel, preprocess = clip.load(clip_model_type, device=device)

    text_embed = build_text_embedding(CLIPModel, textEmbCatNames, multiple_templates)
    print('text_embed: ', text_embed.shape)

    ### CLIP scoring
    imgIdList = list()
    boxAllList = list()
    rpnScoreAlllist = list()
    scoreAllList = list()
    cocoIDAllList = list()

    used_image_ids = imgIdsList[:img_end_idx]
    for iidx in trange(len(used_image_ids)):
        if iidx < imge_start_idx:
            continue

        img_id = used_image_ids[iidx]
        imgIdList.append(img_id)

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
                                                                            box_scalelist=box_scalelist, topK_clip_scores=topK_clip_scores,
                                                                            device=device)

        # add to final results
        boxAllList.append(curBoxList)
        rpnScoreAlllist.append(curRPNScoreList)
        scoreAllList.append(curCLIPScoreList)
        cocoIDAllList.append(curPredCOCOIdList)

        # save current data to file
        if (iidx + 1) % 1000 == 0 or iidx >= len(used_image_ids)-1:
            data = {'img_ids_list': imgIdList,
                    'bbox_all_list': boxAllList,
                    'rpn_score_all_list': rpnScoreAlllist,
                    'clip_score_all_list': scoreAllList,
                    'clip_catIDs_all_list': cocoIDAllList
                    }

            # write to json
            recJsonFile = os.path.join(rec_save_root, '%s_%d.json' % (rec_json_name, iidx))
            with open(recJsonFile, 'w') as outfile:
                json.dump(data, outfile, separators=(',', ':'))

            # clear all lists
            imgIdList = list()  # [idx 1, idx 2, idx 3, ...]
            boxAllList = list()  # [bboxes list (nx4), bboxes (nx4), ...]
            rpnScoreAlllist = list()  # [scores [nx1], ...]
            scoreAllList = list()  # [scores [n x topk], ...]
            cocoIDAllList = list()  # [cat ids [n x topk], ...]

