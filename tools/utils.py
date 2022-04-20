from tqdm import tqdm, trange

import numpy as np
from PIL import Image, ImageDraw

import torch, torchvision
import torch.nn.functional as F

import clip

from detectron2.structures import ImageList, Boxes


### class names
COCO_BASE_CatName = [
    "toilet",
    "bicycle",
    "apple",
    "train",
    "laptop",
    "carrot",
    "motorcycle",
    "oven",
    "chair",
    "mouse",
    "boat",
    "kite",
    "sheep",
    "horse",
    "sandwich",
    "clock",
    "tv",
    "backpack",
    "toaster",
    "bowl",
    "microwave",
    "bench",
    "book",
    "orange",
    "bird",
    "pizza",
    "fork",
    "frisbee",
    "bear",
    "vase",
    "toothbrush",
    "spoon",
    "giraffe",
    "handbag",
    "broccoli",
    "refrigerator",
    "remote",
    "surfboard",
    "car",
    "bed",
    "banana",
    "donut",
    "skis",
    "person",
    "truck",
    "bottle",
    "suitcase",
    "zebra"
]
COCO_NOVEL_CatName = [
    "umbrella",
    "cow",
    "cup",
    "bus",
    "keyboard",
    "skateboard",
    "dog",
    "couch",
    "tie",
    "snowboard",
    "sink",
    "elephant",
    "cake",
    "scissors",
    "airplane",
    "cat",
    "knife"
]


#### for CLIP text embeddings
def article(name):
    return 'an' if name[0] in 'aeiou' else 'a'

def processed_name(name, rm_dot=False):
  # _ for lvis
  # / for obj365
  res = name.replace('_', ' ').replace('/', ' or ').lower()
  if rm_dot:
    res = res.rstrip('.')
  return res

single_template = [
    'a photo of {article} {}.'
]
multiple_templates = [
    'There is {article} {} in the scene.',
    'There is the {} in the scene.',
    'a photo of {article} {} in the scene.',
    'a photo of the {} in the scene.',
    'a photo of one {} in the scene.',

    'itap of {article} {}.',
    'itap of my {}.',  # itap: I took a picture of
    'itap of the {}.',
    'a photo of {article} {}.',
    'a photo of my {}.',
    'a photo of the {}.',
    'a photo of one {}.',
    'a photo of many {}.',

    'a good photo of {article} {}.',
    'a good photo of the {}.',
    'a bad photo of {article} {}.',
    'a bad photo of the {}.',
    'a photo of a nice {}.',
    'a photo of the nice {}.',
    'a photo of a cool {}.',
    'a photo of the cool {}.',
    'a photo of a weird {}.',
    'a photo of the weird {}.',

    'a photo of a small {}.',
    'a photo of the small {}.',
    'a photo of a large {}.',
    'a photo of the large {}.',

    'a photo of a clean {}.',
    'a photo of the clean {}.',
    'a photo of a dirty {}.',
    'a photo of the dirty {}.',

    'a bright photo of {article} {}.',
    'a bright photo of the {}.',
    'a dark photo of {article} {}.',
    'a dark photo of the {}.',

    'a photo of a hard to see {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of {article} {}.',
    'a low resolution photo of the {}.',
    'a cropped photo of {article} {}.',
    'a cropped photo of the {}.',
    'a close-up photo of {article} {}.',
    'a close-up photo of the {}.',
    'a jpeg corrupted photo of {article} {}.',
    'a jpeg corrupted photo of the {}.',
    'a blurry photo of {article} {}.',
    'a blurry photo of the {}.',
    'a pixelated photo of {article} {}.',
    'a pixelated photo of the {}.',

    'a black and white photo of the {}.',
    'a black and white photo of {article} {}.',

    'a plastic {}.',
    'the plastic {}.',

    'a toy {}.',
    'the toy {}.',
    'a plushie {}.',
    'the plushie {}.',
    'a cartoon {}.',
    'the cartoon {}.',

    'an embroidered {}.',
    'the embroidered {}.',

    'a painting of the {}.',
    'a painting of a {}.',
]

def build_text_embedding(model, categories, templates, add_this_is=False, show_process=True):
    run_on_gpu = torch.cuda.is_available()

    with torch.no_grad():
        all_text_embeddings = []
        if show_process:
            print('Building text embeddings...')
        for catName in (tqdm(categories) if show_process else categories):
            texts = [template.format(catName, article=article(catName)) for template in templates]
            if add_this_is:
                texts = ['This is ' + text if text.startswith('a') or text.startswith('the') else text
                         for text in texts]
            texts = clip.tokenize(texts)  # tokenize
            if run_on_gpu:
                texts = texts.cuda()

            text_embeddings = model.encode_text(texts)  # embed with text encoder
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()
            all_text_embeddings.append(text_embedding)

        all_text_embeddings = torch.stack(all_text_embeddings, dim=1)
        if run_on_gpu:
            all_text_embeddings = all_text_embeddings.cuda()

        return all_text_embeddings.transpose(dim0=0, dim1=1)


# get multi-scale CLIP image imbedding
def CLIP_score_multi_scale(clip_model, img_patch_scalelist, text_features, softmax_t=0.01):
    # img_patch_scalelist: [ [n*patches for scale 1], [n*patches for scale 2], ...]
    # patchNum = img_patch_scalelist[0].shape[0]

    patchNum = len(img_patch_scalelist[0])
    patch_per_split = 1000

    splitIdxList = [i*patch_per_split for i in range(1 + patchNum//patch_per_split)]
    if splitIdxList[-1] != patchNum:
        splitIdxList.append(patchNum)

    allSimilarity = []

    for sp_idx in range(len(splitIdxList) - 1):
        startIdx = splitIdxList[sp_idx]
        endIdx = splitIdxList[sp_idx + 1]

        image_features = None
        for s_id, imgPatchesList in enumerate(img_patch_scalelist):
            imgPatches = torch.cat(imgPatchesList[startIdx:endIdx], dim=0)

            with torch.no_grad():
                curImgFeat = clip_model.encode_image(imgPatches)
                curImgFeat /= curImgFeat.norm(dim=-1, keepdim=True)

            if image_features is None:
                image_features = curImgFeat.clone()
            else:
                image_features += curImgFeat

        sacleNum = len(img_patch_scalelist)
        image_features /= sacleNum
        image_features /= image_features.norm(dim=-1, keepdim=True)

        similarity = ((1 / softmax_t) * image_features @ text_features.T).softmax(dim=-1)
        allSimilarity.append(similarity)

    allSimilarity = torch.cat(allSimilarity, dim=0)
    return allSimilarity

# partially run roi_head of the detectron2 model
def refineBoxByRoIHead(roi_head, features, proposals):
    with torch.no_grad():
        features = [features[f] for f in roi_head.box_in_features]
        box_features = roi_head.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = roi_head.box_head(box_features)
        predictions = roi_head.box_predictor(box_features)
        del box_features

        boxes = roi_head.box_predictor.predict_boxes(predictions, proposals)

    # choose only one proposal delats
    meanBoxesList = list()

    num_preds_per_image = [len(p) for p in proposals]
    for iidx, curBoxes in enumerate(boxes):
        predNum = num_preds_per_image[iidx]
        curBoxes = curBoxes.reshape(predNum, -1, 4)  # N x (k x 4)
        curBoxes = curBoxes.mean(dim=1)  # N x 4
        meanBoxesList.append(curBoxes)

    return meanBoxesList

def get_region_proposal(input_img, CA_maskRCNN, DataAug=None, roihead_num=10, topK_box=100):
    # load image tensor & pre-process
    height, width = input_img.shape[:2] # BGR

    if DataAug is not None:
        imgTR = DataAug.get_transform(input_img).apply_image(input_img)
    else:
        imgTR = input_img
    imgTR = torch.as_tensor((imgTR).astype("float32").transpose(2, 0, 1)).to(CA_maskRCNN.device)
    imgTR = ((imgTR - CA_maskRCNN.pixel_mean) / CA_maskRCNN.pixel_std)
    resizeRatio = height / imgTR.shape[1]

    images = ImageList.from_tensors([imgTR], CA_maskRCNN.backbone.size_divisibility)

    # get region proposals by repeating the roi head
    with torch.no_grad():
        features = CA_maskRCNN.backbone(images.tensor)
        proposals, _ = CA_maskRCNN.proposal_generator(images, features, None)

        curProposals = proposals
        for roiCount in range(roihead_num):
            boxes = refineBoxByRoIHead(CA_maskRCNN.roi_heads, features, curProposals)
            curProposals[0].proposal_boxes = Boxes(boxes[0])

        proposal_boxes = curProposals[0].proposal_boxes.tensor.cpu().numpy() * resizeRatio  # need to rescale
        proposal_boxes = proposal_boxes.tolist()
        proposal_boxes = proposal_boxes[:topK_box]

        pp_scores = torch.sigmoid(proposals[0].objectness_logits).cpu().numpy()
        pp_scores = pp_scores.tolist()
        pp_scores = pp_scores[:topK_box]

    return proposal_boxes, pp_scores

def get_CLIP_pred_for_proposals(input_img, proposal_boxes, pp_scores,
                                CLIP_model, preprocess, clip_text_embed, usedCatIds_inOrder,
                                box_scalelist=[1, 1.5], topK_clip_scores=1, device='cuda'):
    '''
    input_img: from cv2.imread, in BGR
    proposal_boxes: [[xyxy], [xyxy], ...]
    pp_scores: objectness scores for each region proposal
    '''
    height, width = input_img.shape[:2]  # BGR
    pilImg = Image.fromarray(input_img[:, :, ::-1])  # RGB

    usedCatNum = len(usedCatIds_inOrder)

    curBoxList = list()
    curRPNScoreList = list()
    curCLIPScoreList = list()
    curPredCatIdList = list()

    clipInput_list_scalelist = [[] for i in range(len(box_scalelist))]
    for b_idx, box in enumerate(proposal_boxes):
        box = scale_box(box, 1, max_H=height, max_W=width)  # ensure every box is in the image
        if box[2] - box[0] >= 5 and box[3] - box[1] >= 5:
            curBoxList.append(box)
            curRPNScoreList.append(pp_scores[b_idx])

            # add scales
            for scale_id, boxScale in enumerate(box_scalelist):
                scaledBox = scale_box(box, boxScale, max_H=height, max_W=width)
                cropImg = pilImg.crop(scaledBox)

                clipInput = preprocess(cropImg).unsqueeze(0).to(device)
                clipInput_list_scalelist[scale_id].append(clipInput)

    if len(curBoxList) > 0:
        allSimilarity = CLIP_score_multi_scale(CLIP_model, clipInput_list_scalelist, clip_text_embed)

        ############### merge CLIP and RPN scores
        for b_idx, box in enumerate(curBoxList):
            clipScores, indices = allSimilarity[b_idx][:usedCatNum].topk(topK_clip_scores)

            curCLIPScoreList.append(clipScores.cpu().numpy().tolist())
            curPredCatIdList.append(usedCatIds_inOrder[indices.cpu().numpy()].tolist())

    return curBoxList, curRPNScoreList, curCLIPScoreList, curPredCatIdList

def detection_postprocessing(box_List, score_List, pred_id_list,
                                use_thre=True, thres=0.05,
                                use_pre_cls_nms=True, pre_cls_nms_thres=0.6, device='cuda'):
    # thresholding
    curBoxArr = np.array(box_List)
    curScoreArr = np.array(score_List)
    curPredIdArr = np.array(pred_id_list)

    if use_thre:
        thresMask = curScoreArr > thres
        curBoxArr = curBoxArr[thresMask, :]
        curScoreArr = curScoreArr[thresMask]
        curPredIdArr = curPredIdArr[thresMask]

    # pre-class NMS
    if use_pre_cls_nms:
        currUsedBoxTR = torch.as_tensor(curBoxArr).to(device)
        currScoreTR = torch.as_tensor(curScoreArr).to(device)
        currCatIDsTR = torch.as_tensor(curPredIdArr).to(device)
        selBoxIds = torchvision.ops.batched_nms(currUsedBoxTR, currScoreTR, currCatIDsTR, pre_cls_nms_thres)

        curBoxList = currUsedBoxTR[selBoxIds, :].cpu().numpy().tolist()
        curScoreList = currScoreTR[selBoxIds].cpu().numpy().tolist()
        curPredIdList = currCatIDsTR[selBoxIds].cpu().numpy().tolist()
    else:
        curBoxList = curBoxArr.tolist()
        curScoreList = curScoreArr.tolist()
        curPredIdList = curPredIdArr.tolist()

    return curBoxList, curScoreList, curPredIdList

#### data processing
def get_coco_ids_by_order(coco, cat_name_list):
    catIDs_list = list()
    for catName in cat_name_list:
        catID = coco.getCatIds(catNms=[catName])
        if len(catID) != 1:
            print('%s is not valid cat name' % catName)
        else:
            catIDs_list.append(catID[0])

    return catIDs_list

def xyxy2xywh(bbox):
    """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
    evaluation.

    Args:
        bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
            ``xyxy`` order.

    Returns:
        list[float]: The converted bounding boxes, in ``xywh`` order.
    """
    if isinstance(bbox, list):
        _bbox = bbox
    else:
        _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0],
        _bbox[3] - _bbox[1],
    ]

def detections2json(img_ids, box_all_list, score_all_list, pred_catId_all_list):
    """Convert proposal results to COCO json style."""
    json_results = []

    for idx, img_id in enumerate(img_ids):
        img_id = img_ids[idx]
        bboxesList = box_all_list[idx]
        scoreList = score_all_list[idx]
        catIdList = pred_catId_all_list[idx]

        for b_idx, box in enumerate(bboxesList):
            box = xyxy2xywh(box)
            score = scoreList[b_idx]
            catId = catIdList[b_idx]

            data = {'image_id': img_id,
                    'category_id': catId,
                    'bbox': box,
                    'score': score}
            json_results.append(data)

    return json_results

def scale_box(box, scale, max_H=np.inf, max_W=np.inf):
    # box: x0, y0, x1, y1
    # scale: float
    x0, y0, x1, y1 = box

    cx = ((x0 + x1) / 2)
    cy = ((y0 + y1) / 2)
    bw = (x1 - x0) * scale
    bh = (y1 - y0) * scale

    new_x0 = max(cx - bw / 2, 0)
    new_y0 = max(cy - bh / 2, 0)
    new_x1 = min(cx + bw / 2, max_W)
    new_y1 = min(cy + bh / 2, max_H)

    return [new_x0, new_y0, new_x1, new_y1]