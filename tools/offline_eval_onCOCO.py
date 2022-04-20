import argparse
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate PLs quality offline')
    parser.add_argument('gt_json', type=str, help='gt coco json file')
    parser.add_argument('pl_json', type=str, help='PL coco json file')
    args = parser.parse_args()
    # print(args)

    #############################################
    gt_COCOJson_file = args.gt_json
    pred_COCOJson_file = args.pl_json   
    #############################################

    PLData = json.load(open(pred_COCOJson_file, 'r'))
    PL_list = list()
    imageId_list = list()
    for anno in PLData['annotations']:
        if 'confidence' in anno.keys():
            data = {'image_id': anno['image_id'],
                    'category_id': anno['category_id'],
                    'bbox': anno['bbox'],
                    'score': anno['confidence']}
            PL_list.append(data)
            imageId_list.append(anno['image_id'])

    print( 'Total PL boxes num: %d, avg num: %.2f' % (len(PL_list), len(PL_list)/len(set(imageId_list))) )

    curSaveJson = './.temp.json'
    with open(curSaveJson, 'w') as outfile:
        json.dump(PL_list, outfile)

    cocoGt = COCO(gt_COCOJson_file)
    cocoDt = cocoGt.loadRes(curSaveJson)

    cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
