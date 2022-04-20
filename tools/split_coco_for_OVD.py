import json
import os, glob, argparse

import numpy as np

from pycocotools.coco import COCO

from utils import COCO_BASE_CatName as baseCatNames
from utils import COCO_NOVEL_CatName as novelCatNames


def binary_search_loop(lst, value):
    low, high = 0, len(lst) - 1
    while low <= high:
        mid = (low + high) // 2
        if lst[mid] < value:
            low = mid + 1
        elif lst[mid] > value:
            high = mid - 1
        else:
            return mid
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect novel category annotations from COCO json file for open vocabulary detection')
    parser.add_argument('coco_json', type=str, help='original coco json file to split')
    parser.add_argument('save_json', type=str, help='split json file')

    args = parser.parse_args()

    ################################
    orig_json_file = args.coco_json         # '../datasets/coco/annotations/instances_val2017.json'
    save_json_file = args.save_json         # './inst_val2017_novel.json'

    usedCatNames = novelCatNames            # change to required category set if needed

    ################################

    coco = COCO(orig_json_file)
    usedCatIds = coco.getCatIds(catNms=usedCatNames)

    data = json.load(open(orig_json_file, 'r'))

    ## annotations
    new_annotations_list = list()
    new_image_ids = list()

    for anno in data['annotations']:
        curImgId = anno['image_id']
        curCatId = anno['category_id']

        if curCatId in usedCatIds:
            new_annotations_list.append(anno)
            new_image_ids.append(curImgId)

    new_image_ids = sorted(list(set(new_image_ids)))

    ## images
    new_imgInfo_list = list()
    for imgInfo in data['images']:
        curImgId = imgInfo['id']

        findCurId = binary_search_loop(new_image_ids, curImgId)
        if findCurId is not None:
            new_imgInfo_list.append(imgInfo)

    # write to json file
    print( 'annotation num: %d, image num: %d, anno per image: %.2f'%(
            len(new_annotations_list), len(new_imgInfo_list), len(new_annotations_list)/len(new_imgInfo_list)) )

    data['annotations'] = new_annotations_list
    data['images'] = new_imgInfo_list
    with open(save_json_file, 'w') as outfile:
        json.dump(data, outfile)

