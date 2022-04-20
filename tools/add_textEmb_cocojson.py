import argparse
import json

from tqdm import tqdm, trange

import numpy as np
import torch
import clip

from utils import single_template, multiple_templates, build_text_embedding


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='add text embedding into coco json file')
    parser.add_argument('coco_json_file', type=str, help='config file for proposal network')

    args = parser.parse_args()

    #######################
    orig_json_file = args.coco_json_file

    #######################

    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_json_file = '%s_txtEmb.json' % orig_json_file[:-5]
    print('write to %s' % save_json_file)

    # load CLIP model
    print(clip.available_models())
    CLIPModel, preprocess = clip.load("ViT-B/32", device=device)

    print('loading json')
    # load annotations for base categories
    data = json.load(open(orig_json_file, 'r'))

    print('wiriting text embedding')
    # text embedding with prompt engineering
    for clsDict in tqdm(data['categories']):
        clsName = clsDict['name']
        clsName = clsName.replace('_', ' ')

        text_embed = build_text_embedding(CLIPModel, [clsName], multiple_templates, show_process=False)
        clsDict['text_emb'] = text_embed[0].cpu().numpy().tolist()

    print('saving %s' % save_json_file)
    with open(save_json_file, 'w') as outfile:
        json.dump(data, outfile)

