import numpy as np
import pandas as pd
import os
import json
import funcy
from iterative_stratification import IterativeStratification 
from skmultilearn.model_selection import iterative_train_test_split
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
# from skmultilearn.model_selection import IterativeStratification
from pycocotools.coco import COCO
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-l','--list', nargs='+',help="List of classes for training")
parser.add_argument('-i','--source',help="Path of JSON file for splitting")
parser.add_argument('-s','--split',help="Splitting train/val or train/test")
parser.add_argument('-sz','--size', help="The size of split set")


args = parser.parse_args()

def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'info': info, 'licenses': licenses, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=False)

def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)


def filter_images(images, annotations):
    annotation_ids = funcy.lmap(lambda i: int(i['image_id']), annotations)
    return funcy.lfilter(lambda a: int(a['id']) in annotation_ids, images)


def read_json(json_path):
    with open (json_path, 'r') as f:
        coco = json.load(f)
        info = coco['info']
        licenses = coco['licenses']
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']
        annotation_categories = funcy.lmap(lambda a: int(a['category_id']), annotations)
        # remove classes with one sample as not suitable for stratified sampling
        annotation_categories =  funcy.lremove(lambda i: annotation_categories.count(i) <=1  , annotation_categories)
        annotations =  funcy.lremove(lambda i: i['category_id'] not in annotation_categories  , annotations)
    return coco, info, licenses, images, annotations, categories

def create_id_df(images, annotations,class_list):
    id_d = {}
    for i,im in enumerate(images):
        id = im['id']
        cat = annotations[i]['category_id']
        id_d[id] = [cat]
        ann_im_id = annotations[i]['image_id']
        if ann_im_id in id_d:
            id_d[ann_im_id].append(cat)

    df = pd.DataFrame(list(id_d.items()),columns = ["id", "categories"])
    mlb = MultiLabelBinarizer(sparse_output=True)
    df = df.join(
                pd.DataFrame.sparse.from_spmatrix(
                    mlb.fit_transform(df.pop('categories')),
                    index=df.index,
                    columns=class_list))
    return df

def iterative_split(df, test_size, stratify_columns):
    """Custom iterative train test split which
    'maintains balanced representation with respect
    to order-th label combinations.'
    From https://madewithml.com/courses/mlops/splitting/#stratified-split
    """
    # One-hot encode the stratify columns and concatenate them
    one_hot_cols = [pd.get_dummies(df[col]) for col in stratify_columns]
    one_hot_cols = pd.concat(one_hot_cols, axis=1).to_numpy()
    stratifier = IterativeStratification(
        n_splits=2, order=len(stratify_columns), sample_distribution_per_fold=[float(test_size), 1-float(test_size)], random_state=42)
    train_indices, test_indices = next(stratifier.split(df.to_numpy(), one_hot_cols))
    # Return the train and test set dataframes
    train, test = df.iloc[train_indices], df.iloc[test_indices]
    return train, test


def main():
    json_path = args.source
    coco_annotation = COCO(annotation_file=json_path)
    coco, info, licenses, images, annotations, categories = read_json(json_path)
    df = create_id_df(images, annotations, args.list)
    val_ratio = args.size
    columns_to_be_stratified = args.list
    print(f'Splitting with train ratio ({1-float(val_ratio)}).')
    train, val = iterative_split(df, val_ratio, columns_to_be_stratified)
    return train, val, coco_annotation, info, licenses, images, annotations, categories 

def get_lists(data_list, coco):
    info_list = []
    ann_list = []
    for id in data_list:
        # Get all the annotations for the specified image.
        img_info = coco.loadImgs([id])[0]
        ann_ids = coco.getAnnIds(imgIds=[id], iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        info_list.append(img_info)
        ann_list.append(anns[0])
    return info_list, ann_list

if __name__ == "__main__":
    # json_path = args.source
    # coco_annotation = COCO(annotation_file=json_path)
    # split train val data set
    train, val, coco_annotation, info, licenses, images, annotations, categories = main()
    images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)
    train_ids = train.id.tolist()
    val_ids = val.id.tolist()
    
    val_info_list, val_ann_list = get_lists(val_ids, coco_annotation)
    train_info_list, train_ann_list = get_lists(train_ids, coco_annotation)
  
    # save .json
    train_out_dir = 'annotations/train.json'
    val_out_dir = 'annotations/'+args.split+'.json'
    print("Saving coco json files")
    save_coco(train_out_dir, info, licenses, train_info_list, train_ann_list, categories)
    save_coco(val_out_dir, info, licenses, val_info_list, val_ann_list, categories)
    print()
