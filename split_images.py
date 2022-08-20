
from coco_split import read_json
import os
import shutil


coco, info, licenses, images, annotations, categories = read_json('data/filtered.json')
coco, info, licenses, train_images, train_annotations, categories = read_json('coco_data/annotations/train.json')
coco, info, licenses, test_images, test_annotations, categories = read_json('coco_data/annotations/test.json')
coco, info, licenses, val_images, val_annotations, categories = read_json('coco_data/annotations/val.json')

test_images = [d['file_name'] for d in test_images]
train_images = [d['file_name'] for d in train_images]
val_images = [d['file_name'] for d in val_images]

def mv_files(dir,ims):
    src = 'data/images'
    dst = 'coco_data/images/'+dir
    if not os.path.exists(dst):
        os.makedirs(dst)
    for im in ims:
        src_p = src+'/'+im
        dst_p = dst+'/'+im
        shutil.copy(src_p, dst_p)
    return

mv_files('test', test_images)
mv_files('train', train_images)
mv_files('val', val_images)

