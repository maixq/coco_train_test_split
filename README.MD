## Usage
These scripts help to merge and split coco datasets exported from a supervisely project of different sub directories into train test val datasets for model training.

### Directory Structure Example
```
├── <Folder exported from supervisely>
│   ├── <batch 1>
│   ├── <batch 2>
│   ├── <batch 3>
├── coco_split.py
├── edit_coco_classes.py
├── filter.py
├── merge.py
├── notebook.ipynb
├── split_images.py
└── test.sh
```
e.g 433981_damage-annotation folder contains the annotated images and JSON files of different batches exported from a supervisely project in coco format.

### Steps
1. Create and activate conda environment 
```
conda env create -f environment.yml
conda activate my_env
```
2. Copy the folder exported from supervisely to current directory
3. Navigate to the folder.
```
cd <path_of_supervisely_folder>
```
4. Run test.sh 
```
sh ../test.sh
```

