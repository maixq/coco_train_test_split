## Usage
These scripts help to merge and split coco datasets exported from a supervisely project of different sub directories into train test val datasets for model training.

### Directory Structure Example

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

e.g 433981_damage-annotation folder contains the annotated images and JSON files of different batches exported from a supervisely project in coco format.

### Steps
1. Create and activate conda environment 
```
conda env create -f environment.yml
conda activate my_env

# install packages if haven't already installed 
pip install tqdm numpy pandas funcy sklearn scikit-multilearn pycocotools
```
2. Copy the folder exported from supervisely to current directory
3. Navigate to the folder.
```
cd <path_of_supervisely_folder>
```
4. Run test.sh 
```
bash ../test.sh
```

5. Filter classes if necessary
``` 
1) dent                 3) scratch              5) reflection
2) paint_deterioration  4) dirt                 6) crack
Classes? You can select multiple space separated options: 1 2 3
```

6. Enter split size
```
Enter split size: 0.1
```
### Final COCO Dataset Directory
The final coco dataset is stored in coco_data folder. The expected directory structure is:

├── annotations
│   ├── test.json
│   ├── train.json
│   └── val.json
└── images
    ├── test
    ├── train
    └── val


