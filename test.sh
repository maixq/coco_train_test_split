# # mkdir
# mkdir -p ../data/images
# mkdir -p ../data/annotations
# mkdir -p ../coco_data/images
# mkdir -p ../coco_data/annotations

# # merge batches of data into single folder 
# for M in */*/*.json; do
#   TRACK=$(basename "$M")
#   ALBUM=$(basename $(dirname "$M"))
#   ARTIST=$(basename $(dirname $(dirname "$M")))
#   echo $M
#   mv "$M" "../data/annotations/$ARTIST-$ALBUM-$TRACK"
# done

# for M in */*/*.jpg; do
#   TRACK=$(basename "$M")
#   ALBUM=$(basename $(dirname "$M"))
#   ARTIST=$(basename $(dirname $(dirname "$M")))
#   echo $M
#   mv "$M" "../data/images/$TRACK"
# done

cd ../data

arr=(*/*.json)
length=${#arr[@]}
echo $length
# for dir in */*.json; do
#     files=($dir)    
#     echo ${files[0]}
# done
# # merge first two json files 
python ../merge.py ${arr[0]} ${arr[1]} output1.json  

# # merge the rest of the json files 
for (( j=2; j<${length}; j++ ));
    do
        var=$j
        v=$((var-1))
        echo ${arr[j]} 
        echo output$v.json
        echo output$var.json
        python ../merge.py ${arr[j]}  output$v.json output$var.json 
done
# get the latest merged json file
latest_file=`ls -t *.json | head -1`
echo $latest_file

# read -p "Enter the classes to be included for training: " classes
names=(dent paint_deterioration scratch dirt reflection crack)
selected=()
PS3='Classes? You can select multiple space separated options: '
select name in "${names[@]}" ; do
    for reply in $REPLY ; do
        selected+=(${names[reply - 1]})
    done
    [[ $selected ]] && break
done
echo Selected: "${selected[@]}"

# filter json file based on classes 
python ../filter.py --input_json $latest_file --output_json filtered.json --categories ${selected[@]}

# stratified sampling
cd "../coco_data"

# read -p "Path of JSON file to be split: " source_f
# read -p "Enter 'val' or 'test' for splitting: " split_set
read -p "Enter split size: " size

python ../coco_split.py -i ../data/filtered.json -l ${selected[@]} -s val -sz $size

python ../coco_split.py -i annotations/train.json -l ${selected[@]} -s test -sz $size

cd "../"
python split_images.py