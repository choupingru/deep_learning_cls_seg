# Classification & Semantic Segmentation 

This repo used VGG16 for classification and FCN(32s) for semantic segmentation. 

## Download Dataset
```bash
bash ./get_dataset.sh
```

## Usage

###Training 
```bash
python main.py --task $1 --b $2 --j $3 
```

###Testing
```bash
python main.py --task $1 --test
```

##About this repo
This repo was HW2 from the lesson in National Taiwan University Deep Learning in Computer vision(DLCV).
