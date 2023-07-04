The file - train_hoe.json contains all the orientation annotation for MEBOW training set.
The file - val_hoe.json contains all the orientation annotation for MEBOW val set.
The keys in the json files have the following format - "xxxxxx_xxxxxx". The number before '_' refers the image id in COCO dataset. The number after '_' refers the annotation id in COCO dataset.
For example, "546219_1694449" means that the COCO image id and the COCO annotation id for this human instance are "546219" and "1694449".
The value in the json files is the human body orientation. The range is from 0.0 to 355.0.