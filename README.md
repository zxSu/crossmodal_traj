# crossmodal_traj
This repo contains the code for the paper: Crossmodal Transformer Based Generative Framework for Pedestrian Trajectory Prediction.

1.Dependencies  
Our code is based on the BiTraP framework (https://github.com/umautobots/bidireaction-trajectory-prediction). Thus, all of the requirements (python version, pytorch version, .....) are the same as BiTraP framework.  
Additionally, to make some plot visualization during training, we applyed the python package of 'visdom'.

2.Datasets  
JAAD and PIE  
**** Note: It suggests that you prepare at least 1T disk to store:   
(1) all of the data (.png, .json, .xml, ……) from JAAD and PIE datasets.   
(2) the features that are pre-processed by some 3rd feature extractors.  
(a) JAAD  
cd datasets  
python jaad_extract_imgs.py  
(b) PIE  
cd datasets/PIE_imgs_extract  
python extract_annot_images.py  



3.Config_files  
In each config file, there are several “path” needed to be modified by yourself, including “ROOT”, “TRAJECTORY_PATH”, “IMG_FEAT_SAVE_ROOT”, “BODY_ORI_SAVE_ROOT” and “POSE_FEAT_SAVE_ROOT”.



4. Inference  
The checkpoints of our models trained on JAAD, PIE can be downloaded here.(coming soon)  
(1) JAAD deterministic prediction  
python my_traj_test.py --config_file "*YOUR_CONFIG_FOLDER*/my_trans_unimodal_JAAD.yml" --decoder_type "bezier_curve" --ckpt_root "*YOUR_CKPT_ROOT*"  
(2) PIE deterministic prediction  
python my_traj_test.py --config_file "*YOUR_CONFIG_FOLDER*/my_trans_unimodal_PIE.yml" --decoder_type "bezier_curve" --ckpt_root "*YOUR_CKPT_ROOT*"  
(3) JAAD multimodal prediction  
python my_traj_test.py --config_file "*YOUR_CONFIG_FOLDER*/my_trans_multimodal_JAAD.yml" --decoder_type "bezier_curve" --ckpt_root "*YOUR_CKPT_ROOT*"  
(4) PIE multimodal prediction  
python my_traj_test.py --config_file "*YOUR_CONFIG_FOLDER*/my_trans_multimodal_PIE.yml" --decoder_type "bezier_curve" --ckpt_root "*YOUR_CKPT_ROOT*"

**** Note: *YOUR_CKPT_ROOT* is the folder that contains the folder "checkpoints/..."




5.Training  
python my_traj_train.py --config_file "*YOUR_CONFIG_FOLDER*/*YML_FILE*" --decoder_type "bezier_curve"
| method | description | YML_FILE | decoder_type |
| :------: | :------: | :------: | :------: |
| JAAD Deterministic (Table.1) | "crossmodal transformers" + "modality-pair attention" + "bezier_curve decoder" | my_trans_unimodal_JAAD.yml | bezier_curve |
| PIE Deterministic (Table.1) | "crossmodal transformers" + "modality-pair attention" + "bezier_curve decoder" | my_trans_unimodal_PIE.yml | bezier_curve |
| JAAD Multimodal (Table.2) | "crossmodal transformers" + "modality-pair attention" + "CVAE" + "bezier_curve decoder" | my_trans_multimodal_JAAD.yml | bezier_curve |
| PIE Multimodal (Table.2) |  "crossmodal transformers" + "modality-pair attention" + "CVAE" + "bezier_curve decoder" | my_trans_multimodal_PIE.yml | bezier_curve |
| JAAD (config-1 in Table.3) | use the encoder in ACL paper | acl_unimodal_JAAD.yml | bezier_curve |
| PIE (config-1 in Table.3) | use the encoder in ACL paper | acl_unimodal_PIE.yml | bezier_curve |
| JAAD (config-2 in Table.3) | replace the modality-pair attention with concatenation in Fig.3 of the paper | concat_unimodal_JAAD.yml | bezier_curve |
| PIE (config-2 in Table.3) | replace the modality-pair attention with concatenation in Fig.3 of the paper | concat_unimodal_PIE.yml | bezier_curve |
| JAAD (config-3 in Table.3) | replace the bezier_curve decoder with MLP | my_trans_unimodal_JAAD.yml | mlp |
| PIE (config-3 in Table.3) | replace the bezier_curve decoder with MLP | my_trans_unimodal_PIE.yml | mlp |
| JAAD (config-4 in Table.3) | replace the bezier_curve decoder with GRU | my_trans_unimodal_JAAD.yml | gru |
| PIE (config-4 in Table.3) | replace the bezier_curve decoder with GRU | my_trans_unimodal_PIE.yml | gru |




