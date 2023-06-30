# crossmodal_traj
This repo contains the code for the paper: Crossmodal Transformer Based Generative Framework for Pedestrian Trajectory Prediction.

1.Dependencies  
Our code is based on the BiTraP framework (https://github.com/umautobots/bidireaction-trajectory-prediction). Thus, all of the requirements (python version, pytorch version, .....) are the same as BiTraP framework.  
Additionally, to make some plot visualization during training, we applyed the python package of 'visdom'.

2.Datasets  
JAAD and PIE

3. Inference
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




4.Training  
python my_traj_train.py --config_file "*YOUR_CONFIG_FOLDER*/*YML_FILE*" --decoder_type "bezier_curve"
| description | YML_FILE | decoder_type |
| :------: | :------: | :------: |
| JAAD Deterministic (Table.1) | my_trans_unimodal_JAAD.yml | bezier_curve |
| PIE Deterministic (Table.1) | my_trans_unimodal_PIE.yml | bezier_curve |
| JAAD Multimodal (Table.2) | my_trans_multimodal_JAAD.yml | bezier_curve |
| PIE Multimodal (Table.2) | my_trans_multimodal_PIE.yml | bezier_curve |



