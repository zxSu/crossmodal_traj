# crossmodal_traj
This repo contains the code for the paper: Crossmodal Transformer Based Generative Framework for Pedestrian Trajectory Prediction.

1.Dependencies
Our code is based on the BiTraP framework (https://github.com/umautobots/bidireaction-trajectory-prediction). Thus, all of the requirements (python version, pytorch version, .....) are the same as BiTraP framework.
Additionally, to make some plot visualization during training, we applyed the python package of 'visdom'.

2.Datasets
JAAD and PIE

3.Training
There are several "config files" in the "configs" folder.
(1) for Table.1 in the paper
"my_trans_unimodal_JAAD.yml" points to the deterministic prediction in JAAD dataset by our method.
"my_trans_unimodal_PIE.yml" points to the deterministic prediction in PIE dataset by our method.
(2) for Table.2 in the paper
"my_trans_multimodal_JAAD.yml" points to the multimodal prediction in JAAD dataset by our method.
