from bitrap.engine.trainer import do_train 
from bitrap.engine.trainer import do_val 
from bitrap.engine.trainer import inference

from bitrap.engine.my_multimodal_trainer import multimodal_train, multimodal_val, multimodal_inference
from bitrap.engine.my_unimodal_trainer import unimodal_train, unimodal_val, unimodal_inference


ENGINE_ZOO = {
                'BiTraPNP': (do_train, do_val, inference),
                'BiTraPGMM': (do_train, do_val, inference),
                'TransPredict_Multimodal': (multimodal_train, multimodal_val, multimodal_inference),
                'TransPredict_Unimodal': (unimodal_train, unimodal_val, unimodal_inference),
                'ACL_Unimodal': (unimodal_train, unimodal_val, unimodal_inference),
                'SimpleTrans_Unimodal': (unimodal_train, unimodal_val, unimodal_inference),
                'SimpleGRU_Unimodal': (unimodal_train, unimodal_val, unimodal_inference)
                }

def build_engine(cfg):
    return ENGINE_ZOO[cfg.METHOD]
