__all__ = ['build_model']

from .bitrap_gmm import BiTraPGMM
from .bitrap_np import BiTraPNP
#
from bitrap.modeling.my_transformer_multimodal import Trans_Predictor_Multimodal
from bitrap.modeling.my_transformer_unimodal import Trans_Predictor_Unimodal
from bitrap.modeling.acl_transformer_unimodal import ACL_Predictor_Unimodal


_MODELS_ = {
    'BiTraPNP': BiTraPNP,
    'BiTraPGMM': BiTraPGMM,
    'TransPredict_Multimodal': Trans_Predictor_Multimodal,
    'TransPredict_Unimodal': Trans_Predictor_Unimodal,
    'ACL_Unimodal': ACL_Predictor_Unimodal,
}

def make_model(cfg):
    model = _MODELS_[cfg.METHOD]
    try:
        return model(cfg, dataset_name=cfg.DATASET.NAME)
    except:
        return model(cfg.MODEL, dataset_name=cfg.DATASET.NAME)
