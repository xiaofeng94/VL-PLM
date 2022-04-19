def add_detector_config(cfg):
    _C = cfg
    _C.SOLVER.RESET_ITER = False
    _C.SOLVER.TRAIN_ITER = -1

    _C.MODEL.ROI_BOX_HEAD.EMB_DIM = 512
    _C.MODEL.ROI_BOX_HEAD.LOSS_WEIGHT_BACKGROUND = 1.0
