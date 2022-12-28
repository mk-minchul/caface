import numpy as np

def l2_normalize(x, axis=1, eps=1e-8):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def average(face_norm_feats, ind_t, intermediates):
    fused = face_norm_feats.mean(0)
    fused_np = l2_normalize(fused, axis=0)
    return fused_np
