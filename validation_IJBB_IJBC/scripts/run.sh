
DATA_ROOT="/data/data/faces/temp_caface_dataset"

# IJBB
python validate_IJB_BC.py \
      --data_root ${DATA_ROOT} \
      --ijb_meta_path IJB/insightface_helper/ijb \
      --dataset_name IJBB \
      --pretrained_model_path ../pretrained_models/CAFace_AdaFaceWebFace4M.ckpt \
      --center_path ../pretrained_models/center_WebFace4MAdaFace_webface4m_subset.pth

# IJBC
python validate_IJB_BC.py \
      --data_root /data/data/faces \
      --ijb_meta_path IJB/insightface_helper/ijb \
      --dataset_name IJBC \
      --pretrained_model_path ../pretrained_models/CAFace_AdaFaceWebFace4M.ckpt \
      --center_path ../pretrained_models/center_WebFace4MAdaFace_webface4m_subset.pth