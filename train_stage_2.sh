python -m tools.ckpt_surgery \
  --src1 checkpoints/model_final.pth \
  --method randinit \
  --save-dir checkpoints

python -m tools.train_net --num-gpus 1 \
  --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_1shot.yaml \
  --opts MODEL.WEIGHTS checkpoints/model_reset_surgery.pth
