#python -m tools.train_net --num-gpus 1 \
#  --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_2shot.yaml

SHOTS=2

python -m tools.test_net_laplacian --num-gpus 1 \
  --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_${SHOTS}shot.yaml --eval-only

#python -m tools.test_net_laplacian --num-gpus 1 \
#  --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_${SHOTS}shot.yaml --eval-only

#python -m tools.test_net --num-gpus 1 \
#  --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_2shot.yaml --eval-only

#python -m tools.test_net --num-gpus 1 \
#  --config-file configs/detr.yaml --eval-only

#python3 tools/run_experiments.py --num-gpus 1 \
#        --shots 2 3 5 10 \
#        --seeds 0 30 \
#        --split 1

#python -m tools.train_net_detr --num-gpus 1 \
#  --config-file configs/detr.yaml --eval-only
