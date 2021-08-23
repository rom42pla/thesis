for shots  in 1 2 3 5 10
do
  python -m tools.test_net_laplacian --num-gpus 1 \
  --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_${shots}shot.yaml --eval-only
done

#for shots  in 1 2 3 5 10 30
#do
#  python -m tools.test_net_laplacian --num-gpus 1 \
#  --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_${shots}shot.yaml --eval-only
#done