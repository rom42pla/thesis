SHOTS=1
DATASET="voc"

for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)

    case "$KEY" in
          --shots) SHOTS=${VALUE} ;;
          --dataset) DATASET=${VALUE,,} ;;
          *)
    esac
done


case $DATASET in
  "voc")
    python -m tools.test_net_laplacian --num-gpus 1 \
      --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_${SHOTS}shot.yaml --eval-only
    ;;

  "coco")
    python -m tools.test_net_laplacian --num-gpus 1 \
      --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_${SHOTS}shot.yaml --eval-only
    ;;

  "lvis")
    python -m tools.test_net_laplacian --num-gpus 1 \
      --config-file configs/LVIS-detection/faster_rcnn_R_101_FPN_cosine_combined_all.yaml --eval-only

    python -m tools.test_net_laplacian --num-gpus 1 \
      --config-file configs/LVIS-detection/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat.yaml --eval-only
    ;;

  *)
    echo "Unknown dataset '${DATASET}'. Supported datasets are in {'voc', 'coco', 'lvis'}"
    ;;
esac
