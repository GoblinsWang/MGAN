cd ../..
CARD=1

CUDA_VISIBLE_DEVICES=$CARD python train.py --path_opt option/RSICD/rsicd_MGAN_With_TFEM.yaml

CUDA_VISIBLE_DEVICES=$CARD python test_ave.py --path_opt option/RSICD/rsicd_MGAN_With_TFEM.yaml
