SRC=en
TGT=(de fr)
# TGT=(de)

run=$1
IMG_MODEL=$2
IMG_MODEL_WEIGHT=$3

cp ${IMG_MODEL_WEIGHT} /root/.cache/torch/checkpoints/;

ARCH='gated_tiny'  # model structure
REGION_FEATURE='region/region_embedding.npy'
PHRASE_FEATURE='phrase/phrase_embedding.npy'
IMGS_PATH='images'

for tgt in ${TGT[@]}; do
    DATA=data-bin/multi30k.${SRC}-${tgt}/  
    SAVE=checkpoints/${ARCH}.${IMG_MODEL}.${SRC}-${tgt}.run_std_${run}
    
    CUDA_VISIBLE_DEVICES=0, python3 train.py $DATA --task mmt \
      --arch $ARCH --share-all-embeddings --dropout 0.3 \
      --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
      --lr 0.003 --min-lr 1e-09 \
      --lr-scheduler inverse_sqrt --warmup-updates 2000 --warmup-init-lr 1e-07 \
      --max-tokens 4096 \
      --criterion label_smoothed_cross_entropy --label-smoothing 0.10 \
      --keep-last-epochs 10 \
      --max-epoch 200 --target-lang $tgt \
      --save-dir $SAVE \
      --image-path ${IMGS_PATH} \
      --img-dim 2048 \
      --img-model $IMG_MODEL \
      --pretrain-weight $IMG_MODEL_WEIGHT \
      --latent-embedding $REGION_FEATURE \
      --phrase-embedding $PHRASE_FEATURE \
      --find-unused-parameters --patience 10 
          
done

# --update-freq 2 --no-progress-bar --log-format json --log-interval 1000 \
# --criterion label_smoothed_cross_entropy