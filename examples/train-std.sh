SRC=en
TGT=(de fr)
# TGT=(de)

run=$1
MODEL=$2

ARCH='std_gated_tiny'  # model structure
REGION_FEATURE='region/region_embedding.npy'
PHRASE_FEATURE='phrase/phrase_embedding.npy'
IMGAE_FEATURE="image_feature/${MODEL}/${MODEL}_image_features.npy"

for tgt in ${TGT[@]}; do
    DATA=data-bin/multi30k.${SRC}-${tgt}/  
    SAVE=checkpoints/${ARCH}.${MODEL}.${SRC}-${tgt}.run_std_${run}
    
    CUDA_VISIBLE_DEVICES=0, python3 train.py $DATA --task std_mmt \
      --arch $ARCH --share-all-embeddings --dropout 0.3 \
      --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
      --lr 0.003 --min-lr 1e-09 \
      --lr-scheduler inverse_sqrt --warmup-updates 2000 --warmup-init-lr 1e-07 \
      --max-tokens 4096 \
      --criterion std_label_smoothed_cross_entropy --label-smoothing 0.10 \
      --keep-last-epochs 10 \
      --target-lang $tgt \
      --save-dir $SAVE \
      --vision-embedding $IMGAE_FEATURE \
      --latent-embedding $REGION_FEATURE \
      --phrase-embedding $PHRASE_FEATURE \
      --find-unused-parameters --patience 10 
          
done

# --update-freq 2 --no-progress-bar --log-format json --log-interval 1000 \
# --criterion label_smoothed_cross_entropy