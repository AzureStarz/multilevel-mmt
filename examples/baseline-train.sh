SRC=en
TGT=(de fr)
run=$1
ARCH='transformer_tiny'
SCALE='tiny'

for tgt in ${TGT[@]}; do
    DATA=data-bin/multi30k.${SRC}-${tgt}/  
    SAVE=checkpoints/transformer.${SRC}-${tgt}.${SCALE}.run${run}
    
    CUDA_VISIBLE_DEVICES=0 python train.py $DATA --task translation \
          --arch $ARCH --share-all-embeddings --dropout 0.3 \
          --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 \
          --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
          --warmup-updates 2000 --lr 0.003 --min-lr 1e-09 \
          --criterion std_label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0 \
          --max-tokens 4096 \
          --target-lang $tgt \
          --save-dir $SAVE \
          --find-unused-parameters --patience 10 
done

