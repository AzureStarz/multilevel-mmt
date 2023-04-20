SRC=en
TGT=(de fr)
run=$1
IMG_MODEL=$2
IMG_MODEL_WEIGHT=$3
ARCH='gated_tiny'

cp ${IMG_MODEL_WEIGHT} /root/.cache/torch/checkpoints/;

for tgt in ${TGT[@]}; do
    DATA=data-bin/multi30k.${SRC}-${tgt}/  
    SAVE=checkpoints/${ARCH}.${IMG_MODEL}.${SRC}-${tgt}.run_ctl_${run}
    # SAVE=checkpoints/${ARCH}.${SRC}-${tgt}.run${run}
    mkdir -p exp_log/ctl/${run}
    for test in test test1 test2; do
        bash evaluate.sh -g 0 -d $DATA -s $test -t mmt -p $SAVE 
    done
done
# >> exp_log/ctl/${ARCH}.${IMG_MODEL}.${SRC}-${tgt}.run_ctl_${run}.log
# >> exp_log/ctl/${run}/${ARCH}.${IMG_MODEL}.${SRC}-${tgt}.run_ctl_${run}.log