SRC=en
TGT=(de fr)
run=$1
MODEL=$2
ARCH='std_gated_tiny'

for tgt in ${TGT[@]}; do
    DATA=data-bin/multi30k.${SRC}-${tgt}/  
    SAVE=checkpoints/${ARCH}.${MODEL}.${SRC}-${tgt}.run_std_${run}
    # SAVE=checkpoints/${ARCH}.${SRC}-${tgt}.run${run}
    mkdir -p exp_log/std/${run}
    for test in test test1 test2; do
        bash evaluate.sh -g 0 -d $DATA -s $test -t std_mmt -p $SAVE >> exp_log/std/${run}/${ARCH}.${MODEL}.${SRC}-${tgt}.run_std_${run}.log
    done
done
# >> exp_log/${ARCH}.${IMG_MODEL}.${SRC}-${tgt}.run${run}.log