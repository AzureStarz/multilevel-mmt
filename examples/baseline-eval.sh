SRC=en
TGT=(de fr)
run=$1

for tgt in ${TGT[@]}; do
    DATA=data-bin/multi30k.${SRC}-${tgt}/  
    SAVE=checkpoints/transformer.${SRC}-${tgt}.tiny.run${run}
    
    for test in test test1 test2; do
        bash evaluate.sh -d $DATA -s $test -t translation -p $SAVE
    done
done