DATASET=dtd
LOADEP=50

for SEED in 1 2 3
do
    bash scripts/cross_datasets/train.sh $DATASET $SEED
    bash scripts/cross_datasets/test.sh $DATASET $SEED $LOADEP
done