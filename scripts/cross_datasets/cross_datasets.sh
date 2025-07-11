DATASET=dtd
SEED=1
LOADEP=50

bash scripts/cross_datasets/train.sh $DATASET $SEED
bash scripts/cross_datasets/test.sh $DATASET $SEED $LOADEP