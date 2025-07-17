DATA=data/
TRAINER=TCP_MOD_MAPLE
WEIGHT=1.0

CFG=vit_b16_ep100_ctxv1
CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
FOLDER=outputs

DATASET=$1

for SEED in 1 2 3
do
    bash scripts/base2new/train.sh ${DATASET} ${SEED}

    bash scripts/base2new/test.sh ${DATASET} ${SEED}
done