DATA=data/
TRAINER=TCP_MOD
WEIGHT=1.0

CFG=vit_b16_ep100_ctxv1
CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
FOLDER=outputs

LOADEP=50
SUB=new

DATASET=$1
SEED=$2


COMMON_DIR=${DATASET}/shots_${SHOTS}_${WEIGHT}/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=${FOLDER}/ctx_${NCTX}/base2new/train_base/${COMMON_DIR}
DIR=${FOLDER}/ctx_${NCTX}/base2new/test_${SUB}/${COMMON_DIR}

rm -r $DIR

if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Skip this job"
else
    echo "Run this job and save the output to ${DIR}"
    CUDA_VISIBLE_DEVICES=0 python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
fi