#!/bin/bash

TRAINER=TCP_MOD

for DATASET in dtd
do
python parse_test_res.py   outputs/ctx_4/base2new/train_base/${DATASET}/shots_16_1.0/${TRAINER}/vit_b16_ep100_ctxv1/ --test-log
python parse_test_res.py   outputs/ctx_4/base2new/test_new/${DATASET}/shots_16_1.0/${TRAINER}/vit_b16_ep100_ctxv1/ --test-log
done
