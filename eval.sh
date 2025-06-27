#!/bin/bash

for DATASET in eurosat
do
python parse_test_res.py   outputs/ctx_4/base2new/train_base/${DATASET}/shots_16_1.0/TCP_MOD/vit_b16_ep100_ctxv1/ --test-log
python parse_test_res.py   outputs/ctx_4/base2new/test_new/${DATASET}/shots_16_1.0/TCP_MOD/vit_b16_ep100_ctxv1/ --test-log
done
