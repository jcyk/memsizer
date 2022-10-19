MKL_SERVICE_FORCE_INTEL=1
causal_proj_dim=4
cross_proj_dim=32
lr=5e-4
q_scale=8.
kv_scale=8.

path=true_vanilla_memsizer
mkdir -p models/${path}
fairseq-train  --user-dir src \
    ../revatt/wmt16.en-de.scale \
    --save-dir models/${path} \
    --arch memsizer_vaswani_wmt_en_de_big \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0     \
    --lr $lr \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --warmup-init-lr 1e-07 \
    --dropout 0.3 \
    --weight-decay 0.0     \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1     \
    --max-tokens 3584 \
    --fp16 \
    --reset-optimizer \
    --seed 8 \
    --update-freq 16 \
    --decoder-use-memsizer \
    --causal-proj-dim ${causal_proj_dim} \
    --cross-proj-dim ${cross_proj_dim} \
    --q-init-scale ${q_scale} \
    --kv-init-scale ${kv_scale} \
    --max-update 90000 \
    --find-unused-parameters \
    --no-epoch-checkpoints \
    --eval-bleu \
    --eval-bleu-args '{"beam": 4, "lenpen": 0.6, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    |& tee models/${path}/${path}.log.txt
