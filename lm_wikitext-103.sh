MKL_SERVICE_FORCE_INTEL=1

path=lm
mkdir -p models/${path}
fairseq-train  --user-dir src \
    ../revatt/wikitext-103 \
    --task language_modeling \
    --arch memsizer_lm_wiki103 \
    --max-lr 1 \
    --t-mult 2 \
    --lr-period-updates 270000 \
    --lr-scheduler cosine \
    --lr-shrink 0.75 \
    --lr-period-updates 270000 \
    --lr-scheduler cosine \
    --lr-shrink 0.75 \
    --warmup-updates 16000 \
    --warmup-init-lr 1e-07 \
    --min-lr 1e-09 \
    --optimizer nag \
    --lr 0.0001 \
    --clip-norm 0.1 \
    --criterion adaptive_loss \
    --max-tokens 3072 \
    --update-freq 3 \
    --tokens-per-sample 512 \
    --seed 8 \
    --sample-break-mode none \
    --skip-invalid-size-inputs-valid-test \
    --ddp-backend no_c10d \
    --dropout 0.2 \
    --fp16 \
    --fp16-init-scale 8 \
    --causal-proj-dim 32 \
    --decoder-layerdrop 0.2 \
    --decoder-layers 32 \
    --use-memsizer \
    --max-update 286000 \
    --find-unused-parameters \
    --q-init-scale 32 \
    --kv-init-scale 1 \
    --save-interval 2 \
    --reset-optimizer \
    --decoder-normalize-before \
    |& tee models/${path}/${path}.log.txt
