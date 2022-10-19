
# Memsizer

the code is for our EMNLP2022 paper [Linearizing Transformer with Key-Value Memory](https://arxiv.org/pdf/2203.12644.pdf)


## Requirements

pip install fairseq==0.10.0

## Examples

For En-De translation, check [mt_ende.sh](./mt_ende.sh).
For language modeling on Wikitext-103, check [lm_wikitext-103.sh](./lm_wikitext-103.sh).

## Memsizer Configuration 
```
parser.add_argument('--use-memsizer', action='store_true', help='use rfa in both encoder and decoder')
parser.add_argument('--encoder-use-rfa', action='store_true', help='use memsizer in encoder')
parser.add_argument('--decoder-use-rfa', action='store_true', help='use memsizer in decoder')
parser.add_argument('--causal-proj-dim', type=int, metavar='D', default=128,
                    help='the number of memory slots in causal attention')
parser.add_argument('--cross-proj-dim', type=int, metavar='D', default=128,
                    help='the number of memory slots in non-causal attention')

parser.add_argument('--q-init-scale', type=float, metavar='D', default=8.0,
                    help='init scale for \Phi proj')
parser.add_argument('--kv-init-scale', type=float, metavar='D', default=8.0,
                    help='init scale for W_l, W_r proj')
```