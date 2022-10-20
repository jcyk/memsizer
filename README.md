
# Memsizer

The code is for our EMNLP2022 paper: [Linearizing Transformer with Key-Value Memory](https://arxiv.org/pdf/2203.12644.pdf) by [Yizhe Zhang](https://dreasysnail.github.io/)* and [Deng Cai](https://jcyk.github.io/)* (*equal contribution).


## Requirements

Our code is based on [fairseq](https://github.com/facebookresearch/fairseq). You can install fairseq via `pip install fairseq==0.10.0`

We provide the implementation of Memsiser in a plug-in module ([src](./src)). In order to import the module, and make Memsizer available to fairseq, add `--user-dir src` to fairseq command lines. See the following examples.


## Examples

For En-De translation, check [mt_ende.sh](./mt_ende.sh).

For language modeling on Wikitext-103, check [lm_wikitext-103.sh](./lm_wikitext-103.sh).

The above should reproduce the results in our paper.

## Memsizer Configuration 
```
parser.add_argument('--use-memsizer', action='store_true', help='use memsizer in both encoder and decoder.')
parser.add_argument('--encoder-use-rfa', action='store_true', help='use memsizer in encoder.')
parser.add_argument('--decoder-use-rfa', action='store_true', help='use memsizer in decoder.')
parser.add_argument('--causal-proj-dim', type=int, default=4, help='the number of memory slots in causal attention.')
parser.add_argument('--cross-proj-dim', type=int, default=32, help='the number of memory slots in non-causal attention.')

parser.add_argument('--q-init-scale', type=float, metavar='D', default=8.0, help='init scale for \Phi.')
parser.add_argument('--kv-init-scale', type=float, metavar='D', default=8.0, help='init scale for W_l and W_r.')
```
