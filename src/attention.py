""""Causal Attention."""
from typing import Dict, Optional
import torch
from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor, nn
from torch.nn import Parameter
from fairseq.modules import LayerNorm

EPS=1e-2


@with_incremental_state
class CausalAttention(nn.Module):
    """Random feature cross attention."""

    def __init__(
        self,
        *,
        args,
        embed_dim: int,
        num_heads: int,
        k_dim: int,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        gate=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5 
        self.k_dim = k_dim

        bias = True
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, num_heads*self.k_dim, bias=bias), q_noise, qn_block_size
        )
        self.k_proj = quant_noise(
            nn.Linear(embed_dim, self.k_dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.k_layer_norm = LayerNorm(self.k_dim)
        self.v_layer_norm = LayerNorm(embed_dim)   

        self.dropout_p = args.attention_dropout
        self.norm_k = self.k_layer_norm is not None
        self.norm_v = self.v_layer_norm is not None
        self.reset_parameters(args)

    def reset_parameters(self, args):
        gain = args.q_init_scale ** -0.5
        nn.init.xavier_uniform_(self.q_proj.weight, gain=gain)
        
        gain = args.kv_init_scale ** -0.5
        nn.init.xavier_uniform_(self.k_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=gain)

        if self.k_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.0)
            nn.init.constant_(self.k_proj.bias, 0.0)
            nn.init.constant_(self.v_proj.bias, 0.0)

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None
    ) -> Tensor:
        """Input shape: Time x Batch x Channel

        Args:
            x: [tgt_len, bsz, embed_dim]
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
                [tgt_len, src_len]
        Return:
            attn: [tgt_len, bsz, embed_dim]
        """
        tgt_len, bsz, embed_dim = x.size()
        assert embed_dim == self.embed_dim
        assert list(x.size()) == [tgt_len, bsz, embed_dim]
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
        else:
            saved_state = None
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        if self.norm_k:
            k = self.k_layer_norm(k)
        if self.norm_v:
            v = self.v_layer_norm(v)

        q = q.contiguous().view(
            tgt_len, bsz, self.num_heads, -1)
        k = k.contiguous().view(
            tgt_len, bsz, -1)
        v = v.contiguous().view(
            tgt_len, bsz, -1)

        if saved_state is not None:
            # Incremental decoding (only for step-wise inference)
            assert tgt_len == 1
            prev_s = None
            prev_prefix_len = None
            if "prev_s" in saved_state:
                assert "prev_prefix_len" in saved_state
                prev_s = saved_state["prev_s"]
                prev_prefix_len = saved_state["prev_prefix_len"]
                assert prev_s is not None
                assert prev_prefix_len is not None
            
            attn, s, prefix_len = self.incremental_revatt(
                    q=q, k=k, v=v,
                    prev_s=prev_s, prefix_len=prev_prefix_len)
            saved_state["prev_s"] = s
            saved_state["prev_prefix_len"] = prefix_len
            # In this branch incremental_state is never None
            # assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        else:
            attn = self.masked_revatt(
                                q=q, k=k, v=v,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                training=self.training,
                                dropout_p=self.dropout_p
                    )
        return attn

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def incremental_revatt(self,
                        q: Tensor,
                        k: Tensor,
                        v: Tensor,
                        prev_s: Optional[Tensor] = None,
                        prefix_len: Optional[Tensor] = None) -> Tensor:
        """Loop causal memsizer implementation.

        Args:
            q: [tgt_len, bsz, num_heads, k_dim]
            k: [src_len, bsz, k_dim]
            v: [src_len, bsz, v_dim]
            s: [bsz, k_dim, v_dim]
            prefix_len: [bsz]
        """
        assert k.size(0) == v.size(0) == q.size(0) == 1

        q = q.squeeze(0)
        k = k.squeeze(0)
        v = v.squeeze(0)
        
        bsz, num_heads, _ = q.size()
        _, embed_dim = v.size()
        _, k_dim = k.size()

        if prev_s is None:
            assert prefix_len is None
            prev_s = v.new_zeros([bsz, k_dim, embed_dim])
            prefix_len = v.new_zeros([bsz])

        cur_s = torch.bmm(k.unsqueeze(-1), v.unsqueeze(-2))

        # prev_s is not scaled
        s = prev_s + cur_s
        prev_s = s
        prefix_len = prefix_len + 1

        #CD: s = s * tot_tgt_len ** -0.5
        scaling = (prefix_len ** (-0.5)).view(-1, 1, 1)  
        s = s * scaling

        q = torch.softmax(q, dim=-1, dtype=q.dtype)
        q = torch.mean(q, dim=1, keepdim=True)
        attns = torch.bmm(q, s).squeeze(1)

        return attns, prev_s, prefix_len


    def masked_revatt(self,
                   q: Tensor,
                   k: Tensor,
                   v: Tensor,
                   key_padding_mask: Optional[Tensor] = None,
                   attn_mask: Optional[Tensor] = None,
                   training = False,
                   dropout_p = 0.0) -> Tensor:
        """Masked causal memsizer implementation.

        Args:
            q: [src_len (tgt_len), bsz, num_heads, k_dim]
            k: [tgt_len, bsz, k_dim]
            v: [tgt_len, bsz, v_dim]
            key_padding_mask (ByteTensor, optional): mask to exclude
                    keys that are pads, of shape `(batch, src_len)`, where
                    padding elements are indicated by 1s.
            attn_mask (FloatTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
                [tgt_len, src_len], masked is -inf else 0.
        Return:
            attn: [tgt_len, bsz, num_heads * head_dim]
        """

        tgt_len, bsz, num_heads, _ = q.size()
        k_dim = k.size(-1)
        embed_dim = v.size(-1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz

        assert attn_mask is not None
        assert attn_mask.size(0) == attn_mask.size(1)
        mask = (attn_mask >= 0.0).type(v.dtype)


        #CD: mask[i, j] = (i+1)**-0.5 if j<=i else 0.
        src_len = torch.arange(1, mask.shape[1]+1, dtype=v.dtype, device=v.device)
        scaling = src_len ** (-0.5)
        scaling = scaling.view(-1, 1)


        #s = torch.bmm(k.view(-1, k_dim, 1), v.view(-1, 1, embed_dim))
        #s = s.view(-1, bsz, k_dim, embed_dim)
        s = torch.einsum("sbk,sbd->sbkd", k, v)        
        mask = scaling * mask
        s = torch.einsum("ts,sbkd->tbkd", mask, s)



        q = torch.softmax(q, dim = -1, dtype=q.dtype)
        q = nn.functional.dropout(q, p=dropout_p, training=training)

        attn = torch.einsum("tbk,tbkd->tbd", torch.mean(q, -2), s)

        
        attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        return attn

class CrossAttention(CausalAttention):
    """Random feature cross attention."""

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None
    ) -> Tensor:
        """Input shape: Time x Batch x Channel

        Args:
            query: [tgt_len, bsz, embed_dim]
            key, value: [src_len, bsz, embed_dim]
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
                [tgt_len, src_len]
        Return:
            attn: [tgt_len, bsz, embed_dim]
        """
        assert attn_mask is None, "We do not support attn_mask for cross attention"
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
        else:
            saved_state = None

        q = self.q_proj(query)

        q = q.contiguous().view(
            tgt_len, bsz, self.num_heads, -1)

        s = None
        if saved_state is not None and "prev_s" in saved_state:
            s = saved_state["prev_s"]
        else:
            src_len = key.size(0)
            k = self.k_layer_norm(self.k_proj(key))
            v = self.v_layer_norm(self.v_proj(key))

            k = (
                k.contiguous()
                .view(src_len, bsz, -1)
            )
            v = (
                v.contiguous()
                .view(src_len, bsz, -1)
            )
            s = self.compute_s(k=k, v=v, key_padding_mask=key_padding_mask)
            if saved_state is not None:
                saved_state["prev_s"] = s
                incremental_state = self._set_input_buffer(incremental_state, saved_state)
        attn = self.cross_revatt(
                          q=q, s=s,
                          training=self.training,
                          dropout_p=self.dropout_p
                          )
        return attn
    
    def cross_revatt(self,
                   q: Tensor,
                   s: Tensor,
                   training = False,
                   dropout_p = 0.0) -> Tensor:
        """
        Args:
            q: [tgt_len, bsz, num_heads, k_dim]
            s: [bsz, k_dim, v_dim]
        Return:
            attn: [tgt_len, bsz, num_heads * v_dim]
        """

        tgt_len, bsz, num_heads, _ = q.size()
        _, k_dim, embed_dim = s.size()

        q = torch.softmax(q, dim=-1, dtype=q.dtype)
        q = nn.functional.dropout(q, p=dropout_p, training=training)
        q = q.mean(2).view(tgt_len, bsz, k_dim).transpose(0, 1)


        attn = torch.bmm(q.view(bsz, tgt_len, k_dim), s.view(bsz, k_dim, embed_dim))
        attn = attn.transpose(0, 1)
        assert list(attn.size()) == [tgt_len, bsz, embed_dim]
        attn = attn.contiguous().view(tgt_len, bsz, embed_dim)

        return attn

    def compute_s(self,
                   k: Tensor,
                   v: Tensor,
                   key_padding_mask: Optional[Tensor] = None
                ) -> Tensor:
        """
        Args:
            k: src_len x bsz x k_dim
            v: src_len x bsz x v_dim
            key_padding_mask: bsz x src_len
        Return:
            s: [bsz, k_dim, v_dim]
        """
        if key_padding_mask is not None:
            mask = key_padding_mask.transpose(0, 1).unsqueeze(-1).to(torch.bool)
            k = k.masked_fill(mask, 0.0)


        #CD: We need the tgt_len in each instance
        if key_padding_mask is not None: 
            assert k.size(0) == key_padding_mask.size(1)
            max_len = key_padding_mask.size(1)
            src_len = max_len - key_padding_mask.to(k.dtype).sum(dim=1)
            scaling = (src_len ** (-0.5)).view(-1, 1, 1)
        else:
            scaling = k.size(0) ** (-0.5)
       

        s = torch.bmm(k.transpose(0, 1).transpose(-1, -2), v.transpose(0, 1))
        s = s * scaling

        return s
