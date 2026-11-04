"""Candidate r6: lexicographic comparison of CDF and uniform digits.

The external random source encodes floor(Q*u)+Q as two radix-4096 digits.
For an integer-grid CDF J/Q-1, J > floor(Q*u)+Q is exactly CDF > u.
All comparisons, conjunctions and outcome selection are standard model layers.
"""
import hashlib
import math
import torch
from token_checkpoint import B,C,Q,RADIX,projection
from compact_checkpoint import CompactDriver


def apply_digit_cdf_patch(model,l):
    meta=model.config.qwenqpu_added_layer_prototype
    if meta.get('sampling_revision'):raise ValueError('Sampling patch already applied')
    start=l.sample_onehot+l.size
    uh,ul,dh,dl,ph,eh,pl,eligible=range(start,start+8)
    if eligible>=1021:raise ValueError('Insufficient free coordinates')
    metadata=[]
    with torch.no_grad():
        for layer in [0,1,2,3]:
            mlp=model.model.layers[layer].mlp
            def unit(gate,up,out,coef=1):
                i=meta['MLP_units_used'][layer]
                if i>=model.config.intermediate_size:raise ValueError('Insufficient MLP units')
                assert not torch.count_nonzero(mlp.down_proj.weight[:,i])
                projection(mlp.gate_proj.weight[i],gate);projection(mlp.up_proj.weight[i],up)
                mlp.down_proj.weight[out,i]=coef;meta['MLP_units_used'][layer]+=1
                metadata.append(dict(layer=layer,row=i,output=out))
            if layer==0:
                for out,read,uniform in [(dh,l.acc_read+2,uh),(dl,l.acc_read+3,ul)]:
                    unit({l.sample_flag:256,0:-128},{read:1/128,uniform:-1/128,0:-C/128},out)
            if layer==1:
                for src,out,knots in [(dh,ph,[(0,1),(1,-1)]),(dh,eh,[(-1,1),(0,-2),(1,1)]),
                                      (dl,pl,[(0,1),(1,-1)])]:
                    for knot,coef in knots:unit({src:128,0:-128*knot},{0:1/128},out,coef)
            if layer==2:
                mlp.down_proj.weight[l.sample_score].zero_()
                unit({0:128},{ph:1/128},eligible)
                unit({eh:128,0:-64},{pl:1/64},eligible)
            if layer==3:
                unit({eligible:-128,l.sample_flag:128,0:-64},{0:1},l.sample_score,-1)
        model.lm_head.weight[l.continue_id,0]=-.5/B
    meta['sampling_revision']='digit-lexicographic-r6'
    meta['uniform_digit_coordinates']=[uh,ul]
    meta['sampling_rule']='First represented cumulative probability strictly greater than u'
    return dict(version=meta['sampling_revision'],units=metadata,uniform_coordinates=[uh,ul],
                source_sha256=hashlib.sha256(open(__file__,'rb').read()).hexdigest(),
                uniform_encoding='floor(Q*u)+Q, two base-4096 digits; no probability or outcome accessed by encoder')


class DigitUniformDriver(CompactDriver):
    def _forward_step(self,tid,uniform=None):
        if uniform is None:return super()._forward_step(tid,uniform)
        if not 0<=float(uniform)<1:raise ValueError('Uniform must lie in [0,1)')
        device=next(self.model.parameters()).device;ids=torch.tensor([[tid]],device=device)
        x=self.model.model.embed_tokens(ids).clone()
        x[0,0,self.lay.uniform]=uniform
        coordinates=self.model.config.qwenqpu_added_layer_prototype['uniform_digit_coordinates']
        digits=divmod(math.floor(Q*float(uniform))+int(Q),RADIX)
        for coordinate,digit in zip(coordinates,digits):x[0,0,coordinate]=digit
        with torch.inference_mode():
            out=self.model(inputs_embeds=x,position_ids=torch.zeros_like(ids),
                attention_mask={'linear_attention':None,'full_attention':None},
                past_key_values=self.cache,use_cache=True,logits_to_keep=1)
        self.cache=out.past_key_values
        return int(out.logits[0,-1].argmax())
