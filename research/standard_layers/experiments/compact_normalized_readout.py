"""Experimental native correction of finite-precision norm drift.

For event mass E and total mass T near one, the layers emit E*(2-T).
Its error from E/T is E*(T-1)**2/T, at most 2.5e-7 when E<=T and
abs(T-1)<=5e-4. Both masses come from the model's ordinary numeric output.
The host only transports their digits and applies the declared range guard.
"""
import hashlib
import torch
from token_checkpoint import B,Q,RADIX,SCALE,projection


def apply_normalized_readout(model,l):
    meta=model.config.qwenqpu_added_layer_prototype
    if meta.get('normalized_readout'):raise ValueError('Already applied')
    start=l.sample_onehot+l.size+16
    flag,eh,el,th,tl,high,event,total,ratio=range(start,start+9)
    if ratio>=1021:raise ValueError('No free normalizer coordinates')
    changes=[]
    with torch.no_grad():
        assert not torch.count_nonzero(model.model.embed_tokens.weight[:,start:ratio+1])
        for layer in [0,1,2]:
            mlp=model.model.layers[layer].mlp
            def unit(gate,up,out,coef=1):
                row=meta['MLP_units_used'][layer]
                assert not torch.count_nonzero(mlp.down_proj.weight[:,row])
                projection(mlp.gate_proj.weight[row],gate);projection(mlp.up_proj.weight[row],up)
                mlp.down_proj.weight[out,row]=coef;meta['MLP_units_used'][layer]+=1
                changes.append(dict(layer=layer,row=row,output=out))
            if layer==0:
                for out,h,lo in [(event,eh,el),(total,th,tl)]:
                    unit({flag:256,0:-128},{h:RADIX/Q/128,lo:1/Q/128,0:-1/128},out)
            if layer==1:
                # SiLU(x)-SiLU(-x)=x; no external multiplication/division.
                for sign in [1,-1]:
                    unit({0:2*sign,total:-sign},{event:1},ratio,sign)
            if layer==2:
                # The original probability-digit units are dormant only for
                # the new operation. At flag=0 their projections are unchanged.
                for row in [0,1]:
                    mlp.gate_proj.weight[row,flag] += -256/SCALE
                for ph,factor in [(0,Q/RADIX),(1,Q)]:
                    form={ratio:factor,0:factor}
                    if ph:form[high]=-RADIX
                    else:form[0]+=.5/RADIX-.5
                    unit({flag:256,l.prob_phase+ph:256,0:-384},
                         {k:v/128 for k,v in form.items()},l.prob_z)
    record=dict(version='native-near-unit-normalizer-r1',coordinates=[flag,eh,el,th,tl,high],
        guard_abs_total_minus_one=5e-4,formula='event_mass * (2 - total_mass)',
        mathematical_ratio_approximation_bound=2.5e-7,new_units=changes,
        source_sha256=hashlib.sha256(open(__file__,'rb').read()).hexdigest(),custom_model_forward=False)
    meta['normalized_readout']=record
    return record


def normalize_emitted_digits(driver,event_digits,total_digits):
    l=driver.lay;model=driver.model
    model_meta=model.config.qwenqpu_added_layer_prototype
    meta=model_meta['normalized_readout']
    token_offset=model_meta.get('protocol_token_offset',0)
    quantum_offset=model_meta.get('quantum_offset',0)
    flag,eh,el,th,tl,high=meta['coordinates'];outputs=[]
    for ph in range(2):
        if hasattr(driver,'prediction_budget') and len(driver.call_trace)>=driver.prediction_budget:
            raise ValueError('Total prediction budget exhausted')
        ids=torch.tensor([[token_offset+l.halt_id]],device=next(model.parameters()).device)
        x=model.model.embed_tokens(ids).clone()
        x[0,0,quantum_offset+flag]=1;x[0,0,quantum_offset+l.prob_phase+ph]=1
        for coordinate,value in zip([eh,el,th,tl],[*event_digits,*total_digits]):x[0,0,quantum_offset+coordinate]=value
        if ph:x[0,0,quantum_offset+high]=outputs[0]
        kw={} if model_meta.get('language_weights_included') else dict(position_ids=torch.zeros_like(ids),
            attention_mask={'linear_attention':None,'full_attention':None})
        with torch.inference_mode():
            out=model(inputs_embeds=x,**kw,past_key_values=driver.cache,use_cache=True,logits_to_keep=1)
        driver.cache=out.past_key_values;token=int(out.logits[0,-1].argmax())-token_offset
        if token//RADIX!=ph:raise RuntimeError('Invalid normalized probability digit')
        outputs.append(token-ph*RADIX)
        driver.call_trace.append(dict(operation='normalize-emitted-digits',phase=ph,
            event_digits=list(event_digits),total_digits=list(total_digits),argmax_token=token))
    return dict(probability=(outputs[0]*RADIX+outputs[1])/Q-1,numeric_digits=outputs)


def normalized_probability(driver,mask):
    # Raw readout is still available for audit. The returned probability is
    # computed by the normalizer weights, not this host-side guard.
    total=driver.probability([1]*driver.lay.size)
    if abs(total['probability']-1)>5e-4:
        raise RuntimeError('Total outside the declared near-unit range; no corrected answer')
    event=driver.probability(mask)
    result=normalize_emitted_digits(driver,event['numeric_digits'],total['numeric_digits'])
    return dict(**result,raw_event_probability=event['probability'],raw_total_probability=total['probability'])
