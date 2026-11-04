"""Construct a stock Qwen3.5 model with new QPU layers from existing files.

The original Qwenputer CPU prefix and chip vocabulary are omitted. Language
weights are sliced, zero padded and RMS gains adjusted; new QPU tensors are
mapped into separate residual coordinates of three standard decoder layers.
This module never downloads files or invokes an external quantum executor.
"""
from pathlib import Path
import json
import math
import re
import torch
from safetensors import safe_open
from transformers import Qwen3_5ForCausalLM,Qwen3_5TextConfig


class TensorSource:
    def __init__(self,path):
        self.path=Path(path)
        self.config=json.loads((self.path/'config.json').read_text())
        index=self.path/'model.safetensors.index.json'
        if index.exists():self.mapping=json.loads(index.read_text())['weight_map']
        else:
            files=list(self.path.glob('*.safetensors'))
            if len(files)!=1:raise ValueError('One safetensors file or an index is required')
            with safe_open(files[0],framework='pt',device='cpu') as f:
                self.mapping={k:files[0].name for k in f.keys()}
    def tensor(self,key,slices=None):
        with safe_open(self.path/self.mapping[key],framework='pt',device='cpu') as f:
            return f.get_tensor(key) if slices is None else f.get_slice(key)[slices]


def merged_config(base,prefix):
    chip=json.loads((base.path/'chipformer.json').read_text())
    if not (base.config['hidden_size']==3968 and base.config['num_hidden_layers']==43
            and chip['chip_layers']==11 and chip['chip_dims_offset']==2560):
        raise ValueError('The verified inherited Qwenputer layout is required')
    if not prefix.config.get('qwenqpu_added_layer_prototype',{}).get('autonomous_controller'):
        raise ValueError('An autonomous QPU prefix checkpoint is required')
    meta=prefix.config['qwenqpu_added_layer_prototype']
    if meta['format']=='compact-addressed-v7':allowed=range(1,7)
    elif meta['format']=='compact-banked-v8':allowed=range(1,8)
    else:allowed=(3,4)
    if meta['qubits'] not in allowed:raise ValueError('Unsupported verified-layout capacity')
    cfg=dict(base.config)
    prefix_layers=prefix.config['num_hidden_layers']
    cfg.update(hidden_size=4096,num_hidden_layers=32+prefix_layers,
        layer_types=prefix.config['layer_types']+base.config['layer_types'][11:],
        vocab_size=chip['chip_token_offset']+prefix.config['vocab_size'],
        linear_num_key_heads=prefix.config['linear_num_key_heads'],
        linear_num_value_heads=prefix.config['linear_num_value_heads'],
        intermediate_size=max(base.config['intermediate_size'],prefix.config['intermediate_size']),
        tie_word_embeddings=False,dtype='float32')
    cfg['qwenqpu_added_layer_prototype']=dict(meta,language_weights_included=True,
        replaced_CPU_layers=11,language_layers_from=prefix_layers,quantum_offset=2560,
        protocol_token_offset=chip['chip_token_offset'],prefix_hidden=prefix.config['hidden_size'])
    return Qwen3_5TextConfig(**cfg)


def _assign(model,key,value):
    wanted=model.get_parameter(key)
    if value.shape!=wanted.shape:raise ValueError((key,tuple(value.shape),tuple(wanted.shape)))
    parent,name=key.rsplit('.',1)
    setattr(model.get_submodule(parent),name,torch.nn.Parameter(value,requires_grad=False))


def build_language_reference(base_path,device='cuda',progress=None):
    """Extract the inherited language coordinates into an ordinary 32-layer LM.

    This comparison model uses exactly the available inherited language tensors,
    including their BF16 rounding. It does not download a second base checkpoint.
    """
    base=TensorSource(base_path)
    cfg=dict(base.config)
    chip=json.loads((base.path/'chipformer.json').read_text())
    d0=chip['chip_dims_offset'];first=chip['qwen_layers_from']
    cfg.update(hidden_size=d0,num_hidden_layers=base.config['num_hidden_layers']-first,
        layer_types=base.config['layer_types'][first:],vocab_size=chip['chip_token_offset'],
        tie_word_embeddings=False,dtype='float32')
    cfg=Qwen3_5TextConfig(**cfg)
    with torch.device('meta'):model=Qwen3_5ForCausalLM(cfg)
    keys=list(dict(model.named_parameters()))
    for number,key in enumerate(keys,1):
        match=re.match(r'model.layers\.(\d+)\.(.*)',key)
        source=(f'model.layers.{int(match[1])+first}.{match[2]}' if match else key)
        if key=='lm_head.weight':source='model.embed_tokens.weight'
        shape=model.get_parameter(key).shape
        if key in ['model.embed_tokens.weight','lm_head.weight']:
            value=torch.empty(shape,device=device,dtype=torch.float32)
            for a in range(0,cfg.vocab_size,8192):
                b=min(a+8192,cfg.vocab_size)
                value[a:b]=base.tensor(source,(slice(a,b),slice(0,d0))).to(device,dtype=torch.float32)
        else:
            w=base.tensor(source)
            if source.endswith('layernorm.weight') or source=='model.norm.weight':
                value=(1+w[:d0].to(device,dtype=torch.float32))*math.sqrt(base.config['hidden_size']/d0)-1
            elif w.ndim==2 and w.shape[1]==base.config['hidden_size']:
                value=w[:,:d0].to(device,dtype=torch.float32)
            elif w.ndim==2 and w.shape[0]==base.config['hidden_size']:
                value=w[:d0,:].to(device,dtype=torch.float32)
            else:value=w.to(device,dtype=torch.float32)
        _assign(model,key,value.contiguous())
        if progress is not None:progress(key,number,len(keys))
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    model.model.rotary_emb=Qwen3_5TextRotaryEmbedding(config=cfg,device=device)
    return model.eval().requires_grad_(False)


def build(base_path,prefix_path,device='cuda',progress=None):
    base,prefix=TensorSource(base_path),TensorSource(prefix_path)
    cfg=merged_config(base,prefix)
    with torch.device('meta'):model=Qwen3_5ForCausalLM(cfg)
    target_bytes=sum(p.numel()*4 for p in model.parameters())
    if device=='cuda':
        free,total=torch.cuda.mem_get_info()
        if free<target_bytes+5*2**30:raise MemoryError('Insufficient free GPU memory for model plus 5 GiB reserve')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    d0=2560;oldh=3968;off=d0;ph=prefix.config['hidden_size']
    proto=cfg.qwenqpu_added_layer_prototype
    prefix_layers=prefix.config['num_hidden_layers']
    text_vocab=proto['protocol_token_offset'];qv=prefix.config['vocab_size']
    oldk=base.config['linear_num_key_heads']*128
    newk=cfg.linear_num_key_heads*128
    oldv=base.config['linear_num_value_heads']*128
    qflag,tflag=4095,4094
    keys=list(dict(model.named_parameters()))
    audit={'base_files':str(Path(base_path).name),'prefix_files':str(Path(prefix_path).name),
           'CPU_parameter_layers_loaded':[],'language_layers_loaded':list(range(11,43)),
           'target_bytes_FP32':target_bytes,'parameters':sum(p.numel() for p in model.parameters()),
           'quantum_prefix_layers':prefix_layers,'custom_forward':False,'mapped_tensors':0}
    for key in keys:
        shape=model.get_parameter(key).shape
        value=torch.zeros(shape,device=device,dtype=torch.float32)
        if key in ['model.embed_tokens.weight','lm_head.weight']:
            for first in range(0,text_vocab,8192):
                last=min(text_vocab,first+8192)
                w=base.tensor('model.embed_tokens.weight',(slice(first,last),slice(0,d0)))
                value[first:last,:d0]=w.to(device=device,dtype=torch.float32)
            pk='model.embed_tokens.weight' if key=='model.embed_tokens.weight' else 'lm_head.weight'
            value[text_vocab:text_vocab+qv,off:off+ph]=prefix.tensor(pk).to(device=device,dtype=torch.float32)
            if key=='model.embed_tokens.weight':
                # Match the prefix's fixed norm scale with sixteen ballast
                # coordinates in the full 4096-dimensional residual stream.
                original_ballast=len(proto.get('ballast_coordinates',[0,1,ph-1]))
                extra=16-original_ballast
                assert off+ph+extra<4094
                value[text_vocab:text_vocab+qv,off+ph:off+ph+extra]=2.0**40
                value[:text_vocab,tflag]=2.0**-40
                value[text_vocab:text_vocab+qv,qflag]=1.
            else:
                value[:text_vocab,qflag]=-2.0**60
                value[text_vocab:text_vocab+qv,tflag]=-2.0**60
        elif key=='model.norm.weight':
            w=base.tensor(key,slice(0,d0)).to(device=device,dtype=torch.float32)
            value[:d0]=(1+w)*math.sqrt(oldh/cfg.hidden_size)-1
        else:
            match=re.match(r'model.layers\.(\d+)\.(.*)',key)
            if not match:raise ValueError('Unmapped standard parameter: '+key)
            layer,suffix=int(match[1]),match[2]
            if layer<prefix_layers:
                if suffix.startswith('self_attn.'):
                    # Both attention sublayers in the prefix are zero maps.
                    pass
                else:
                    w=prefix.tensor(key).to(device=device,dtype=torch.float32)
                    if suffix.endswith('layernorm.weight'):value[off:off+ph]=w
                    elif suffix in ['mlp.gate_proj.weight','mlp.up_proj.weight']:
                        value[:w.shape[0],off:off+ph]=w
                    elif suffix=='mlp.down_proj.weight':value[off:off+ph,:w.shape[1]]=w
                    elif w.ndim==2 and w.shape[1]==ph:value[:,off:off+ph]=w
                    elif w.ndim==2 and w.shape[0]==ph:value[off:off+ph,:]=w
                    elif tuple(shape)==tuple(w.shape):value.copy_(w)
                    else:raise ValueError('Unmapped QPU tensor '+key)
            else:
                source=f'model.layers.{layer-prefix_layers+11}.{suffix}'
                if suffix.endswith('layernorm.weight'):
                    w=base.tensor(source,slice(0,d0)).to(device=device,dtype=torch.float32)
                    value[:d0]=(1+w)*math.sqrt(oldh/cfg.hidden_size)-1
                elif suffix=='linear_attn.in_proj_qkv.weight':
                    w=base.tensor(source,(slice(None),slice(0,d0))).to(device=device,dtype=torch.float32)
                    value[:oldk,:d0]=w[:oldk]
                    value[newk:newk+oldk,:d0]=w[oldk:2*oldk]
                    value[2*newk:2*newk+oldv,:d0]=w[2*oldk:]
                elif suffix=='linear_attn.conv1d.weight':
                    w=base.tensor(source).to(device=device,dtype=torch.float32)
                    value[:oldk]=w[:oldk];value[newk:newk+oldk]=w[oldk:2*oldk]
                    value[2*newk:2*newk+oldv]=w[2*oldk:]
                else:
                    w=base.tensor(source)
                    if w.ndim==2 and w.shape[1]==oldh:
                        w=base.tensor(source,(slice(None),slice(0,d0))).to(device=device,dtype=torch.float32)
                        value[:w.shape[0],:d0]=w
                    elif w.ndim==2 and w.shape[0]==oldh:
                        w=base.tensor(source,(slice(0,d0),slice(None))).to(device=device,dtype=torch.float32)
                        value[:d0,:w.shape[1]]=w
                    elif tuple(w.shape)==tuple(shape):value.copy_(w.to(device=device,dtype=torch.float32))
                    elif w.ndim==1 and w.shape[0]<=shape[0]:value[:w.shape[0]]=w.to(device=device,dtype=torch.float32)
                    else:raise ValueError('Unmapped language tensor '+source)
        _assign(model,key,value)
        audit['mapped_tensors']+=1
        if progress is not None:progress(key,audit['mapped_tensors'],len(keys))
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    model.model.rotary_emb=Qwen3_5TextRotaryEmbedding(config=cfg,device=device)
    assert all(p.device.type!='meta' for p in model.parameters())
    model.eval().requires_grad_(False)
    return model,audit


class FullModelTokenDriver:
    """Offset-only adapter; all model calls execute the complete standard model."""
    def __init__(self,model,layout):
        from token_checkpoint import AutonomousDriver
        self.driver=AutonomousDriver(model,layout)
        offset=model.config.qwenqpu_added_layer_prototype['protocol_token_offset']
        qoff=model.config.qwenqpu_added_layer_prototype['quantum_offset']
        def step(tid,uniform=None,event_mask=None):
            device=next(model.parameters()).device
            with torch.inference_mode():
                ids=torch.tensor([[offset+tid]],device=device)
                if uniform is None and event_mask is None:kw={'input_ids':ids}
                else:
                    x=model.model.embed_tokens(ids).clone()
                    if uniform is not None:x[0,0,qoff+layout.uniform]=uniform
                    if event_mask is not None:
                        x[0,0,qoff+layout.event_mask:qoff+layout.event_mask+layout.size]=torch.tensor(event_mask,device=device)
                    kw={'inputs_embeds':x}
                out=model(**kw,past_key_values=self.driver.cache,use_cache=True,logits_to_keep=1)
                self.driver.cache=out.past_key_values
                result=int(out.logits[0,-1].argmax())-offset
                if not 0<=result<layout.vocab:
                    raise RuntimeError('Complete model emitted a language token during QPU execution')
                return result
        self.driver.step=step


class FullCompactTokenDriver:
    """Token-offset adapter for the compact checkpoint in a complete stock LM."""
    def __init__(self,model,layout):
        from compact_checkpoint import CompactDriver
        self.driver=CompactDriver(model,layout)
        meta=model.config.qwenqpu_added_layer_prototype
        offset=meta['protocol_token_offset'];qoff=meta['quantum_offset']
        def forward_step(tid,uniform=None):
            device=next(model.parameters()).device
            with torch.inference_mode():
                ids=torch.tensor([[offset+tid]],device=device);kw={'input_ids':ids}
                if uniform is not None:
                    x=model.model.embed_tokens(ids).clone();x[0,0,qoff+layout.uniform]=uniform
                    kw={'inputs_embeds':x}
                out=model(**kw,past_key_values=self.driver.cache,use_cache=True,logits_to_keep=1)
                self.driver.cache=out.past_key_values
                result=int(out.logits[0,-1].argmax())-offset
                if not 0<=result<layout.vocab:raise RuntimeError('Language token during compact quantum execution')
                return result
        self.driver._forward_step=forward_step

