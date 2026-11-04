"""Native REJECT outcome and envelope sampling for quantized Born probabilities.

Proposals are uniform on [0, 2). Given nonnegative probabilities with total
mass <= 2, accepting a native outcome and retrying native REJECT samples the
normalized represented distribution. The host supplies randomness and counts
outcomes; it never computes an amplitude, probability, CDF or acceptance test.
This is a new sampling protocol, not a rescore of the retained [0,1) tests.
"""
import collections
import hashlib
import math
import torch
from token_checkpoint import B, MASK, Q, RADIX
from compact_checkpoint import CompactDriver
from compact_sampling_digits import apply_digit_cdf_patch


def apply_rejection_patch(model, layout):
    l = layout
    meta = model.config.qwenqpu_added_layer_prototype
    if meta.get('sampling_revision') != 'digit-lexicographic-r6':
        raise ValueError('Requires the separately tested digit comparator')
    last_flag = meta['uniform_digit_coordinates'][0] + 8
    if last_flag >= 1021:
        raise ValueError('No free last-address flag')
    emb, head = model.model.embed_tokens.weight, model.lm_head.weight
    reject = model.config.vocab_size
    new_emb = torch.nn.Embedding(reject+1, l.hidden, device=emb.device, dtype=emb.dtype)
    new_head = torch.nn.Linear(l.hidden, reject+1, bias=False, device=head.device, dtype=head.dtype)
    with torch.no_grad():
        assert not torch.count_nonzero(emb[:,last_flag])
        new_emb.weight[:reject].copy_(emb)
        new_head.weight[:reject].copy_(head)
        new_emb.weight[reject].copy_(emb[l.halt_id])
        new_emb.weight[l.sample_start+l.size-1,last_flag] = 1
        new_head.weight[reject].zero_()
        new_head.weight[reject,0] = (-.5-MASK)/B
        new_head.weight[reject,last_flag] = MASK
        new_head.weight[reject,l.sample_penalty] = 1
        new_head.weight[l.continue_id,last_flag] = -1
    model.model.embed_tokens = new_emb.requires_grad_(False)
    model.lm_head = new_head.requires_grad_(False)
    model.config.vocab_size = reject+1
    model.vocab_size = reject+1
    meta.update(sampling_revision='native-rejection-r7', sampling_envelope=2,
                sampling_reject_token=reject, sampling_last_address_flag=last_flag)
    return dict(version=meta['sampling_revision'], reject_token=reject, envelope=2,
                added_embedding_and_output_rows=1, custom_model_forward=False,
                source_sha256=hashlib.sha256(open(__file__,'rb').read()).hexdigest())


class RejectionDriver(CompactDriver):
    def _forward_step(self, tid, uniform=None):
        if uniform is None:
            return super()._forward_step(tid)
        if not 0 <= float(uniform) < 2:
            raise ValueError('Proposal must lie in [0,2)')
        ids = torch.tensor([[tid]], device=next(self.model.parameters()).device)
        x = self.model.model.embed_tokens(ids).clone()
        x[0,0,self.lay.uniform] = uniform
        digits = divmod(math.floor(Q*float(uniform))+int(Q), RADIX)
        for coordinate,digit in zip(self.model.config.qwenqpu_added_layer_prototype['uniform_digit_coordinates'],digits):
            x[0,0,coordinate] = digit
        with torch.inference_mode():
            out=self.model(inputs_embeds=x,position_ids=torch.zeros_like(ids),
                attention_mask={'linear_attention':None,'full_attention':None},
                past_key_values=self.cache,use_cache=True,logits_to_keep=1)
        self.cache=out.past_key_values
        return int(out.logits[0,-1].argmax())

    def proposal(self, uniform):
        # Same host address/query schedule as CompactDriver.probability. All
        # cumulative sums, strict comparisons and REJECT selection are native.
        l = self.lay
        reject = self.model.config.qwenqpu_added_layer_prototype['sampling_reject_token']
        self.step(l.halt_id)
        self.step(l.prob_reset)
        trace=[]
        for a in range(l.size):
            for ph in range(2):
                rid=4*a+2*ph+1
                query=self.step(l.query_select_start+rid)
                if not l.query_start+2*rid <= query < l.query_start+2*rid+2:
                    raise RuntimeError('Invalid native query')
                digit=self.step(query)
                if digit//RADIX != ph:
                    raise RuntimeError('Invalid native probability digit')
                self.step(digit)
            output=self.step(l.sample_start+a, uniform)
            trace.append([l.sample_start+a,output])
            if l.outcome_start <= output < l.outcome_start+l.size:
                return dict(outcome=output-l.outcome_start, rejected=False, decisions=trace)
            if output == reject:
                if a != l.size-1:
                    raise RuntimeError('Premature native REJECT')
                return dict(outcome=None, rejected=True, decisions=trace)
            if output != l.continue_id:
                raise RuntimeError('Invalid native sample decision')
        raise RuntimeError('Missing native terminal decision')

    def sample_shots(self, shots, rng, max_proposals_per_shot=64):
        if type(shots) is not int or shots < 1:
            raise ValueError('Positive integer shots required')
        counts=collections.Counter();records=[]
        for shot in range(shots):
            for attempt in range(max_proposals_per_shot):
                uniform=2*float(rng.random())
                result=self.proposal(uniform)
                records.append(dict(shot=shot,attempt=attempt,uniform=uniform,**result))
                if not result['rejected']:
                    counts[format(result['outcome'],f'0{self.lay.n}b')]+=1
                    break
            else:
                raise RuntimeError('Proposal budget exhausted; no fabricated outcome')
        return dict(counts=dict(sorted(counts.items())),shots=shots,proposals=records)
