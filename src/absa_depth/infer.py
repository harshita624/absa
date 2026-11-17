"""Batched exit simulation and BERT incremental early-exit inference."""
from __future__ import annotations
import torch
from .gate import halting_distribution


def simulate_exit(mu: torch.Tensor, lambdas: torch.Tensor, tau: float):
    """Select each aspect's first threshold crossing, falling back to final layer."""
    # mu: (layers, aspects, 2), lambdas: (layers, aspects)
    hits = lambdas > tau
    any_hit = hits.any(0)
    depth = torch.where(any_hit, hits.float().argmax(0)+1, torch.full_like(hits[0].long(), mu.shape[0]))
    pred = mu[(depth-1).long(), torch.arange(mu.shape[1], device=mu.device)]
    return pred, depth


def simulate_variance_exit(mu: torch.Tensor, logvar: torch.Tensor, theta: float):
    """Exit at the first layer whose mean predictive variance is below theta."""
    variance=logvar.exp().mean(dim=-1)
    hits=variance < theta
    any_hit=hits.any(0)
    depth=torch.where(any_hit,hits.float().argmax(0)+1,
                      torch.full_like(hits[0].long(),mu.shape[0]))
    pred=mu[(depth-1).long(),torch.arange(mu.shape[1],device=mu.device)]
    return pred,depth


@torch.inference_mode()
def predict_batched(model, batch, tau: float):
    """Full-depth computation followed by threshold simulation; returns original scale."""
    model.eval(); mus, _, gates, _ = model(batch)
    pred, depth = simulate_exit(mus, gates, tau)
    return pred*4+5, depth


@torch.inference_mode()
def predict_true_early(model, tokenizer, text: str, aspects: list[str], tau: float, device: torch.device):
    """Run BERT layers incrementally for one sentence and stop at deepest aspect exit."""
    model.eval()
    if not hasattr(model.encoder, "encoder") or not hasattr(model.encoder, "embeddings"):
        raise NotImplementedError("true incremental inference is implemented for BERT encoders")
    encoded = tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=256)
    offsets = encoded.pop("offset_mapping").to(device)
    encoded = {k:v.to(device) for k,v in encoded.items()}
    ext_mask = model.encoder.get_extended_attention_mask(encoded["attention_mask"], encoded["input_ids"].shape,
                                                          device=device)
    hidden = model.encoder.embeddings(input_ids=encoded["input_ids"],
                                      token_type_ids=encoded.get("token_type_ids"))
    spans=[]
    for aspect in aspects:
        p=text.casefold().find(aspect.casefold()) if aspect and aspect.upper()!="NULL" else -1
        spans.append((max(p,0), max(p,0)+len(aspect) if p>=0 else 0))
    starts=torch.tensor([x[0] for x in spans],device=device); ends=torch.tensor([x[1] for x in spans],device=device)
    amask=(offsets[0,:,1][None,:]>starts[:,None]) & (offsets[0,:,0][None,:]<ends[:,None]) & (ends>starts)[:,None]
    amask[~amask.any(1),0]=True
    predictions=[None]*len(aspects); depths=torch.zeros(len(aspects),device=device,dtype=torch.long)
    prev=None; indices=torch.zeros(len(aspects),device=device,dtype=torch.long)
    local={"attention_mask":encoded["attention_mask"],"sentence_index":indices,"offset_mapping":offsets,
           "char_starts":starts,"char_ends":ends}
    for i,layer in enumerate(model.encoder.encoder.layer):
        result=layer(hidden, attention_mask=ext_mask)
        hidden=result[0] if isinstance(result,tuple) else result
        r,_=model.readout[i](hidden, encoded["attention_mask"], indices, amask)
        raw=model.heads[i](r); mu,logvar=raw[:,:2],raw[:,2:].clamp(-6,4)
        delta=torch.zeros_like(mu) if prev is None else (mu-prev).abs()
        lam=model.gate(r,logvar.detach() if model.gate_detach_inputs else logvar,
                       delta.detach() if model.gate_detach_inputs else delta,i).squeeze(-1)
        done=(lam>tau) | (i==model.layers-1)
        for j in torch.where(done & (depths==0))[0].tolist():
            predictions[j]=mu[j]; depths[j]=i+1
        if bool((depths>0).all()): break
        prev=mu
    pred=torch.stack(predictions)*4+5
    return pred,depths
