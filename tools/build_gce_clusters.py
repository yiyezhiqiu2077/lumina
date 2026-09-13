#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from diffusers import VQModel

def kmeans(x, k, seed, n_init=3, iters=50):
    best = None
    for trial in range(n_init):
        gen = torch.Generator(device=x.device).manual_seed(seed + trial)
        centers = x[torch.randperm(len(x), generator=gen, device=x.device)[:k]].clone()
        for _ in range(iters):
            labels = torch.cdist(x, centers).argmin(1)
            counts = torch.bincount(labels, minlength=k)
            sums = torch.zeros_like(centers).index_add_(0, labels, x)
            new = sums / counts.clamp_min(1)[:, None]
            empty = counts == 0
            if empty.any(): new[empty] = x[torch.randperm(len(x), generator=gen, device=x.device)[:int(empty.sum())]]
            if torch.allclose(centers, new, atol=1e-5): break
            centers = new
        labels = torch.cdist(x, centers).argmin(1); inertia = (x - centers[labels]).square().sum()
        if best is None or inertia < best[0]: best = (inertia, labels)
    return best[1]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--device',default='cuda:0'); a=p.parse_args()
    vq=VQModel.from_pretrained(a.model,subfolder='vqvae',local_files_only=True).to(a.device)
    x=vq.quantize.embedding.weight.detach().float(); print(json.dumps({'path':'vqvae.quantize.embedding.weight','shape':list(x.shape),'dtype':str(x.dtype),'min':x.min().item(),'max':x.max().item(),'mean_norm':x.norm(dim=1).mean().item()}))
    levels={}
    for k in (1024,512):
        ids=kmeans(x,k,0); groups=[torch.where(ids==i)[0] for i in range(k)]
        if any(len(g)==0 for g in groups): raise RuntimeError(f'empty kmeans cluster K={k}')
        max_size=max(len(g) for g in groups); cmap=torch.full((k,max_size),-1,dtype=torch.long)
        for i,g in enumerate(groups): cmap[i,:len(g)]=g.cpu()
        sizes=torch.tensor([len(g) for g in groups],dtype=torch.long)
        levels[k]={'token_to_cluster':ids.cpu().long(),'cluster_map':cmap,'cluster_sizes':sizes}
        print(json.dumps({'K':k,'min':sizes.min().item(),'max':sizes.max().item(),'mean':sizes.float().mean().item(),'median':sizes.float().median().item(),'std':sizes.float().std().item()}))
    a.output.parent.mkdir(parents=True,exist_ok=True); torch.save({'codebook_size':len(x),'embedding_dim':x.shape[1],'embedding_path':'vqvae.quantize.embedding.weight','levels':levels},a.output)
if __name__=='__main__': main()
