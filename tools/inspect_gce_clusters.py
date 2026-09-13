#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from diffusers import VQModel

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--clusters',required=True); p.add_argument('--device',default='cuda:0'); a=p.parse_args()
    data=torch.load(a.clusters,map_location='cpu',weights_only=True); x=VQModel.from_pretrained(a.model,subfolder='vqvae',local_files_only=True).quantize.embedding.weight.float()
    g=torch.Generator().manual_seed(0)
    result={}
    for k,level in data['levels'].items():
        ids=level['token_to_cluster']; cmap=level['cluster_map']; sizes=level['cluster_sizes']
        intra=[]
        for token in torch.randperm(len(ids),generator=g)[:256]:
            c=ids[token]; members=cmap[c,:sizes[c]]; intra.append((x[token]-x[members]).norm(dim=-1).mean())
        pairs=torch.randint(len(x),(256,2),generator=g); rnd=(x[pairs[:,0]]-x[pairs[:,1]]).norm(dim=-1).mean()
        result[int(k)]={'intra_distance':float(torch.stack(intra).mean()),'random_distance':float(rnd),'intra_lt_random':bool(torch.stack(intra).mean()<rnd)}
    print(json.dumps(result,indent=2));
    if not all(v['intra_lt_random'] for v in result.values()): raise SystemExit('cluster quality sanity failed')
if __name__=='__main__': main()
