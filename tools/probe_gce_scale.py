#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from transformers import AutoTokenizer
from datasets.magicbrush_tokens import MagicBrushTokenDataset
from model import LLaDAForMultiModalGeneration
def main():
 p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--manifest',required=True); p.add_argument('--clusters',required=True); p.add_argument('--batches',type=int,default=10); a=p.parse_args()
 t=AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=True); d=MagicBrushTokenDataset(Path(a.manifest),t,seed=42); m=LLaDAForMultiModalGeneration.from_pretrained(a.model,torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True).to('cuda').train(); m.configure_gce(a.clusters)
 rows=[]
 for i in range(a.batches):
  _,x=m(input_ids=[d[i]['input_ids']],labels=[d[i]['labels']],use_gce=True,gce_logit_grad_probe=True); rows.append({k:float(v) for k,v in x.items()})
 summary={k:{'mean':float(np.mean([r[k] for r in rows])),'median':float(np.median([r[k] for r in rows])),'p90':float(np.quantile([r[k] for r in rows],.9))} for k in ('ce_loss','gce_loss','gce_to_ce_logit_grad_norm')}; ratios=[r['gce_loss']/r['ce_loss'] for r in rows]; summary['gce_to_ce_loss_ratio']={q:float(f(ratios)) for q,f in {'mean':np.mean,'median':np.median,'p90':lambda x:np.quantile(x,.9)}.items()}; print(json.dumps({'summary':summary,'rows':rows},indent=2));
 if summary['gce_to_ce_logit_grad_norm']['p90']>10 or summary['gce_to_ce_loss_ratio']['p90']>10: raise SystemExit('GCE scale gate failed')
if __name__=='__main__': main()
