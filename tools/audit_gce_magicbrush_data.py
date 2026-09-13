#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer
from datasets.magicbrush_tokens import MagicBrushTokenDataset

def digest(row):
    return hashlib.sha256(json.dumps({k:row[k] for k in ('input_ids','labels','conditional','sample_key')},sort_keys=True).encode()).hexdigest()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',type=Path,required=True); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=True)
    left=MagicBrushTokenDataset(a.manifest,tok,seed=a.seed); right=MagicBrushTokenDataset(a.manifest,tok,seed=a.seed)
    for epoch in (0,1):
        left.set_epoch(epoch); right.set_epoch(epoch)
        hashes=[digest(left[i]) for i in range(min(64,len(left)))]
        if hashes != [digest(right[i]) for i in range(min(64,len(right)))]: raise SystemExit('64-sample deterministic hash failed')
        print(json.dumps({'epoch':epoch,'hash':hashlib.sha256(''.join(hashes).encode()).hexdigest()}))
    lengths=sorted(left[i]['total_sequence_length'] for i in range(len(left)))
    q=lambda r:lengths[min(len(lengths)-1,round(r*(len(lengths)-1)))]
    print(json.dumps({'num_samples':len(lengths),'min':lengths[0],'mean':sum(lengths)/len(lengths),'median':q(.5),'p95':q(.95),'p99':q(.99),'max':lengths[-1]}))
if __name__=='__main__': main()
