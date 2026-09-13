#!/usr/bin/env python3
from __future__ import annotations
import argparse, contextlib, json, os, random, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np, torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from attention_supervision.lora import inject_lora, load_lora_state_dict, lora_state_dict
from datasets.magicbrush_tokens import MagicBrushTokenDataset
from model import LLaDAForMultiModalGeneration

def main():
 p=argparse.ArgumentParser(); p.add_argument('--model',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,required=True);p.add_argument('--accum',type=int,required=True);p.add_argument('--steps',type=int,required=True);p.add_argument('--save-every',type=int,default=0);p.add_argument('--max-seq-len',type=int,default=3072);p.add_argument('--use-gce',action='store_true');p.add_argument('--clusters');p.add_argument('--resume',type=Path);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
 dist.init_process_group('nccl'); rank=dist.get_rank(); world=dist.get_world_size(); local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local);dev=torch.device('cuda',local)
 random.seed(a.seed+rank);np.random.seed(a.seed+rank);torch.manual_seed(a.seed+rank);torch.cuda.manual_seed_all(a.seed+rank)
 if rank==0:
  if a.output.exists() and not a.resume:
   raise FileExistsError(f"output already exists; use a new --output or pass --resume: {a.output}")
  a.output.mkdir(parents=True,exist_ok=bool(a.resume))
 dist.barrier();tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=True);d=MagicBrushTokenDataset(Path(a.manifest),tok,seed=a.seed,max_sequence_length=a.max_seq_len);sampler=DistributedSampler(d,num_replicas=world,rank=rank,shuffle=True,seed=a.seed,drop_last=True);loader=DataLoader(d,batch_size=a.batch_size,sampler=sampler,drop_last=True,num_workers=2,collate_fn=lambda x:x,pin_memory=True)
 m=LLaDAForMultiModalGeneration.from_pretrained(a.model,torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True);inject_lora(m,rank=16,alpha=16.,dropout=.05)
 if a.use_gce:
  if not a.clusters: raise ValueError('--clusters required with --use-gce')
  m.configure_gce(a.clusters)
 m.model.set_activation_checkpointing('whole_layer');m.to(dev);opt=torch.optim.AdamW([x for x in m.parameters() if x.requires_grad],lr=1e-5,betas=(.9,.95),weight_decay=.1);step=0
 if a.resume:
  load_lora_state_dict(m,torch.load(a.resume/'lora.pt',map_location='cpu',weights_only=True)); state=torch.load(a.resume/'state.pt',map_location='cpu');opt.load_state_dict(state['optimizer']);step=int(state['step'])
 m=DDP(m,device_ids=[local],broadcast_buffers=False);m.train();opt.zero_grad(set_to_none=True);micro=step*a.accum;start=time.time();torch.cuda.reset_peak_memory_stats(dev)
 while step<a.steps:
  d.set_epoch(micro//max(1,len(loader)));sampler.set_epoch(micro//max(1,len(loader)))
  for rows in loader:
   boundary=(micro+1)%a.accum==0; ctx=contextlib.nullcontext() if boundary else m.no_sync()
   with ctx,torch.autocast('cuda',dtype=torch.bfloat16):
    out=m(input_ids=[r['input_ids'] for r in rows],labels=[r['labels'] for r in rows],use_gce=a.use_gce)
    loss,metrics=out if a.use_gce else (out,{})
    if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
    (loss/a.accum).backward()
   micro+=1
   if not boundary:continue
   norm=torch.nn.utils.clip_grad_norm_([x for x in m.parameters() if x.requires_grad],4.0);opt.step();opt.zero_grad(set_to_none=True);step+=1
   record={'step':step,'loss':float(loss),'metrics':{k:float(v) for k,v in metrics.items()},'grad_norm':float(norm),'sec_per_step':(time.time()-start)/max(step-(int(a.resume is not None)*20),1)}
   if rank==0:
    print(json.dumps(record),flush=True)
    with (a.output/'step_metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
   if (a.save_every and step%a.save_every==0) or step==a.steps:
    if rank==0:
     dest=a.output/f'checkpoint-{step:06d}';dest.mkdir(exist_ok=True);torch.save(lora_state_dict(m.module),dest/'lora.pt');torch.save({'optimizer':opt.state_dict(),'step':step},dest/'state.pt')
     if step==a.steps:(a.output/'result.json').write_text(json.dumps({'steps':step,'peak_gib':torch.cuda.max_memory_allocated(dev)/2**30,'seconds_per_step':(time.time()-start)/max(step-(int(a.resume is not None)*20),1)})+'\n')
   if step==a.steps:
    dist.barrier();dist.destroy_process_group();return
if __name__=='__main__':main()
