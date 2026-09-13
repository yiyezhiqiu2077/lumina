# Distributed training

Multi-node training: pending alignment with 谭樾.

Current single-node references are:

| Experiment | GPUs | Batch/GPU | Accumulation | Global batch |
| --- | ---: | ---: | ---: | ---: |
| CE / GCE | 4 | 4 | 2 | 32 |
| attention supervision | 2 | 8 | 4 | 64 |

For a possible 2-node x 4-GPU run, `8 x batch4 x accum1 = 32` is only a
candidate for the CE/GCE global batch and must be confirmed before use. Confirm
the launcher (`torchrun` or `srun`), GPUs per node, NCCL network settings,
shared filesystem, `DistributedSampler` epoch/cursor semantics, checkpoint
resume, scheduler behavior, and rank-0 save policy with 谭樾. Do not hard-code a
multi-node launcher until those choices are agreed.
