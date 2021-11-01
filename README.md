To profile l2 cache hit rate with ncu, do the following (just example):


one gpu:
nvprof --devices 2 --kernels '::^((?!nccl).)*$:' --profile-child-processes --print-gpu-trace --profile-from-start off --skip-kernel-replay-save-restore on --concurrent-kernels on --metrics l2_tex_hit_rate --csv --log-file onegpu%p.csv python3 onegpu.py


multi gpu:

nvprof --devices 2,3 --kernels '::^((?!nccl).)*$:' --profile-child-processes --print-gpu-trace --profile-from-start off --skip-kernel-replay-save-restore on --concurrent-kernels on --metrics l2_tex_hit_rate --csv --log-file twogpus%p.csv python3 multigpu.py