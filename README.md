To profile l2 cache hit rate with ncu, do the following (just example):


one gpu:

nvprof --devices 0 --kernels '::^((?!nccl).)*$:' --profile-child-processes --print-gpu-trace --profile-from-start off --concurrent-kernels on --metrics l2_read_transactions,l2_tex_read_transactions,l2_tex_read_hit_rate,l2_global_load_bytes,tex_cache_hit_rate --csv --log-file onegpu%p.csv python3 onegpu.py


multi gpu:

nvprof --devices 0,1,2,3 --kernels '::^((?!nccl).)*$:' --profile-child-processes --print-gpu-trace --profile-from-start off --concurrent-kernels on --metrics l2_read_transactions,l2_tex_read_transactions,l2_tex_read_hit_rate,l2_global_load_bytes,tex_cache_hit_rate --csv --log-file multigpu%p.csv python3 multigpu.py







