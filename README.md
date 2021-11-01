To profile l2 cache hit rate with ncu, do the following (just example):


one gpu:
nvprof --devices 2,3 --kernels '::^((?!nccl).)*$:' --profile-child-processes --profile-from-start off --skip-kernel-replay-save-restore on --concurrent-kernels on --metrics l2_tex_hit_rate --quiet --csv --log-file twogpus.csv python3 multigpu.py