*this is garbage file i've been using as log, don't pay attention to it*


one gpu:

nvprof --devices 1 --kernels '^((?!nccl).)*$' --profile-child-processes --print-gpu-trace --normalized-time-unit ms --profile-from-start off --concurrent-kernels on --metrics l2_tex_read_hit_rate,l2_tex_write_hit_rate,l2_tex_read_transactions,l2_tex_write_transactions --csv --log-file ./csv/resnet50/one%p.csv python3 onegpu.py

multi gpu:

nvprof --devices 1 --kernels '^((?!nccl).)*$' --profile-child-processes --print-gpu-trace --normalized-time-unit ms --profile-from-start off --concurrent-kernels on --metrics l2_tex_read_hit_rate,l2_tex_write_hit_rate,l2_tex_read_transactions,l2_tex_write_transactions --csv --log-file ./csv/resnet50/multi%p.csv python3 multigpu.py

________________________________________________


'^((?!nccl).)*$'
'^.*nccl.*$'


multigpu with allreduce


nvprof --devices 0,1,2,3 --profile-child-processes --profile-from-start off --concurrent-kernels on --metrics l2_read_transactions --csv --print-gpu-trace --trace gpu --normalized-time-unit ms --log-file ./csv/multigpu_and_allreduce/l2_read_t%p.csv python3 multigpu.py



metrics:

L2R - l2_read_transactions
L12R - l2_tex_read_transactions
L2W - l2_write_transactions
L12W - l2_tex_write_transactions

l2_read_transactions,
l2_write_transactions,
dram_read_bytes
dram_write_bytes


nvprof --devices 0,1,2,3 --kernels '^((?!nccl).)*$' --profile-child-processes --profile-from-start off --concurrent-kernels on --metrics l2_tex_read_hit_rate,l2_tex_read_transactions,l2_tex_write_hit_rate,l2_tex_write_transactions --csv --print-gpu-trace --trace gpu --normalized-time-unit ms --log-file ./csv/l12_one_resnet50%p.csv python3 onegpu.py


nvprof --devices 0,1,2,3 --kernels '^((?!nccl).)*$' --profile-child-processes --profile-from-start off --concurrent-kernels on --metrics l2_tex_read_hit_rate,l2_tex_read_transactions,l2_tex_write_hit_rate,l2_tex_write_transactions --csv --print-gpu-trace --trace gpu --normalized-time-unit ms --log-file ./csv/l12_multi_resnet50%p.csv python3 multigpu.py

______

nvprof --devices 0,1,2,3 --kernels '^((?!nccl).)*$' --profile-child-processes --profile-from-start off --concurrent-kernels on --metrics l2_read_transactions,l2_global_load_bytes,l2_local_load_bytes,l2_surface_load_bytes --csv --print-gpu-trace --trace gpu --normalized-time-unit ms --log-file ./csv/bytes_check%p.csv python3 onegpu.py


events: all you need to know to find hit rate

sector 0 of L2:

l2_subp0_write_sector_misses
l2_subp0_read_sector_misses

l2_subp0_total_read_sector_queries
l2_subp0_total_write_sector_queries

sector 1 of L2:

l2_subp1_write_sector_misses
l2_subp1_read_sector_misses

l2_subp1_total_read_sector_queries
l2_subp1_total_write_sector_queries


_________
