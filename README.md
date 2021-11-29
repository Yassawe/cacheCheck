*this is garbage file i've been using as log, don't pay attention to it*


one gpu:


nvprof --devices 1 --profile-child-processes --print-gpu-trace --normalized-time-unit ms --profile-from-start off --concurrent-kernels on --metrics l2_tex_write_hit_rate --csv --log-file ./csv/hitrates/101_sw%p.csv python3 onegpu.py

multi gpu:

nvprof --devices 0,1,2,3 --kernels '^((?!nccl).)*$' --profile-child-processes --print-gpu-trace --normalized-time-unit ms --profile-from-start off --concurrent-kernels on --metrics l2_tex_write_hit_rate  --csv --log-file ./csv/hitrates/101_mw%p.csv python3 multigpu.py

________________________________________________



metrics:

l2_tex_hit_rate

l2_tex_read_hit_rate
l2_tex_write_hit_rate

l2_tex_read_transactions
l2_tex_write_transactions


l2_read_transactions,
l2_write_transactions,
dram_read_bytes
dram_write_bytes

_______________

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
