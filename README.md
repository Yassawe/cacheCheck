To profile l2 cache hit rate with nvprof, do the following (just example):


one gpu:

nvprof --devices 0 --kernels '^((?!nccl).)*$' --profile-child-processes --print-gpu-trace --profile-from-start off --concurrent-kernels on --metrics l2_tex_read_transactions,l2_utilization,l2_tex_read_hit_rate --csv --log-file ./csv/nvprof_onegpu_vgg_entire%p.csv python3 onegpu.py

ncu --devices 0 --target-processes all --csv --replay-mode application --page details --log-file onegpu_entire_vgg16.csv --nvtx --profile-from-start no --metrics lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sector_op_read_hit_rate.pct,lts__t_sectors.avg.pct_of_peak_sustained_elapsed python3 onegpu.py



multi gpu:

nvprof --devices 0,1,2,3 --kernels '^((?!nccl).)*$' --profile-child-processes --print-gpu-trace --profile-from-start off --concurrent-kernels on --metrics l2_tex_read_transactions,l2_utilization,l2_tex_read_hit_rate --csv --log-file multigpu%p.csv python3 multigpu.py



multigpu nooverlap:

nvprof --devices 0 --kernels '^((?!nccl).)*$' --profile-child-processes --print-gpu-trace --profile-from-start off --concurrent-kernels on --metrics l2_tex_read_transactions,l2_utilization,l2_tex_read_hit_rate --csv --log-file ./csv/nvprof_multigpu_nooverlap_vgg%p.csv python3 multigpu_nooverlap.py

ncu --devices 0 --kernel-name regex:'^((?!nccl).)*$' --target-processes all --csv --replay-mode application --page details --log-file multigpu_nooverlap.csv --nvtx --profile-from-start no --metrics lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sector_op_read_hit_rate.pct,lts__t_sectors.avg.pct_of_peak_sustained_elapsed python3 multigpu_nooverlap.py




