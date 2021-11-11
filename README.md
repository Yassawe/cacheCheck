*this is garbage file i've been using as log, don't pay attention to it*


one gpu:

nvprof --devices 0 --kernels '^((?!nccl).)*$' --profile-child-processes --print-gpu-trace --profile-from-start off --concurrent-kernels on --metrics l2_tex_read_transactions,l2_utilization,l2_tex_read_hit_rate --csv --log-file ./csv/nvprof_onegpu_vgg_entire%p.csv python3 onegpu.py

ncu --devices 0 --target-processes all --csv --replay-mode application --page details --log-file onegpu_entire_vgg16.csv --nvtx --profile-from-start no --metrics lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sector_op_read_hit_rate.pct,lts__t_sectors.avg.pct_of_peak_sustained_elapsed python3 onegpu.py



multi gpu:

nvprof --devices 0,1,2,3 --profile-child-processes --profile-from-start off --concurrent-kernels on --metrics l2_write_transactions --csv --print-gpu-trace --trace gpu --normalized-time-unit ms --log-file ./csv/nvprof_l2w_multigpu_vgg%p.csv python3 multigpu.py


nvprof --devices 0,1,2,3 --kernels '^.*nccl.*$' --replay-mode disabled --profile-child-processes --profile-from-start off --concurrent-kernels on --metrics l2_read_transactions,l2_tex_read_transactions,l2_write_transactions,l2_tex_write_transactions --csv --print-gpu-trace --trace gpu --normalized-time-unit ms python3 multigpu.py


--replay-mode disabled

'^((?!nccl).)*$'
'^.*nccl.*$'


multigpu nooverlap:

nvprof --devices 0 --kernels '^((?!nccl).)*$' --profile-child-processes --print-gpu-trace --profile-from-start off --concurrent-kernels on --metrics l2_tex_read_transactions,l2_utilization,l2_tex_read_hit_rate --csv --log-file ./csv/nvprof_multigpu_nooverlap_vgg%p.csv python3 multigpu_nooverlap.py

ncu --devices 0 --kernel-name regex:'^((?!nccl).)*$' --target-processes all --csv --replay-mode application --page details --log-file l12_multi_rtx.csv --nvtx --profile-from-start no --metrics lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sector_op_read_hit_rate.pct,lts__t_sectors.avg.pct_of_peak_sustained_elapsed python3 multigpu.py


metrics:

L2R - l2_read_transactions
L12R - l2_tex_read_transactions
L2W - l2_write_transactions
L12W - l2_tex_write_transactions

l2_read_transactions,l2_tex_read_transactions,l2_write_transactions,l2_tex_write_transactions


current work:

nvprof --devices 0,1,2,3 --kernels '^((?!nccl).)*$' --profile-child-processes --profile-from-start off --concurrent-kernels on --metrics l2_tex_read_hit_rate,l2_tex_read_transactions,l2_tex_write_hit_rate,l2_tex_write_transactions --csv --print-gpu-trace --trace gpu --normalized-time-unit ms --log-file ./csv/l12_one%p.csv python3 onegpu.py


nvprof --devices 0,1,2,3 --kernels '^((?!nccl).)*$' --profile-child-processes --profile-from-start off --concurrent-kernels on --metrics l2_tex_read_hit_rate,l2_tex_read_transactions,l2_tex_write_hit_rate,l2_tex_write_transactions --csv --print-gpu-trace --trace gpu --normalized-time-unit ms --log-file ./csv/l12_multi%p.csv python3 multigpu.py


write trans

ncu --devices 0,1,2,3 --kernel-name regex:'^((?!nccl).)*$' --target-processes all --csv --replay-mode application --page details --log-file l12_multi_rtx.csv --nvtx --profile-from-start no --metrics lts__t_sectors_srcunit_tex_op_read.sum python3 multigpu.py