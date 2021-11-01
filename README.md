To profile l2 cache hit rate with ncu, do the following (just example):

ncu --devices 0 --target-processes all --csv -o oneGPUStats --nvtx --profile-from-start no --metrics lts__t_sector_hit_rate.pct /opt/conda/bin/python3 onegpu.py

Inside the python script, you can use 
torch.cuda.nvtx.range_push and torch.cuda.nvtx.range_pop