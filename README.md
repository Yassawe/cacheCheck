To profile l2 cache hit rate with ncu, do the following:

ncu --devices *[list of devices]* --target-processes all --csv -o *[output file name]* --nvtx --nvtx-include *[list of NVTX ranges]* --metrics lts__t_sector_hit_rate.pct /opt/conda/bin/python3 *[your pytorch program]*

Inside the python script, you can use 
torch.cuda.nvtx.range_push and torch.cuda.nvtx.range_pop