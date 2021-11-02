To profile l2 cache hit rate with ncu, do the following (just example):


one gpu:
ncu --devices 0 --target-processes all --csv --replay-mode application --export ResNet50_1GPUStats --page details --log-file ResNet50_single.csv --nvtx --profile-from-start no --metrics lts__t_sector_hit_rate.pct /opt/conda/bin/python3 onegpu.py


4gpus:
ncu --devices 0 --target-processes all --csv --replay-mode application --export ResNet50_4GPUStats --page details --log-file ResNet50_multiple.csv --nvtx --profile-from-start no --metrics lts__t_sector_hit_rate.pct /opt/conda/bin/python3 4gpu.py
