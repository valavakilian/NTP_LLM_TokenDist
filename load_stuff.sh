cd  /scratch/st-cthrampo-1/vaalaa/NTP_LLM_TokenDist_WikiBig_reloadTest/scripts
salloc  --account=st-cthrampo-1 --time=1:0:0 -N 1 -n 2  --mem=32G
module load gcc/9.4.0 python/3.12.0 py-virtualenv/16.7.6 http_proxy
source /arc/project/st-cthrampo-1/vala/rocky_env_312/bin/activate
