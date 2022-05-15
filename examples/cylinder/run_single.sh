export PYTHONPATH=/work/somecode/Science/final/PaddleScience:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=5 
# rm -rf single_workerlog.baseline
# python3 cylinder3d_unsteady_ad.py | tee -a single_workerlog.baseline

rm -rf single_workerlog.new
python3 cylinder3d_unsteady_ad.py | tee -a single_workerlog.new
# python3  cylinder3d_unsteady_ad_backup.py | tee -a single_workerlog.new