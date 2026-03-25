echo "mipnerf开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
start=$(date +%s)
OAR_JOB_ID=bicycle python train.py -s /data2/ningzhh/data/mipnerf360/bicycle -m output/rl2/bicycle -i images_4 --eval --densification_interval 100  --optimizer_type default  --grad_abs_thresh 0.0004 --use_rl_densification --rl_use_my_value --use_prune_estimator
OAR_JOB_ID=flowers python train.py -s /data2/ningzhh/data/mipnerf360/flowers -m output/rl2/flowers -i images_4 --eval --densification_interval 100  --optimizer_type default --grad_abs_thresh 0.0004 --use_rl_densification --rl_use_my_value --use_prune_estimator
OAR_JOB_ID=garden python train.py -s /data2/ningzhh/data/mipnerf360/garden -m output/rl2/garden -i images_4 --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.02 --loss_thresh 0.06  --grad_abs_thresh 0.0002 --use_rl_densification --rl_use_my_value --use_prune_estimator
OAR_JOB_ID=stump python train.py -s /data2/ningzhh/data/mipnerf360/stump -m output/rl2/stump -i images_4 --eval --densification_interval 100  --optimizer_type default --grad_abs_thresh 0.0004 --use_rl_densification --rl_use_my_value --use_prune_estimator
OAR_JOB_ID=treehill python train.py -s /data2/ningzhh/data/mipnerf360/treehill -m output/rl2/treehill -i images_4 --eval --densification_interval 100  --optimizer_type default --grad_abs_thresh 0.0005 --use_rl_densification --rl_use_my_value --use_prune_estimator
OAR_JOB_ID=room python train.py -s /data2/ningzhh/data/mipnerf360/room -m output/rl2/room -i images_2 --eval --densification_interval 100  --optimizer_type default --highfeature_lr 0.02 --grad_abs_thresh 0.0002 --use_rl_densification --rl_use_my_value --use_prune_estimator
OAR_JOB_ID=counter python train.py -s /data2/ningzhh/data/mipnerf360/counter -m output/rl2/counter -i images_2 --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.02 --grad_abs_thresh 0.0002 --use_rl_densification --rl_use_my_value --use_prune_estimator
OAR_JOB_ID=kitchen python train.py -s /data2/ningzhh/data/mipnerf360/kitchen -m output/rl2/kitchen -i images_2 --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.02 --grad_abs_thresh 0.0001 --use_rl_densification --rl_use_my_value --use_prune_estimator
OAR_JOB_ID=bonsai python train.py -s /data2/ningzhh/data/mipnerf360/bonsai -m output/rl2/bonsai -i images_2 --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.02 --grad_abs_thresh 0.0001 --use_rl_densification --rl_use_my_value --use_prune_estimator
end=$(date +%s)
echo "mipnerf结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "mipnerf耗时: $((end - start)) s"

echo "tanksandtemples开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
start=$(date +%s)
OAR_JOB_ID=truck python train.py -s /data2/ningzhh/data/tanksandtemples/truck -m output/rl2/truck --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.04 --grad_abs_thresh 0.0001 --use_rl_densification --rl_use_my_value --use_prune_estimator --mult 0.7
OAR_JOB_ID=train python train.py -s /data2/ningzhh/data/tanksandtemples/train -m output/rl2/train --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.042 --grad_abs_thresh 0.0001 --use_rl_densification --rl_use_my_value --use_prune_estimator --mult 0.7
end=$(date +%s)
echo "tanksandtemples结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "tanksandtemples耗时: $((end - start)) s"

echo "db开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
start=$(date +%s)
OAR_JOB_ID=drjohnson python train.py -s /data2/ningzhh/data/db/drjohnson -m output/rl2/drjohnson --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.0025 --lowfeature_lr 0.0005 --grad_abs_thresh 0.00025 --use_rl_densification --rl_use_my_value --use_prune_estimator --mult 0.7
OAR_JOB_ID=playroom python train.py -s /data2/ningzhh/data/db/playroom -m output/rl2/playroom --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.0015 --grad_abs_thresh 0.00025 --use_rl_densification --rl_use_my_value --use_prune_estimator --mult 0.7
end=$(date +%s)
echo "db结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "db耗时: $((end - start)) s"


python render.py -m output/rl2/bicycle --skip_train
python metrics.py -m output/rl2/bicycle

python render.py -m output/rl2/flowers --skip_train
python metrics.py -m output/rl2/flowers

python render.py -m output/rl2/garden --skip_train
python metrics.py -m output/rl2/garden

python render.py -m output/rl2/stump --skip_train
python metrics.py -m output/rl2/stump

python render.py -m output/rl2/treehill --skip_train
python metrics.py -m output/rl2/treehill

python render.py -m output/rl2/room --skip_train
python metrics.py -m output/rl2/room

python render.py -m output/rl2/counter --skip_train
python metrics.py -m output/rl2/counter

python render.py -m output/rl2/kitchen --skip_train
python metrics.py -m output/rl2/kitchen

python render.py -m output/rl2/bonsai --skip_train
python metrics.py -m output/rl2/bonsai

python render.py -m output/rl2/truck --skip_train --mult 0.7
python metrics.py -m output/rl2/truck

python render.py -m output/rl2/train --skip_train --mult 0.7
python metrics.py -m output/rl2/train

python render.py -m output/rl2/drjohnson --skip_train --mult 0.7
python metrics.py -m output/rl2/drjohnson

python render.py -m output/rl2/playroom --skip_train --mult 0.7
python metrics.py -m output/rl2/playroom

python tmp.py
