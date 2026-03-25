echo "mipnerf开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
start=$(date +%s)
OAR_JOB_ID=bicycle python train.py -s /data2/ningzhh/data/mipnerf360/bicycle -m output/rl4/bicycle -i images_4 --eval --densification_interval 100  --optimizer_type default  --grad_abs_thresh 0.0004 --rl_use_my_value --use_rl_densification
OAR_JOB_ID=flowers python train.py -s /data2/ningzhh/data/mipnerf360/flowers -m output/rl4/flowers -i images_4 --eval --densification_interval 100  --optimizer_type default --dense 0.005 --grad_abs_thresh 0.0004 --rl_use_my_value --use_rl_densification
OAR_JOB_ID=garden python train.py -s /data2/ningzhh/data/mipnerf360/garden -m output/rl4/garden -i images_4 --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.02 --loss_thresh 0.06  --grad_abs_thresh 0.0002 --rl_use_my_value --use_rl_densification
OAR_JOB_ID=stump python train.py -s /data2/ningzhh/data/mipnerf360/stump -m output/rl4/stump -i images_4 --eval --densification_interval 100  --optimizer_type default  --dense 0.004 --grad_abs_thresh 0.0004 --rl_use_my_value --use_rl_densification
OAR_JOB_ID=treehill python train.py -s /data2/ningzhh/data/mipnerf360/treehill -m output/rl4/treehill -i images_4 --eval --densification_interval 100  --optimizer_type default  --dense 0.01 --grad_abs_thresh 0.001 --rl_use_my_value --use_rl_densification
OAR_JOB_ID=room python train.py -s /data2/ningzhh/data/mipnerf360/room -m output/rl4/room -i images_2 --eval --densification_interval 100  --optimizer_type default --highfeature_lr 0.02 --grad_abs_thresh 0.0002 --rl_use_my_value --use_rl_densification
OAR_JOB_ID=counter python train.py -s /data2/ningzhh/data/mipnerf360/counter -m output/rl4/counter -i images_2 --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.02 --grad_abs_thresh 0.0002 --rl_use_my_value --use_rl_densification
OAR_JOB_ID=kitchen python train.py -s /data2/ningzhh/data/mipnerf360/kitchen -m output/rl4/kitchen -i images_2 --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.02 --grad_abs_thresh 0.0001 --rl_use_my_value --use_rl_densification
OAR_JOB_ID=bonsai python train.py -s /data2/ningzhh/data/mipnerf360/bonsai -m output/rl4/bonsai -i images_2 --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.02 --grad_abs_thresh 0.0001 --rl_use_my_value --use_rl_densification
end=$(date +%s)
echo "mipnerf结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "mipnerf耗时: $(( (end - start) / 60 )) m"

# echo "tanksandtemples开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
# start=$(date +%s)
# OAR_JOB_ID=truck python train.py -s /data2/ningzhh/data/tanksandtemples/truck -m output/rl4/truck --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.04 --grad_abs_thresh 0.0002 --mult 0.7 --use_rl_densification --rl_use_my_value
# OAR_JOB_ID=train python train.py -s /data2/ningzhh/data/tanksandtemples/train -m output/rl4/train --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.042 --grad_abs_thresh 0.0002 --dense 0.015 --mult 0.7 --use_rl_densification --rl_use_my_value
# end=$(date +%s)
# echo "tanksandtemples结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
# echo "tanksandtemples耗时: $(( (end - start) / 60 )) m"

# echo "db开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
# start=$(date +%s)
# OAR_JOB_ID=drjohnson python train.py -s /data2/ningzhh/data/db/drjohnson -m output/rl4/drjohnson --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.0025 --lowfeature_lr 0.0005 --grad_abs_thresh 0.0003 --dense 0.005 --mult 0.7 --use_rl_densification --rl_use_my_value
# OAR_JOB_ID=playroom python train.py -s /data2/ningzhh/data/db/playroom -m output/rl4/playroom --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.0015 --dense 0.003 --mult 0.7 --grad_abs_thresh 0.0003 --use_rl_densification --rl_use_my_value
# end=$(date +%s)
# echo "db结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
# echo "db耗时: $(( (end - start) / 60 )) m"


python render.py -m output/rl4/bicycle --skip_train
python metrics.py -m output/rl4/bicycle

python render.py -m output/rl4/flowers --skip_train
python metrics.py -m output/rl4/flowers

python render.py -m output/rl4/garden --skip_train
python metrics.py -m output/rl4/garden

python render.py -m output/rl4/stump --skip_train
python metrics.py -m output/rl4/stump

python render.py -m output/rl4/treehill --skip_train
python metrics.py -m output/rl4/treehill

python render.py -m output/rl4/room --skip_train
python metrics.py -m output/rl4/room

python render.py -m output/rl4/counter --skip_train
python metrics.py -m output/rl4/counter

python render.py -m output/rl4/kitchen --skip_train
python metrics.py -m output/rl4/kitchen

python render.py -m output/rl4/bonsai --skip_train
python metrics.py -m output/rl4/bonsai

# python render.py -m output/rl4/truck --skip_train --mult 0.7
# python metrics.py -m output/rl4/truck

# python render.py -m output/rl4/train --skip_train --mult 0.7
# python metrics.py -m output/rl4/train

# python render.py -m output/rl4/drjohnson --skip_train --mult 0.7
# python metrics.py -m output/rl4/drjohnson

# python render.py -m output/rl4/playroom --skip_train --mult 0.7
# python metrics.py -m output/rl4/playroom

# python tmp.py
