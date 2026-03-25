# echo "mipnerf开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
# start=$(date +%s)
OAR_JOB_ID=bicycle python train.py -s /data2/ningzhh/data/mipnerf360/bicycle -m output/tmp/bicycle -i images_4 --eval --densification_interval 100  --optimizer_type default --grad_thresh 0.0002  --grad_abs_thresh 0.001 --use_rl_densification --rl_use_my_value --use_prune_estimator --visualize_policy
OAR_JOB_ID=flowers python train.py -s /data2/ningzhh/data/mipnerf360/flowers -m output/tmp/flowers -i images_4 --eval --densification_interval 100  --optimizer_type default --grad_abs_thresh 0.0004 --use_rl_densification --rl_use_my_value --use_prune_estimator --visualize_policy
OAR_JOB_ID=garden python train.py -s /data2/ningzhh/data/mipnerf360/garden -m output/tmp/garden -i images_4 --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.02 --loss_thresh 0.06  --grad_abs_thresh 0.0004 --use_rl_densification --rl_use_my_value --use_prune_estimator --visualize_policy
OAR_JOB_ID=stump python train.py -s /data2/ningzhh/data/mipnerf360/stump -m output/tmp/stump -i images_4 --eval --densification_interval 100  --optimizer_type default --grad_abs_thresh 0.0004 --use_rl_densification --rl_use_my_value --use_prune_estimator --visualize_policy
OAR_JOB_ID=treehill python train.py -s /data2/ningzhh/data/mipnerf360/treehill -m output/tmp/treehill -i images_4 --eval --densification_interval 100  --optimizer_type default --grad_thresh 0.0002 --grad_abs_thresh 0.001 --use_rl_densification --rl_use_my_value --use_prune_estimator --visualize_policy
OAR_JOB_ID=room python train.py -s /data2/ningzhh/data/mipnerf360/room -m output/tmp/room -i images_2 --eval --densification_interval 100  --optimizer_type default --highfeature_lr 0.02 --grad_abs_thresh 0.0002 --use_rl_densification --rl_use_my_value --use_prune_estimator --visualize_policy
OAR_JOB_ID=counter python train.py -s /data2/ningzhh/data/mipnerf360/counter -m output/tmp/counter -i images_2 --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.02 --grad_abs_thresh 0.0002 --use_rl_densification --rl_use_my_value --use_prune_estimator --visualize_policy
OAR_JOB_ID=kitchen python train.py -s /data2/ningzhh/data/mipnerf360/kitchen -m output/tmp/kitchen -i images_2 --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.02 --grad_abs_thresh 0.0003 --use_rl_densification --rl_use_my_value --use_prune_estimator --visualize_policy
OAR_JOB_ID=bonsai python train.py -s /data2/ningzhh/data/mipnerf360/bonsai -m output/tmp/bonsai -i images_2 --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.02 --grad_abs_thresh 0.0001 --use_rl_densification --rl_use_my_value --use_prune_estimator --visualize_policy
# end=$(date +%s)
# echo "mipnerf结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
# echo "mipnerf耗时: $(( (end - start) / 60 )) m"

# echo "tanksandtemples开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
# start=$(date +%s)
OAR_JOB_ID=truck python train.py -s /data2/ningzhh/data/tanksandtemples/truck -m output/tmp/truck --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.04 --grad_abs_thresh 0.0001 --use_rl_densification --rl_use_my_value --use_prune_estimator --visualize_policy
OAR_JOB_ID=train python train.py -s /data2/ningzhh/data/tanksandtemples/train -m output/tmp/train --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.042 --grad_abs_thresh 0.0001 --use_rl_densification --rl_use_my_value --use_prune_estimator --visualize_policy
# end=$(date +%s)
# echo "tanksandtemples结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
# echo "tanksandtemples耗时: $(( (end - start) / 60 )) m"

# echo "db开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
# start=$(date +%s)
OAR_JOB_ID=drjohnson python train.py -s /data2/ningzhh/data/db/drjohnson -m output/tmp/drjohnson --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.0025 --lowfeature_lr 0.0005 --grad_abs_thresh 0.00025 --use_rl_densification --rl_use_my_value --use_prune_estimator --visualize_policy
OAR_JOB_ID=playroom python train.py -s /data2/ningzhh/data/db/playroom -m output/tmp/playroom --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.0015 --grad_abs_thresh 0.00025 --use_rl_densification --rl_use_my_value --use_prune_estimator --visualize_policy
# end=$(date +%s)
# echo "db结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
# echo "db耗时: $(( (end - start) / 60 )) m"


# python render.py -m output/tmp/bicycle --skip_train
# python metrics.py -m output/tmp/bicycle

# python render.py -m output/tmp/flowers --skip_train
# python metrics.py -m output/tmp/flowers

# python render.py -m output/tmp/garden --skip_train
# python metrics.py -m output/tmp/garden

# python render.py -m output/tmp/stump --skip_train
# python metrics.py -m output/tmp/stump

# python render.py -m output/tmp/treehill --skip_train
# python metrics.py -m output/tmp/treehill

# python render.py -m output/tmp/room --skip_train
# python metrics.py -m output/tmp/room

# python render.py -m output/tmp/counter --skip_train
# python metrics.py -m output/tmp/counter

# python render.py -m output/tmp/kitchen --skip_train
# python metrics.py -m output/tmp/kitchen

# python render.py -m output/tmp/bonsai --skip_train
# python metrics.py -m output/tmp/bonsai

# python render.py -m output/tmp/truck --skip_train
# python metrics.py -m output/tmp/truck

# python render.py -m output/tmp/train --skip_train
# python metrics.py -m output/tmp/train

# python render.py -m output/tmp/drjohnson --skip_train
# python metrics.py -m output/tmp/drjohnson

# python render.py -m output/tmp/playroom --skip_train
# python metrics.py -m output/tmp/playroom

# python tmp.py
