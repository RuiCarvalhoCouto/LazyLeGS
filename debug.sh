timestamp=$(date +%Y%m%d_%H%M%S)

OAR_JOB_ID=bicycle_big python train.py -s /data2/ningzhh/data/mipnerf360/bicycle -m output/rl/debug/bicycle/ -i images --eval --optimizer_type default --grad_abs_thresh 0.0008 --use_rl_densification --use_prune_score
# OAR_JOB_ID=flowers_big python train.py -s /data2/ningzhh/data/mipnerf360/flowers -m output/rl/debug/flowers/ -i images --eval --optimizer_type default --dense 0.005 --grad_abs_thresh 0.0006 --use_rl_densification --use_prune_score
OAR_JOB_ID=flowers_big python train.py -s /data2/ningzhh/data/mipnerf360/flowers -m output/rl/debug/flowers/ -i images --eval --optimizer_type default --dense 0.005 --grad_abs_thresh 0.001 --use_rl_densification --use_prune_score
OAR_JOB_ID=garden_big python train.py -s /data2/ningzhh/data/mipnerf360/garden -m output/rl/debug/garden/ -i images --eval --optimizer_type default --highfeature_lr 0.02 --loss_thresh 0.06  --grad_abs_thresh 0.0003 --use_rl_densification --use_prune_score
# OAR_JOB_ID=stump_big python train.py -s /data2/ningzhh/data/mipnerf360/stump -m output/rl/debug/stump/ -i images --eval --optimizer_type default --dense 0.004 --grad_abs_thresh 0.0008 --use_rl_densification --use_prune_score
OAR_JOB_ID=stump_big python train.py -s /data2/ningzhh/data/mipnerf360/stump -m output/rl/debug/stump/ -i images --eval --optimizer_type default --dense 0.004 --grad_abs_thresh 0.001 --use_rl_densification --use_prune_score
OAR_JOB_ID=treehill_big python train.py -s /data2/ningzhh/data/mipnerf360/treehill -m output/rl/debug/treehill/ -i images --eval --optimizer_type default --dense 0.01 --grad_abs_thresh 0.0018 --use_rl_densification --use_prune_score
OAR_JOB_ID=room_big python train.py -s /data2/ningzhh/data/mipnerf360/room -m output/rl/debug/room/ -i images --eval --optimizer_type default --highfeature_lr 0.02 --grad_abs_thresh 0.0004 --use_rl_densification --use_prune_score
OAR_JOB_ID=counter_big python train.py -s /data2/ningzhh/data/mipnerf360/counter -m output/rl/debug/counter/ -i images --eval --optimizer_type default --highfeature_lr 0.02 --grad_abs_thresh 0.0004 --use_rl_densification --use_prune_score
OAR_JOB_ID=kitchen_big python train.py -s /data2/ningzhh/data/mipnerf360/kitchen -m output/rl/debug/kitchen/ -i images --eval --optimizer_type default --highfeature_lr 0.02 --grad_abs_thresh 0.0002 --use_rl_densification -use_prune_score
OAR_JOB_ID=bonsai_big python train.py -s /data2/ningzhh/data/mipnerf360/bonsai -m output/rl/debug/bonsai/ -i images --eval --optimizer_type default --highfeature_lr 0.02 --grad_abs_thresh 0.0002 --use_rl_densification -use_prune_score
OAR_JOB_ID=truck_big python train.py -s /data2/ningzhh/data/tanksandtemples/truck -m output/rl/debug/truck/ --eval --optimizer_type default --highfeature_lr 0.04 --grad_abs_thresh 0.0004 --mult 0.7 --use_rl_densification -use_prune_score
OAR_JOB_ID=train_big python train.py -s /data2/ningzhh/data/tanksandtemples/train -m output/rl/debug/train/ --eval --optimizer_type default --highfeature_lr 0.042 --grad_abs_thresh 0.0004 --dense 0.015 --mult 0.7 --use_rl_densification -use_prune_score  
OAR_JOB_ID=playroom_big python train.py -s /data2/ningzhh/data/db/playroom -m output/rl/debug/playroom/ --eval --optimizer_type default --highfeature_lr 0.0015 --dense 0.003 --mult 0.7 --grad_abs_thresh 0.0005 --use_rl_densification -use_prune_score
OAR_JOB_ID=drjohnson_big python train.py -s /data2/ningzhh/data/db/drjohnson -m output/rl/debug/drjohnson/ --eval --optimizer_type default --highfeature_lr 0.0025 --lowfeature_lr 0.0005 --grad_abs_thresh 0.0005 --dense 0.005 --mult 0.7 --use_rl_densification -use_prune_score


python render.py -m output/rl/debug/bicycle/ --skip_train
python metrics.py -m output/rl/debug/bicycle/

python render.py -m output/rl/debug/flowers/ --skip_train
python metrics.py -m output/rl/debug/flowers/

python render.py -m output/rl/debug/garden/ --skip_train
python metrics.py -m output/rl/debug/garden/

python render.py -m output/rl/debug/stump/ --skip_train
python metrics.py -m output/rl/debug/stump/

python render.py -m output/rl/debug/treehill/ --skip_train
python metrics.py -m output/rl/debug/treehill/

python render.py -m output/rl/debug/room/ --skip_train
python metrics.py -m output/rl/debug/room/

python render.py -m output/rl/debug/counter/ --skip_train
python metrics.py -m output/rl/debug/counter/

python render.py -m output/rl/debug/kitchen/ --skip_train
python metrics.py -m output/rl/debug/kitchen/

python render.py -m output/rl/debug/bonsai/ --skip_train
python metrics.py -m output/rl/debug/bonsai/

python render.py -m output/rl/debug/truck/ --skip_train --mult 0.7
python metrics.py -m output/rl/debug/truck/

python render.py -m output/rl/debug/train/ --skip_train --mult 0.7
python metrics.py -m output/rl/debug/train/

python render.py -m output/rl/debug/playroom/ --skip_train --mult 0.7
python metrics.py -m output/rl/debug/playroom/

python render.py -m output/rl/debug/drjohnson/ --skip_train --mult 0.7
python metrics.py -m output/rl/debug/drjohnson/
