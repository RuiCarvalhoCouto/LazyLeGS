echo "mipnerf开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
start=$(date +%s)
OAR_JOB_ID=bicycle_big python train.py -s /data2/ningzhh/data/mipnerf360/bicycle -m output/fastgs/bicycle_big -i images_4 --eval --densification_interval 100  --optimizer_type default --test_iterations 30000  --grad_abs_thresh 0.0008
OAR_JOB_ID=flowers_big python train.py -s /data2/ningzhh/data/mipnerf360/flowers -m output/fastgs/flowers_big -i images_4 --eval --densification_interval 100  --optimizer_type default --test_iterations 30000 --dense 0.005 --grad_abs_thresh 0.001
OAR_JOB_ID=garden_big python train.py -s /data2/ningzhh/data/mipnerf360/garden -m output/fastgs/garden_big -i images_4 --eval --densification_interval 100  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.02 --loss_thresh 0.06  --grad_abs_thresh 0.0003
OAR_JOB_ID=stump_big python train.py -s /data2/ningzhh/data/mipnerf360/stump -m output/fastgs/stump_big -i images_4 --eval --densification_interval 100  --optimizer_type default --test_iterations 30000  --dense 0.004 --grad_abs_thresh 0.001
OAR_JOB_ID=treehill_big python train.py -s /data2/ningzhh/data/mipnerf360/treehill -m output/fastgs/treehill_big -i images_4 --eval --densification_interval 100  --optimizer_type default --test_iterations 30000  --dense 0.01 --grad_abs_thresh 0.0018
OAR_JOB_ID=room_big python train.py -s /data2/ningzhh/data/mipnerf360/room -m output/fastgs/room_big -i images_2 --eval --densification_interval 100  --optimizer_type default --test_iterations 30000 --highfeature_lr 0.02 --grad_abs_thresh 0.0004 
OAR_JOB_ID=counter_big python train.py -s /data2/ningzhh/data/mipnerf360/counter -m output/fastgs/counter_big -i images_2 --eval --densification_interval 100  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.02 --grad_abs_thresh 0.0004
OAR_JOB_ID=kitchen_big python train.py -s /data2/ningzhh/data/mipnerf360/kitchen -m output/fastgs/kitchen_big -i images_2 --eval --densification_interval 100  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.02 --grad_abs_thresh 0.0002
OAR_JOB_ID=bonsai_big python train.py -s /data2/ningzhh/data/mipnerf360/bonsai -m output/fastgs/bonsai_big -i images_2 --eval --densification_interval 100  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.02 --grad_abs_thresh 0.0002
end=$(date +%s)
echo "mipnerf结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "mipnerf耗时: $((end - start) / 60)) m"

echo "tanksandtemples开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
start=$(date +%s)
OAR_JOB_ID=truck_big python train.py -s /data2/ningzhh/data/tanksandtemples/truck -m output/fastgs/truck_big --eval --densification_interval 100  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.04 --grad_abs_thresh 0.0004 --mult 0.7
OAR_JOB_ID=train_big python train.py -s /data2/ningzhh/data/tanksandtemples/train -m output/fastgs/train_big --eval --densification_interval 100  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.042 --grad_abs_thresh 0.0004 --dense 0.015 --mult 0.7
end=$(date +%s)
echo "tanksandtemples结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "tanksandtemples耗时: $((end - start) / 60)) m"

echo "db开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
start=$(date +%s)
OAR_JOB_ID=drjohnson_big python train.py -s /data2/ningzhh/data/db/drjohnson -m output/fastgs/drjohnson_big --eval --densification_interval 100  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.0025 --lowfeature_lr 0.0005 --grad_abs_thresh 0.0005 --dense 0.005 --mult 0.7
OAR_JOB_ID=playroom_big python train.py -s /data2/ningzhh/data/db/playroom -m output/fastgs/playroom_big --eval --densification_interval 100  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.0015 --dense 0.003 --mult 0.7 --grad_abs_thresh 0.0005
end=$(date +%s)
echo "db结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "db耗时: $((end - start) / 60)) m"


python render.py -m output/fastgs/bicycle_big --skip_train
python metrics.py -m output/fastgs/bicycle_big

python render.py -m output/fastgs/flowers_big --skip_train
python metrics.py -m output/fastgs/flowers_big

python render.py -m output/fastgs/garden_big --skip_train
python metrics.py -m output/fastgs/garden_big

python render.py -m output/fastgs/stump_big --skip_train
python metrics.py -m output/fastgs/stump_big

python render.py -m output/fastgs/treehill_big --skip_train
python metrics.py -m output/fastgs/treehill_big

python render.py -m output/fastgs/room_big --skip_train
python metrics.py -m output/fastgs/room_big

python render.py -m output/fastgs/counter_big --skip_train
python metrics.py -m output/fastgs/counter_big

python render.py -m output/fastgs/kitchen_big --skip_train
python metrics.py -m output/fastgs/kitchen_big

python render.py -m output/fastgs/bonsai_big --skip_train
python metrics.py -m output/fastgs/bonsai_big

python render.py -m output/fastgs/truck_big --skip_train --mult 0.7
python metrics.py -m output/fastgs/truck_big

python render.py -m output/fastgs/train_big --skip_train --mult 0.7
python metrics.py -m output/fastgs/train_big

python render.py -m output/fastgs/drjohnson_big --skip_train --mult 0.7
python metrics.py -m output/fastgs/drjohnson_big

python render.py -m output/fastgs/playroom_big --skip_train --mult 0.7
python metrics.py -m output/fastgs/playroom_big
