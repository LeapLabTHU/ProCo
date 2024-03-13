PORT=$(($RANDOM % 1000 + 10000))

arch=resnet50
dataset="inat"
wd=1e-4
lr=0.2
batch_size=32
N_GPU=8
epochs=90
use_norm=True
data="/home/data/iNaturalist18" # replace with your own path
root_log=saved

mark=${arch}_dataset${dataset}_wd${wd}_lr${lr}_batch_size${batch_size}_N_GPU${N_GPU}_epochs${epochs}_use_norm${use_norm}
output_dir=./${root_log}/${mark}

echo ${mark}
mkdir -p ${output_dir}
cp -r ./ProCo ${output_dir}
cp ./sh/ProCo_inat_R50_90epochs.sh ${output_dir}

torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:${PORT} \
	--nnodes=1 --nproc_per_node="${N_GPU}" \
	ProCo/main.py \
	--data ${data} \
	--mark "${mark}" \
	--arch ${arch} \
	-b ${batch_size} \
	--use_norm ${use_norm} \
	--epochs ${epochs} \
	--wd ${wd} \
	--lr ${lr} \
	--dataset ${dataset} \
	--root_log ${root_log}
