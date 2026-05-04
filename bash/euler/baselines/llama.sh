#!/bin/bash -l

#SBATCH -p gpu
#SBATCH -t 24:00:00
#SBATCH -C a100
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=5
#SBATCH --cpus-per-task=4

module load python
module load cuda
module load cudnn
module load nccl

source $VENVDIR/disco/bin/activate

python3 ZEBRA/pretrain_llama.py \
  --config-name=euler_ns.yaml \
  data.tokenizer_path=/mnt/home/lserrano/ceph/zebra/tokenizer/rd/rural-sky-68/last.ckpt \
  data.dataset_name=euler-ns \
  data.file_dir=/mnt/home/lserrano/ceph/data/euler_ns_short \
  data.num_gpus=4 \
  data.vorticity_scale=20.0 \
  model.max_length=16384 \
  model.num_dimensions=2 \
  training.tokenize_on_the_fly=True \
  data.n_input_frames=16 \
  data.n_output_frames=16 \
  data.slice_size=32 \
  data.num_context_trajectories=1 \
  data.trajectories_per_environment=512 \
  training.batch_size=32
