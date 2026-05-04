#!/bin/bash -l

#SBATCH -p gpu
#SBATCH -t 12:00:00
#SBATCH -C a100
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=5
#SBATCH --cpus-per-task=4
#SBATCH -J llama_euler_8h

module load python
module load cuda
module load cudnn
module load nccl

source $VENVDIR/disco/bin/activate

python3 ZEBRA/pretrain_llama.py \
  --config-name=euler_ns.yaml \
  data.tokenizer_path=/mnt/home/lserrano/zebra/outputs/tokenizer_euler-ns_emb1024_dim64/last.ckpt \
  data.dataset_name=euler-ns \
  data.file_dir=/mnt/home/lserrano/ceph/data/euler_ns_short \
  data.num_gpus=8 \
  data.vorticity_scale=10.0 \
  model.max_length=16384 \
  model.max_position_embeddings=16384 \
  model.num_dimensions=2 \
  training.tokenize_on_the_fly=False \
  data.n_input_frames=16 \
  data.n_output_frames=8 \
  data.slice_size=24 \
  data.num_context_trajectories=0 \
  data.trajectories_per_environment=512 \
  training.batch_size=8 \
  training.max_steps=100000
