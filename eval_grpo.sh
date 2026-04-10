#!/bin/bash

# 1. Environment Setup (Matching your PPO logic)
module purge
module load 2025
module load Anaconda3/2025.06-1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate searchr1

# 2. Automatically get the Retriever IP
IP_FILE="/scratch-shared/rpetcu/current_retriever_ip_searchr1.txt"

if [ -f "$IP_FILE" ]; then
    RETRIEVER_IP=$(cat "$IP_FILE")
    echo "Connecting to Retriever at: $RETRIEVER_IP"
else
    echo "ERROR: Retriever IP file not found at $IP_FILE"
    exit 1
fi

export HF_HOME="/scratch-shared/rpetcu/hf_cache"
export TRITON_CACHE_DIR="/scratch-shared/rpetcu/triton_cache"
export PYTHONUNBUFFERED=1

# Ensure the directories exist on scratch
mkdir -p $HF_HOME
mkdir -p $TRITON_CACHE_DIR

# 3. NCCL and GPU configuration
export NCCL_SOCKET_IFNAME=ib-bond0
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_P2P_DISABLE=1
export NCCL_TIMEOUT=1800
export NCCL_DEBUG=WARN

export CUDA_VISIBLE_DEVICES=0,1,2,3
WAND_PROJECT='Search-R1'

### MODIFY HERE
# model="llama3.2-3b"
model="Qwen-Qwen2.5-3B" 
prompt_decomposition="true"
r_answerability="true"
alpha=0.8
beta=0.2
lr="1e6"
train_dataset="nq+hotpotqa"
eval_dataset="bamboogle"
### MODIFY HERE


# Default eval batch sizes
train_batch_size=512
val_batch_size=256

# Bamboogle is small; reduce batch sizes so dataloaders are not empty.
if [ "$eval_dataset" == "bamboogle" ]; then
  train_batch_size=32
  val_batch_size=32
fi

echo "Using batch sizes: train=${train_batch_size}, val=${val_batch_size}"


# export DATA_DIR='/home/rpetcu/projects/Search-R1/data/nq_search'
# export DATA_DIR='/home/rpetcu/projects/Search-R1/data/nq_search_decomposed'
# export DATA_DIR='/home/rpetcu/projects/Search-R1/data/nq_search_decomposed_val'
export DATA_DIR="/home/rpetcu/projects/Search-R1/data/${eval_dataset}_search_rsearch"

# export BASE_MODEL='meta-llama/Llama-3.2-3B'
export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'



# EXPERIMENT_NAME="nq-search-r1-grpo-query-split-reward-Bonus-format-${model}"
EXPERIMENT_NAME="${train_dataset}-grpo-format-${model}"

# Standard string comparison
if [ "$prompt_decomposition" == "true" ]; then
  EXPERIMENT_NAME="${EXPERIMENT_NAME}-prompt-decomp"
fi

if [ "$r_answerability" == "true" ]; then
  EXPERIMENT_NAME="${EXPERIMENT_NAME}-r_answerability-alpha${alpha}-beta${beta}"
fi
if [ "$prompt_decomposition" == "true" ]; then
  max_obs_length=1200
else
  max_obs_length=500
fi

EXPERIMENT_NAME="${EXPERIMENT_NAME}-lr${lr}"

export EXPERIMENT_NAME
echo "Saving to WandB as: $EXPERIMENT_NAME"



export VLLM_ATTENTION_BACKEND=XFORMERS

export SAVE_DIR="/scratch-shared/rpetcu/verl_checkpoints/$EXPERIMENT_NAME"

# 4. Test on Val (because Search-R1 validates on test)
# actor_rollout_ref.model.path=$BASE_MODEL
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
  data.train_batch_size=$train_batch_size \
  data.val_batch_size=$val_batch_size \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path="/scratch-shared/rpetcu/verl_checkpoints/R-Search-3b-grpo-nq-hotpotqa/actor/global_step_195" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['wandb'] \
    +trainer.val_only=true \
    +trainer.val_before_train=true \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=50 \
    trainer.default_local_dir=$SAVE_DIR \
    trainer.default_hdfs_dir=null \
    +trainer.dirs_exist_ok=true \
    max_turns=4 \
    retriever.url="http://${RETRIEVER_IP}:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee $EXPERIMENT_NAME.log
