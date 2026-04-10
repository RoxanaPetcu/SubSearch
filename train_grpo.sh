# 2. Automatically get the Retriever IP
IP_FILE="$RETRIEVER_IP_PATH"

if [ -f "$IP_FILE" ]; then
    RETRIEVER_IP=$(cat "$IP_FILE")
    echo "Connecting to Retriever at: $RETRIEVER_IP"
else
    echo "ERROR: Retriever IP file not found at $IP_FILE"
    exit 1
fi

export HF_HOME="/scratch-shared/$USER//hf_cache"
export TRITON_CACHE_DIR="/scratch-shared/$USER//triton_cache"
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
WAND_PROJECT='SubSearch'

export DATA_PATH="$DATA_PATH"

# export BASE_MODEL='meta-llama/Llama-3.2-3B'
# export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
# export BASE_MODEL='Qwen/Qwen2.5-3B'


# model="llama3.2-3b"
model="Qwen-Qwen2.5-3B" 
prompt_decomposition="true"
r_answerability="false"
alpha=0.8
beta=0.2
lr="1e6"

EXPERIMENT_NAME="nq+hotpotqa-Instruct-grpo-format-${model}"

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

export SAVE_DIR="$SAVE_DIR_BASE/$EXPERIMENT_NAME"

# 4. GRPO Execution FOR BASE:
# actor_rollout_ref.model.path=$BASE_MODEL
# actor_rollout_ref.actor.optim.lr=1e-6 \    
# actor_rollout_ref.actor.kl_loss_coef=0.001 \
# actor_rollout_ref.rollout.temperature=1 \

# FOR INSTRUCT:
# EM only
# lr=5e-7
# kl=0.003
# temperature=0.8

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_PATH/train.parquet \
    data.val_files=$DATA_PATH/val.parquet \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=$max_obs_length \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
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
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.total_training_steps=800 \
    trainer.default_local_dir=$SAVE_DIR \
    trainer.default_hdfs_dir=null \
    +trainer.dirs_exist_ok=false \
    max_turns=4 \
    retriever.url="http://${RETRIEVER_IP}:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee $EXPERIMENT_NAME.log