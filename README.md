# SubSearch: Intermediate Rewards for Unsupervised Guided Reasoning in Complex Retrieval

<p align="center">
  <a href="https://arxiv.org/abs/2604.07415">
    <img src="https://img.shields.io/badge/Paper-ArXiv-red?style=for-the-badge&logo=arxiv" alt="Paper">
  </a>
  &nbsp;&nbsp;
  <a href="https://huggingface.co/RoxanaMaria/SubSearch-qwen2.5-3b-grpo">
    <img src="https://img.shields.io/badge/Resources-GitHub-black?style=for-the-badge&logo=github" alt="Resources">
  </a>
</p>

**SubSearch** is a reinforcement learning (RL) framework designed for training **hierarchical reason-and-search LLMs**. While standard models often struggle with complex, multi-hop information needs, SubSearch enables language models to autonomously decompose a query into a verifiable tree of sub-queries.

Built upon veRL and extending the foundational ideas of Search-R1 and DeepSeek-R1, SubSearch introduces a dual-level reward system. It moves beyond simple outcome-based rewards by integrating intrinsic process sensors: Splittability (rewarding logical decomposition at the query level) and Answerability (rewarding document sufficiency at the sub-query level). This creates a robust, open-source RL training pipeline for building agents capable of deep, multi-step research.

We support various RL methods (e.g., PPO, GRPO), diverse base models (e.g., Llama 3.2, Qwen 2.5), and flexible retrieval backends, ranging from local vector stores to live web search.

## Installation

### Search-r1 environment
```bash
conda create -n subsearch_rl python=3.9
conda activate subsearch_rl

pip install torch==2.4.0 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install vllm==0.6.3 flash-attn --no-build-isolation

pip install verl outlines sentence-transformers wandb
pip install -e .
```

### Retriever environment (optional)
If you would like to call a local retriever as the search engine, you can install the environment as follows. (We recommend using a seperate environment.)
```bash
conda create -n retriever python=3.10
conda activate retriever

conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c 
pip install transformers datasets pyserini

conda install -c pytorch -c nvidia faiss-gpu=1.8.0
pip install sentence-transformers openai pydantic-settings tiktoken

pip install uvicorn fastapi
```


## Quick start

Train a reasoning + search LLM on NQ dataset with e5 as the retriever and wikipedia as the corpus.

(1) Download the indexing and corpus.
```bash
save_path=/path/to/save/data
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

(2) Process the NQ dataset. Change template_type for prompt decomposition. 
```bash
python scripts/data_process/nq_search.py
```

(3) Launch a local retrieval server.
```bash
export RETRIEVER_IP_PATH=/path/to/save/IP/file
export DATA_PATH=/path/to/data
conda activate retriever
bash run_retriever.sh
```

(4) Run RL training (PPO) with Llama-3.2-3b-base.
```bash
export RETRIEVER_IP_PATH=/path/to/save/IP/file
export DATA_PATH=/path/to/data
export SAVE_DIR_BASE=/path//to/checkpoints
conda activate subsearch_rl
bash train_grpo.sh
```
