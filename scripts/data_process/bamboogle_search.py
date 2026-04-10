"""
Preprocess Bamboogle into Search-R1 parquet format.
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


def make_prefix(question: str, template_type: str) -> str:
    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
    elif template_type == 'decomposed':
        prefix =  f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search>, and it will return the top searched results between <information> and </information>.  \
If the original query is complex or involves multiple parts, you are encouraged to decompose it into at most 3 smaller sub-questions, separated by ##. For example: <search> sub-question 1 ## sub-question 2 </search>  and it will return the top searched results between <information> documents sub-question 1 ## documents sub-question 2 </information>.  \
You can search as many times as you want. \
Only decompose when the question has multiple independent parts (e.g., different entities, aspects, or comparisons). Do not decompose questions that do not need it. \
If you find no further external knowledge needed, you can directly provide the answer inside<answer> and </answer> without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
    elif template_type == 'rsearch':
        prefix =  f"""You are a helpful assistant that can solve the given question step by step. For each step, start by explaining your thought process. If additional information is needed, provide a specific query enclosed in <search> and </search>. The system will return the top search results within <observation> and </observation>. You can perform multiple searches as needed.
When you know the final answer, use <original_evidence> and </original_evidence> to provide all potentially relevant original information from the observations. Ensure the information is complete and preserves the original wording without modification. If no searches were conducted or observations were made, omit the evidence section. Finally, provide the final answer within <answer> and </answer> tags. For example, <answer> Beijing </answer>. Question: {question}\n"""
    else:
        raise NotImplementedError
    return prefix



def normalize_answers(example):
	if "golden_answers" in example and example["golden_answers"] is not None:
		answers = example["golden_answers"]
	elif "answers" in example and example["answers"] is not None:
		answers = example["answers"]
	elif "answer" in example and example["answer"] is not None:
		answers = [example["answer"]]
	else:
		answers = []

	if isinstance(answers, str):
		answers = [answers]

	return [str(a).strip() for a in answers if str(a).strip()]


def build_split(dataset_dict, split_name: str):
	if split_name in dataset_dict:
		return dataset_dict[split_name]
	if split_name == "validation":
		for alt in ["val", "dev", "test", "train"]:
			if alt in dataset_dict:
				return dataset_dict[alt]
	if split_name == "test":
		for alt in ["dev", "validation", "val", "train"]:
			if alt in dataset_dict:
				return dataset_dict[alt]
	if split_name == "train":
		for alt in ["validation", "val", "dev", "test"]:
			if alt in dataset_dict:
				return dataset_dict[alt]
	raise ValueError(f"No usable split found for {split_name}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--local_dir", default="./data/bamboogle_search_rsearch")
	parser.add_argument("--hdfs_dir", default=None)
	parser.add_argument("--template_type", type=str, default="rsearch")
	args = parser.parse_args()

	data_source = "bamboogle"
	dataset = datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", data_source)

	train_dataset = build_split(dataset, "train")
	val_dataset = build_split(dataset, "validation")
	test_dataset = build_split(dataset, "test")

	def make_map_fn(split):
		def process_fn(example, idx):
			question = str(example["question"]).strip()
			if question and question[-1] != "?":
				question += "?"

			data = {
				"data_source": data_source,
				"prompt": [{"role": "user", "content": make_prefix(question, args.template_type)}],
				"ability": "fact-reasoning",
				"reward_model": {
					"style": "rule",
					"ground_truth": {"target": normalize_answers(example)},
				},
				"extra_info": {
					"split": split,
					"index": idx,
				},
			}
			return data

		return process_fn

	train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
	val_dataset = val_dataset.map(function=make_map_fn("validation"), with_indices=True)
	test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

	os.makedirs(args.local_dir, exist_ok=True)
	train_dataset.to_parquet(os.path.join(args.local_dir, "train.parquet"))
	val_dataset.to_parquet(os.path.join(args.local_dir, "val.parquet"))
	test_dataset.to_parquet(os.path.join(args.local_dir, "test.parquet"))

	if args.hdfs_dir is not None:
		makedirs(args.hdfs_dir)
		copy(src=args.local_dir, dst=args.hdfs_dir)
