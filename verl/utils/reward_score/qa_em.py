# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
import random
import threading
import math


from sentence_transformers import SentenceTransformer, util
from .qa_em_format import (
    is_valid_sequence,
    extract_solution,
    is_retrieval_correct,
    check_decomposition_usage,
    extract_information_blocks
)
# Thread-local storage for sample ID
_thread_local = threading.local()

def set_sample_id(sample_id):
    """Set the current sample ID for logging."""
    _thread_local.sample_id = sample_id

def get_sample_id():
    """Get the current sample ID, or return 'UNKNOWN' if not set."""
    return getattr(_thread_local, 'sample_id', 'UNKNOWN')
import torch



def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score

def extract_all_search_queries(solution_str):
    """
    Extract all <search>...</search> contents.
    Returns:
        List[List[str]] where each inner list is the subqueries
        from one <search> call.
    """
    # Example: [["Japan finances", "Spain finances"], ["Differences between Japan and Spain debt"]]

    search_pattern = r'<search>(.*?)</search>'
    matches = re.finditer(search_pattern, solution_str, re.DOTALL)

    all_queries = []
    for m in matches:
        raw = m.group(1).strip()
        if not raw:
            continue

        # split on ##
        subqueries = [q.strip() for q in raw.split('##') if q.strip()]
        all_queries.append(subqueries)

    return all_queries

def split_docs_from_block(block: str):
    """
    Split a single information block into individual docs.
    Returns List[str], one per Doc i(Title: ...)
    """
    block = (block or "").strip()
    if not block:
        return []

    # Split while keeping "Doc N(Title: ...)" with content
    parts = re.split(r'(?=\bDoc\s+\d+\(Title:)', block)
    docs = [p.strip() for p in parts if p.strip()]

    # Fallback: if no Doc markers found, treat whole block as one doc
    if len(docs) == 1 and not re.search(r'\bDoc\s+\d+\(Title:', docs[0]):
        return [block]

    return docs


def extract_all_information_blocks(solution_str):
    ## ADDED
    """
    Extract all <information>...</information> contents.
    Returns:
        List[List[str]] where each inner list corresponds to
        documents per subquery (split by ##).
    """
    # Example: [
    # [
    #     "doc1 text for Japan...\ndoc2 text...",
    #     "doc1 text for Spain...\ndoc2 text..."
    # ],
    # [
    #     "doc1 text about differences...\ndoc2 text..."
    # ]
    # ]

    info_pattern = r'<information>(.*?)</information>'
    matches = re.finditer(info_pattern, solution_str, re.DOTALL)

    all_steps = []

    for m in matches:
        raw = m.group(1).strip()
        if not raw:
            continue

        # split by subquery (##)
        subq_blocks = [b.strip() for b in raw.split('##') if b.strip()]

        docs_per_subq = []
        for block in subq_blocks:
            docs = split_docs_from_block(block)
            docs_per_subq.append(docs)

        all_steps.append(docs_per_subq)

    return all_steps


def extract_question_from_solution(solution_str: str) -> str:
    match = re.search(r'Question:\s*(.*?)(?:\n|$)', solution_str)
    return match.group(1).strip() if match else ""

def compute_r_answerability(
    solution_str: str,
    judge_fn,
    doc_char_limit: int = 1200,
    first_k: int = 3,
):
    queries_tree = extract_all_search_queries(solution_str)      
    docs_tree = extract_all_information_blocks(solution_str)     
    
    do_print = random.randint(1, 64) == 1
    
    if not queries_tree or not docs_tree:
        return 0.0

    unique_subqs = set()
    unique_docs = set()
    n_steps = min(len(queries_tree), len(docs_tree))
    
    # --- PASS 1: Consistent Precomputation ---
    for t in range(n_steps):
        subqueries = queries_tree[t]
        docs_per_subq = docs_tree[t]
        m = min(len(subqueries), len(docs_per_subq))
        for j in range(m):
            subq = subqueries[j]
            # CRITICAL FIX: Only precompute what we will actually score
            docs = docs_per_subq[j][:first_k]
            unique_subqs.add(subq)
            for doc in docs:
                unique_docs.add(doc)
            if do_print:
                sample_id = get_sample_id()
                print(f"[{sample_id}] --------------------------------")
                print(f"[{sample_id}] Subquery: {subq}")
                print(f"[{sample_id}] Documents: {docs}")
            

    model = getattr(judge_fn, 'model', None)
    threshold = getattr(judge_fn, 'threshold', 0.7)

    # --- PASS 2: OOM-Safe Embedding Generation ---
    subq_emb_dict = {}
    doc_emb_dict = {}
    
    if unique_subqs and unique_docs:
        with torch.no_grad():
            subq_list = list(unique_subqs)
            doc_list = list(unique_docs)
            
            # CRITICAL FIX: Use small batch_size to prevent VRAM spikes
            subq_embs = model.encode(subq_list, batch_size=32, convert_to_tensor=True, normalize_embeddings=True)
            doc_embs = model.encode(doc_list, batch_size=32, convert_to_tensor=True, normalize_embeddings=True)
            
            subq_emb_dict = {q: subq_embs[i] for i, q in enumerate(subq_list)}
            doc_emb_dict = {d: doc_embs[i] for i, d in enumerate(doc_list)}

    # --- PASS 3: Vectorized Scoring ---
    step_scores = []
    for t in range(n_steps):
        subqueries = queries_tree[t]
        docs_per_subq = docs_tree[t]
        m = min(len(subqueries), len(docs_per_subq))
        if m == 0:
            step_scores.append(0.0)
            continue

        subq_scores = []
        for j in range(m):
            subq = subqueries[j]
            docs = docs_per_subq[j][:first_k]
            if not docs:
                subq_scores.append(0.0)
                continue

            emb_q = subq_emb_dict.get(subq)
            # Collect all precomputed doc embeddings for this subquery
            current_doc_embs = [doc_emb_dict.get(d) for d in docs if d in doc_emb_dict]
            
            if emb_q is None or not current_doc_embs:
                subq_scores.append(0.0)
                continue

            doc_matrix = torch.stack(current_doc_embs) 
            similarities = torch.matmul(doc_matrix, emb_q)
            subq_scores.append(similarities.mean().item())

        if subq_scores:
            step_scores.append(sum(subq_scores) / len(subq_scores))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return sum(step_scores) / len(step_scores) if step_scores else 0.0


def add_formatting(reward_score, solution_str, ground_truth, method='strict', structure_format_score=0, final_format_score=0, retrieval_score=0, format_score=0, score=1., decomposition_bonus=0.):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
        decomposition_bonus: bonus reward when decomposition is used successfully
    """
    is_valid_format, _ = is_valid_sequence(solution_str)
    retrieval_correct = False
    if is_valid_format:
        retrieval_correct = is_retrieval_correct(solution_str, ground_truth['target'])
    
    # Check decomposition usage
    # used_decomposition, decomposition_helped = check_decomposition_usage(solution_str)
    
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        # print(f"Used decomposition: {used_decomposition}, Helped: {decomposition_helped}")
        print(f"Solution string: {solution_str}")
    
    # # Apply decomposition bonus only when it's used AND led to successful retrieval
    # decomposition_bonus = decomposition_bonus if (decomposition_helped and retrieval_correct) else 0
            
    if answer is None:
        if is_valid_format:
            if retrieval_correct:
                return structure_format_score + retrieval_score # + decomposition_bonus # 0.3 + bonus
            else:
                return structure_format_score # 0.2
        else:
            return 0
    else:
        if reward_score > 0:
            if is_valid_format:
                if retrieval_correct:
                    return min(1.0, reward_score + retrieval_score + decomposition_bonus)
                return reward_score
            else:
                return max(0, reward_score - structure_format_score)
        else:
            if is_valid_format:
                if retrieval_correct:
                    return structure_format_score + retrieval_score + decomposition_bonus
                else:
                    return structure_format_score
            else:
                return final_format_score

def compute_score_em(solution_str, ground_truth, training_step=0, method='strict', format_score=0., score=1., add_format=True, structure_format_score=0.2, final_format_score=0.1, retrieval_score=0.1, decomposition_bonus=0.0):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if answer is None:
        em_result = False
        final_score = 0
    else:
        em_result = em_check(answer, ground_truth['target'])
        final_score = score if em_result else format_score

    if add_format:
        final_score = add_formatting(
            final_score, 
            solution_str, 
            ground_truth,
            method=method,
            structure_format_score=structure_format_score,
            final_format_score=final_format_score,
            retrieval_score=retrieval_score,
            decomposition_bonus=decomposition_bonus
        )

    if do_print:
        sample_id = get_sample_id()
        print(f"[{sample_id}] --------------------------------")
        print(f"[{sample_id}] Reasoning Trace: {solution_str}")
        print(f"[{sample_id}] Extracted answer: {answer}")
        print(f"[{sample_id}] Golden answers: {ground_truth['target']}")
        print(f"[{sample_id}] EM Result: {em_result} (score: {final_score})")
    
    return final_score


_JUDGE_FN = None
_SCHEDULER_STATE = {
    "ema": None,
    "prev_ema": None,
}
_SCHEDULER_STATE_LOCK = threading.Lock()

def set_judge_fn(judge_fn):
    """Set the global judge function once (e.g., at startup / worker init)."""
    global _JUDGE_FN
    _JUDGE_FN = judge_fn

def _get_judge_fn():
    """Get the global judge function (initialize lazily if needed)."""
    global _JUDGE_FN
    if _JUDGE_FN is None:
        _JUDGE_FN = init_mpnet_judge(
            # model_name="sentence-transformers/all-mpnet-base-v2",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu", 
            threshold=0.35,
        )
    return _JUDGE_FN

# -------------------------
# Judge initializer (loads model ONCE)
# -------------------------

def init_mpnet_judge(
    # model_name: str = "sentence-transformers/all-mpnet-base-v2",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu",
    threshold: float = 0.35,
):
    """
    Returns judge_fn(subq, doc) -> "YES"/"NO"
    Loads the embedding model once when init_mpnet_judge is called.
    """
    model = SentenceTransformer(model_name, device=device)
    model.eval()

    def judge_fn(subq: str, doc: str) -> str:
        if not subq or not doc:
            return "NO"

        with torch.no_grad():
            emb_q = model.encode(subq, convert_to_tensor=True, normalize_embeddings=True)
            emb_d = model.encode(doc, convert_to_tensor=True, normalize_embeddings=True)
            sim = util.cos_sim(emb_q, emb_d).item()

        return sim
        # return "YES" if sim >= threshold else "NO"
    
    judge_fn.model = model
    judge_fn.threshold = threshold
    return judge_fn

def _score_decomposition(parent_emb, children_embs):
    """
    Score a single decomposition step.

    If one child: rephrase reward — peaks at cosine sim ~0.85, 
                  falls off if too similar or too different.
    If multiple children: 0.5 * coverage + 0.5 * part_score
      - coverage: centroid of children vs parent
      - part_score: each child relevant to parent but distinct from siblings (leave-one-out)
    """
    coeff_identical_to_parent = 0.85
    coeff_coverage, coeff_disimilar = 0.5, 0.5

    if len(children_embs) == 1:
        # Decomposition one:
        sim = torch.dot(parent_emb, children_embs[0]).item()
        rephrase_score = 1.0 - abs(sim - coeff_identical_to_parent) / (1 - coeff_identical_to_parent)
        return max(0.0, rephrase_score)
    else:
        # Coverage: how well does the centroid of children cover the parent
        mean_children = children_embs.mean(dim=0)
        mean_children = mean_children / mean_children.norm()
        coverage = torch.dot(parent_emb, mean_children).item() ## ideally 0.7, 0.2 is bad

        # Decomposition a few: each child should be relevant to parent but distinct from siblings
        # Uses leave-one-out to avoid inflating sibling similarity
        child_scores = []
        for i in range(len(children_embs)):
            child_emb = children_embs[i]
            sim_to_parent = torch.dot(parent_emb, child_emb).item()

            sibling_embs = torch.cat([children_embs[:i], children_embs[i+1:]], dim=0)

            if len(sibling_embs) == 0:
                sim_to_siblings = 0.0
            else:
                sims = torch.matmul(sibling_embs, child_emb)
                sim_to_siblings = sims.mean().item()

            child_scores.append(sim_to_parent * (1 - sim_to_siblings))

        part_score = sum(child_scores) / len(child_scores)
        return coeff_coverage * coverage + coeff_disimilar * part_score


def _schedule_alpha_beta(
    alpha: float,
    beta: float,
    training_step: int,
    alpha_warmup_steps: int,
    schedule_type: str = "linear",
    beta_decay_steps: int = 2000,
    min_beta: float = 0.02,
):
    """
    Schedule alpha/beta weights.

    Modes:
      - linear: legacy behavior over alpha_warmup_steps
      - smart: two-phase schedule
          1) warmup: conservative shift (keeps intermediate rewards alive)
          2) cosine decay: beta smoothly decays to min_beta
    """
    step = max(0, int(training_step))

    if schedule_type == "linear":
        if alpha_warmup_steps <= 0:
            return 1.0, 0.0

        progress = min(step / alpha_warmup_steps, 1.0)
        alpha_t = alpha + (1.0 - alpha) * progress
        beta_t = beta * (1.0 - progress)
        return alpha_t, beta_t

    warmup = max(1, int(alpha_warmup_steps))
    decay = max(1, int(beta_decay_steps))
    min_beta = max(0.0, min(float(min_beta), float(beta)))

    beta_after_warmup = max(min_beta, beta * 0.75)
    alpha_after_warmup = min(1.0, max(alpha, 1.0 - beta_after_warmup))

    if step < warmup:
        progress = (step / warmup) ** 1.5
        alpha_t = alpha + (alpha_after_warmup - alpha) * progress
        beta_t = beta + (beta_after_warmup - beta) * progress
    elif step < warmup + decay:
        progress = (step - warmup) / decay
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        beta_t = min_beta + (beta_after_warmup - min_beta) * cosine
        alpha_t = 1.0 - beta_t
    else:
        beta_t = min_beta
        alpha_t = 1.0 - min_beta

    alpha_t = min(1.0, max(0.0, alpha_t))
    beta_t = min(1.0, max(0.0, beta_t))
    return alpha_t, beta_t


def reset_scheduler_state():
    """Reset adaptive scheduler EMA state (useful between train runs)."""
    with _SCHEDULER_STATE_LOCK:
        _SCHEDULER_STATE["ema"] = None
        _SCHEDULER_STATE["prev_ema"] = None


def _update_answer_ema(value: float, ema_decay: float):
    """Update and return (ema, slope) for answer reward tracking."""
    v = min(1.0, max(0.0, float(value)))
    d = min(0.9999, max(0.0, float(ema_decay)))

    with _SCHEDULER_STATE_LOCK:
        prev = _SCHEDULER_STATE["ema"]
        if prev is None:
            ema = v
            slope = 0.0
        else:
            ema = d * prev + (1.0 - d) * v
            slope = ema - prev
        _SCHEDULER_STATE["prev_ema"] = prev
        _SCHEDULER_STATE["ema"] = ema

    return ema, slope


def _adaptive_beta_gate(
    beta_t: float,
    beta_init: float,
    min_beta: float,
    r_answer: float,
    adaptive_beta: bool,
    ema_decay: float,
    target_em: float,
    ema_slope_tol: float,
    adaptive_rate: float,
    max_beta_step: float,
):
    """
    Performance-aware beta adjustment on top of step scheduler.

    Rules:
      - If EM is low and plateauing: increase beta (more intermediate signal).
      - If EM is high and still improving: decrease beta (focus on final EM).
      - Always respect [min_beta, beta_init] and per-step max change.
    """
    if not adaptive_beta:
        return beta_t

    ema, slope = _update_answer_ema(r_answer, ema_decay)

    beta_lo = min(min_beta, beta_init)
    beta_hi = max(min_beta, beta_init)
    beta_next = beta_t

    gate_mode = "nothing"
    # Plateau at low EM -> encourage exploration via intermediate rewards.
    if ema < target_em and abs(slope) <= ema_slope_tol:
        gap = target_em - ema
        beta_next = beta_t + adaptive_rate * gap * (beta_hi - beta_t)
        gate_mode = "increase"
    # Strong EM with positive trend -> gradually focus on answer EM.
    elif ema > target_em and slope > ema_slope_tol:
        gap = ema - target_em
        beta_next = beta_t - adaptive_rate * gap * (beta_t - beta_lo)
        gate_mode = "decrease"

    # Limit update magnitude per call to avoid oscillations.
    max_step = max(0.0, float(max_beta_step))
    beta_next = max(beta_t - max_step, min(beta_t + max_step, beta_next))
    beta_next = max(beta_lo, min(beta_hi, beta_next))
    return beta_next, ema, slope, gate_mode


def compute_r_decomposition(
    solution_str: str,
    original_query: str,
    judge_fn,
):
    """
    Reward for query decomposition quality at each search step.
    For each <search> block, scores how well the subqueries decompose
    the original query: coverage, distinctness, and rephrase quality.
    Returns a score in [0, 1].
    """
    queries_tree = extract_all_search_queries(solution_str)

    if not queries_tree or not original_query:
        return 0.0

    model = getattr(judge_fn, 'model', None)

    # Single pass: embed everything at once
    unique_strings = {original_query}
    for step in queries_tree:
        for subq in step:
            unique_strings.add(subq)

    with torch.no_grad():
        str_list = list(unique_strings)
        embs = model.encode(str_list, batch_size=32, convert_to_tensor=True, normalize_embeddings=True)
        emb_dict = {s: embs[i] for i, s in enumerate(str_list)}

    parent_emb = emb_dict[original_query]

    do_print = random.randint(1, 64) == 1
    step_scores = []
    for subqueries in queries_tree:
        children_embs_list = [emb_dict[q] for q in subqueries if q in emb_dict]
        if not children_embs_list:
            continue

        children_embs = torch.stack(children_embs_list)
        score = _score_decomposition(parent_emb, children_embs)
        step_scores.append(score)

        if do_print:
            sample_id = get_sample_id()
            print(f"[{sample_id}] Decomposition step: {subqueries} -> score={score:.4f}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return sum(step_scores) / len(step_scores) if step_scores else 0.0


def compute_score_em_plus_answerability(
    solution_str,
    ground_truth,
    alpha: float = 0.8,
    beta: float = 0.2,
    training_step: int = 0,
    alpha_warmup_steps: int = 500,
    scheduler_type: str = "smart",
    beta_decay_steps: int = 5000,
    min_beta: float = 0.03,
    adaptive_beta: bool = True,
    ema_decay: float = 0.98,
    target_em: float = 0.65,
    ema_slope_tol: float = 5e-4,
    adaptive_rate: float = 0.15,
    max_beta_step: float = 0.005,
    method: str = "strict",
    format_score: float = 0.0,
    score: float = 1.0,
    add_format: bool = True,
    structure_format_score: float = 0.2,
    final_format_score: float = 0.1,
    retrieval_score: float = 0.1,
    decomposition_bonus: float = 0.0,
):
    """
    Combined reward:
      r = alpha * r_answer(EM) + beta * r_answerability

    Where:
      r_answer ∈ {0,1} (or format_score)
      r_answerability ∈ [0,1]
    """

    r_answerability = None

    # 1) final answer reward
    r_answer = compute_score_em(
        solution_str=solution_str,
        ground_truth=ground_truth,
        method=method,
        format_score=format_score,
        score=score,
        add_format=False,
    )

    # 2) answerability reward
    judge_fn = _get_judge_fn()
    r_answerability = compute_r_answerability(
        solution_str=solution_str,
        judge_fn=judge_fn,
    )

    # 3) combine
    sample_id = get_sample_id()

    _alpha_t, beta_t = _schedule_alpha_beta(
        alpha,
        beta,
        training_step,
        alpha_warmup_steps,
        schedule_type=scheduler_type,
        beta_decay_steps=beta_decay_steps,
        min_beta=min_beta,
    )
    # beta_t = _adaptive_beta_gate(
    #     beta_t=beta_t,
    #     beta_init=beta,
    #     min_beta=min_beta,
    #     r_answer=r_answer,
    #     adaptive_beta=adaptive_beta,
    #     ema_decay=ema_decay,
    #     target_em=target_em,
    #     ema_slope_tol=ema_slope_tol,
    #     adaptive_rate=adaptive_rate,
    #     max_beta_step=max_beta_step,
    # )
    r_combined = r_answer + beta_t * r_answerability * (1 - r_answer)

    if add_format:
        r_combined = add_formatting(
            r_combined,
            solution_str,
            ground_truth,
            method=method,
            structure_format_score=structure_format_score,
            final_format_score=final_format_score,
            retrieval_score=retrieval_score,
            decomposition_bonus=decomposition_bonus,
        )

    return r_combined


def compute_score_em_plus_decomposition(
    solution_str,
    ground_truth,
    alpha: float = 0.8,
    beta: float = 0.2,
    training_step: int = 0,
    alpha_warmup_steps: int = 200,
    scheduler_type: str = "linear",
    beta_decay_steps: int = 2000,
    min_beta: float = 0.02,
    adaptive_beta: bool = False,
    ema_decay: float = 0.98,
    target_em: float = 0.65,
    ema_slope_tol: float = 5e-4,
    adaptive_rate: float = 0.2,
    max_beta_step: float = 0.01,
    method: str = "strict",
    format_score: float = 0.0,
    score: float = 1.0,
    add_format: bool = True,
    structure_format_score: float = 0.2,
    final_format_score: float = 0.1,
    retrieval_score: float = 0.1,
    decomposition_bonus: float = 0.0,
):
    """
    Combined reward:
      r = r_answer + beta * r_decomposition * (1 - r_answer)

    Where:
      r_answer in {0, 1} — exact match on final answer
      r_decomposition in [0, 1] — quality of query decomposition at each search step
      beta controls max bonus for good decomposition on a failed answer
    """
    # 1) Final answer reward
    r_answer = compute_score_em(
        solution_str=solution_str,
        ground_truth=ground_truth,
        method=method,
        format_score=format_score,
        score=score,
        add_format=False,
    )

    # 2) Decomposition reward
    judge_fn = _get_judge_fn()
    original_query = extract_question_from_solution(solution_str)
    r_decomposition = compute_r_decomposition(
        solution_str=solution_str,
        original_query=original_query,
        judge_fn=judge_fn,
    )

    # 3) Combine: decomposition only adds signal when answer is wrong
    sample_id = get_sample_id()
    print(f"[{sample_id}] r_answer={r_answer:.4f}  r_decomposition={r_decomposition:.4f}")

    _alpha_t, beta_t = _schedule_alpha_beta(
        alpha,
        beta,
        training_step,
        alpha_warmup_steps,
        schedule_type=scheduler_type,
        beta_decay_steps=beta_decay_steps,
        min_beta=min_beta,
    )
    beta_t = _adaptive_beta_gate(
        beta_t=beta_t,
        beta_init=beta,
        min_beta=min_beta,
        r_answer=r_answer,
        adaptive_beta=adaptive_beta,
        ema_decay=ema_decay,
        target_em=target_em,
        ema_slope_tol=ema_slope_tol,
        adaptive_rate=adaptive_rate,
        max_beta_step=max_beta_step,
    )
    r_combined = r_answer + beta_t * r_decomposition * (1 - r_answer)

    if add_format:
        r_combined = add_formatting(
            r_combined,
            solution_str,
            ground_truth,
            method=method,
            structure_format_score=structure_format_score,
            final_format_score=final_format_score,
            retrieval_score=retrieval_score,
            decomposition_bonus=decomposition_bonus,
        )

    return r_combined



def compute_score_em_plus_answerability_decomposition(
    solution_str,
    ground_truth,
    alpha: float = 0.8,
    beta: float = 0.2,
    training_step: int = 0,
    alpha_warmup_steps: int = 200,
    scheduler_type: str = "smart",
    beta_decay_steps: int = 2000,
    min_beta: float = 0.02,
    adaptive_beta: bool = True,
    ema_decay: float = 0.98,
    target_em: float = 0.65,
    ema_slope_tol: float = 5e-4,
    adaptive_rate: float = 0.2,
    max_beta_step: float = 0.01,
    method: str = "strict",
    format_score: float = 0.0,
    score: float = 1.0,
    add_format: bool = True,
    structure_format_score: float = 0.2,
    final_format_score: float = 0.1,
    retrieval_score: float = 0.1,
    decomposition_bonus: float = 0.0,
):
    """
    Combined reward:
      r = r_answer + beta * (0.5 * r_answerability + 0.5 * r_decomposition) * (1 - r_answer)

    Where:
      r_answer in {0, 1} — exact match on final answer
      r_answerability in [0, 1] — do retrieved docs answer the subqueries
      r_decomposition in [0, 1] — quality of query decomposition at each search step
      beta controls max bonus for good intermediate behavior on a failed answer
    """
    # 1) Final answer reward
    r_answer = compute_score_em(
        solution_str=solution_str,
        ground_truth=ground_truth,
        method=method,
        format_score=format_score,
        score=score,
        add_format=False,
    )

    judge_fn = _get_judge_fn()

    # 2) Answerability reward
    r_answerability = compute_r_answerability(
        solution_str=solution_str,
        judge_fn=judge_fn,
    )

    # 3) Decomposition reward
    original_query = extract_question_from_solution(solution_str)
    r_decomposition = compute_r_decomposition(
        solution_str=solution_str,
        original_query=original_query,
        judge_fn=judge_fn,
    )

    # 4) Combine
    r_intermediate = 0.5 * r_answerability + 0.5 * r_decomposition
    sample_id = get_sample_id()
    # print(f"[{sample_id}] r_answer={r_answer:.4f}  r_answerability={r_answerability:.4f}  r_decomposition={r_decomposition:.4f}  r_intermediate={r_intermediate:.4f}")

    _alpha_t, beta_t = _schedule_alpha_beta(
        alpha,
        beta,
        training_step,
        alpha_warmup_steps,
        schedule_type=scheduler_type,
        beta_decay_steps=beta_decay_steps,
        min_beta=min_beta,
    )
    beta_t, ema, slope, gate_mode = _adaptive_beta_gate(
        beta_t=beta_t,
        beta_init=beta,
        min_beta=min_beta,
        r_answer=r_answer,
        adaptive_beta=adaptive_beta,
        ema_decay=ema_decay,
        target_em=target_em,
        ema_slope_tol=ema_slope_tol,
        adaptive_rate=adaptive_rate,
        max_beta_step=max_beta_step,
    )
    r_combined = r_answer + beta_t * r_intermediate * (1 - r_answer)

    if add_format:
        r_combined = add_formatting(
            r_combined,
            solution_str,
            ground_truth,
            method=method,
            structure_format_score=structure_format_score,
            final_format_score=final_format_score,
            retrieval_score=retrieval_score,
            decomposition_bonus=decomposition_bonus,
        )
    print(
        f"[{sample_id}] "
        f"step={training_step} "
        f"r_answer={r_answer:.4f} "
        f"r_answerability={r_answerability:.4f} "
        f"r_decomposition={r_decomposition:.4f} "
        f"r_intermediate={r_intermediate:.4f} "
        f"beta={beta_t:.4f} "
        f"ema={ema if ema is not None else -1:.4f} "
        f"slope={slope if slope is not None else -1:.6f} "
        f"gate={gate_mode}"
    )
    return r_combined


def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        sample_id = get_sample_id()
        print(f"[{sample_id}] --------------------------------")
        print(f"[{sample_id}] Reasoning Trace: {solution_str}")
        print(f"[{sample_id}] Extracted answer: {answer}")
        print(f"[{sample_id}] Golden answers: {ground_truth['target']}")
    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth['target']):
            return score
        else:
            return format_score
