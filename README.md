# CITY-Jev: Evaluating Agent Execution Decisions Under Perturbations

**CanITrustU-Jev** provides a quick, exploratory evaluation of **how Jev-style models perform on execution decisions in agent workflows**. It covers tasks such as choosing actions, assessing intermediate steps, verifying outcomes, and evaluating evidence and safety across different application scenarios. The goal is to understand where these models perform well, where they struggle, and how confidence-based abstention affects their reliability. Input perturbations provide an additional view of how sensitive their decisions are to changes in presentation.

We transform 10 upstream data sources into a common candidate-selection format. Each question is evaluated with its original input and five perturbations, yielding **2,000 questions and 12,000 decision evaluations**. We report Accuracy and Abstention-Aware Accuracy by application scenario, decision task, answer format, and source dataset, using both worst-case and mean scores across the six inputs.

This is an independent evaluation of adapted tasks; its scores are not the official scores of the upstream benchmarks. The results below come from a single run of **Jev 1.13.0**. Configurations for other model adapters do not imply completed evaluations.

> **Review status:** This project was developed primarily using automated tools, with partial human involvement and review. Its data transformations, perturbations, evaluation logic, reported results, and documentation require further verification. The current content should be considered preliminary.

## Data Sources

Counts below refer to **original questions in this evaluation**, before excluding failed requests. They are not the sizes of the upstream datasets and do not necessarily represent independent trajectories. Links point to upstream projects or dataset repositories.

| Source | Dataset ID | Questions | Adapted Task and Answer Format | Ground-Truth Basis |
|---|---|---:|---|---|
| [AgentProcessBench](https://github.com/RUCBM/AgentProcessBench) | `agentprocess` | 450 | Process evaluation; fixed-category classification | Upstream step-level process-quality annotations |
| [AgentRewardBench](https://huggingface.co/datasets/McGill-NLP/agent-reward-bench) | `agentreward` | 135 | Success, loop, or side-effect verification; yes/no judgment | Agreement among upstream annotations |
| [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) | `bfcl` | 200 | Function selection; dynamic candidate selection | Correct-function labels from V4 multiple |
| [Mind2Web](https://github.com/OSU-NLP-Group/Mind2Web) | `mind2web` | 203 | Web target selection; dynamic candidate selection | Upstream positive targets and negative candidates |
| [WebLINX](https://github.com/McGill-NLP/weblinx) | `weblinx` | 49 | Web action selection with dialogue context; dynamic candidate selection | Upstream target-action annotations |
| [SWE-agent trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories) | `swetraj` | 100 | Software repair outcome verification; yes/no judgment | Execution outcome labels attached to trajectories |
| [LongMemEval](https://github.com/xiaowu0162/LongMemEval) | `longmemeval` | 250 | Whether evidence supports an answer; yes/no judgment | Oracle evidence and answerability annotations |
| [AgentHarm](https://huggingface.co/datasets/ai-safety-institute/AgentHarm) | `agentharm` | 200 | Request safety analysis; yes/no judgment | Harmful / benign source labels |
| [InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) | `injecagent` | 250 | Injection attack type identification; fixed-category classification | Upstream attack-type annotations |
| [WorkArena](https://github.com/ServiceNow/WorkArena) | `workarena` | 163 | Consistency between knowledge and an answer; yes/no judgment | Correct values from knowledge-task configurations and rule-generated incorrect values |
| **Total** | | **2,000** | | |

The `gold` answer comes from upstream annotations, execution outcomes, or verifiable rules applied to task configurations. Generative models are used for some expression perturbations, not to generate ground-truth answers. Sampling is limited to locally available data and does not cover every upstream dataset in full. Label distributions are not uniformly balanced.

### Classification Dimensions

- **Application scenario:** business service interactions, information retrieval and knowledge QA, tool and API interactions, web and browser operations, and software development and repair. Both ordinary and dialogue-based web navigation fall under web and browser operations. AgentProcessBench is classified by its internal task domains.
- **Decision task:** safety analysis, outcome verification, action selection, evidence assessment, and process evaluation.
- **Answer format:** dynamic candidate selection, fixed-category classification, and yes/no judgment. All three use a candidate-selection interface.

### Five Perturbations

Each perturbation is applied independently to the original question. The aim is to change input expression while preserving the meaning of the correct answer.

| Identifier | Perturbation | Input Change |
|---|---|---|
| `v0_original` | Original input | The original question in the common schema |
| `v1_order` | Option order reversal | Reverse candidate IDs and their descriptions together |
| `v2_label` | Option identifier replacement | Use opaque IDs and preserve original labels in nested descriptions, also changing the description structure |
| `v3_format` | State formatting | Serialize the state as JSON text, changing formatting, quotation, or escaping |
| `v4_context` | Irrelevant context insertion | Add auxiliary material about another case and instructions limiting the task scope; auxiliary text is generated per question |
| `v5_paraphrase` | Instruction paraphrasing | Rewrite decision instructions while retaining the intended task |

Generated context and paraphrases may introduce semantic deviations. The perturbations have not undergone exhaustive human verification of semantic equivalence.

<!-- JEV-ROBUST-RESULTS:START -->
## Results

Model: `jev-1.13.0`. Confidence threshold: **0.5**.

Of **2,000** original questions, **26** are excluded following **56** failed requests, leaving **1,974** questions and **11,844** decision evaluations. If any request fails or any prediction is missing for a question or its perturbations, the entire question is excluded from scoring.

- **Per-decision Accuracy:** 1 if `choice == gold`, otherwise 0.
- **Per-decision Abstention-Aware Accuracy:** 1 if the answer is correct or the API returns `confidence < 0.5`, otherwise 0. Low confidence is treated as abstention by an offline policy; confidence equal to the threshold is treated as answering.
- **Strict score:** take the minimum score across the original input and five perturbations for each question, then average over included questions. All six decisions must pass for a question to score 1.
- **Mean score:** average the six decision scores for each question, then average over included questions.

The threshold of 0.5 follows an example in the [TypeSafe Confidence documentation](https://docs.typesafe.ai/confidence); it is not a mandatory universal threshold. We use the returned `confidence`, not the highest class probability. Abstention-Aware Accuracy is defined for this project and should be read alongside the abstention rate.

Abstention rate across included decisions: **19.37%**.

### By Application Scenario

| Category | Included Questions | Excluded Questions | Strict Accuracy | Mean Accuracy | Strict Abstention-Aware Accuracy | Mean Abstention-Aware Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Business Service Interactions | 158 | 0 | 50.63% | 53.59% | 62.66% | 67.72% |
| Information Retrieval and Knowledge QA | 565 | 1 | 73.98% | 80.56% | 87.96% | 91.45% |
| Tool and API Interactions | 789 | 0 | 81.50% | 84.37% | 88.72% | 91.21% |
| Web and Browser Operations | 362 | 25 | 70.72% | 75.87% | 83.98% | 89.64% |
| Software Development and Repair | 100 | 0 | 74.00% | 77.67% | 84.00% | 84.67% |
| Overall | 1974 | 26 | 74.52% | 78.92% | 85.31% | 88.78% |

### By Decision Task

| Category | Included Questions | Excluded Questions | Strict Accuracy | Mean Accuracy | Strict Abstention-Aware Accuracy | Mean Abstention-Aware Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Safety Analysis | 450 | 0 | 84.22% | 87.78% | 90.89% | 93.15% |
| Outcome Verification | 229 | 6 | 74.24% | 78.97% | 82.10% | 85.01% |
| Action Selection | 433 | 19 | 83.14% | 85.80% | 92.38% | 95.73% |
| Evidence Assessment | 413 | 0 | 83.78% | 89.43% | 92.98% | 95.92% |
| Process Evaluation | 449 | 1 | 48.11% | 53.71% | 67.48% | 73.05% |
| Overall | 1974 | 26 | 74.52% | 78.92% | 85.31% | 88.78% |

### By Answer Format

| Category | Included Questions | Excluded Questions | Strict Accuracy | Mean Accuracy | Strict Abstention-Aware Accuracy | Mean Abstention-Aware Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Dynamic Candidate Selection | 433 | 19 | 83.14% | 85.80% | 92.38% | 95.73% |
| Fixed-Category Classification | 699 | 1 | 61.95% | 66.48% | 76.11% | 80.33% |
| Yes/No Judgment | 842 | 6 | 80.52% | 85.71% | 89.31% | 92.22% |
| Overall | 1974 | 26 | 74.52% | 78.92% | 85.31% | 88.78% |

### By Source Dataset

| Category | Included Questions | Excluded Questions | Strict Accuracy | Mean Accuracy | Strict Abstention-Aware Accuracy | Mean Abstention-Aware Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| agentharm | 200 | 0 | 81.00% | 85.75% | 90.00% | 92.83% |
| agentprocess | 449 | 1 | 48.11% | 53.71% | 67.48% | 73.05% |
| agentreward | 129 | 6 | 74.42% | 79.97% | 80.62% | 85.27% |
| bfcl | 200 | 0 | 100.00% | 100.00% | 100.00% | 100.00% |
| injecagent | 250 | 0 | 86.80% | 89.40% | 91.60% | 93.40% |
| longmemeval | 250 | 0 | 73.20% | 82.53% | 88.40% | 93.27% |
| mind2web | 184 | 19 | 79.89% | 84.78% | 92.93% | 96.65% |
| swetraj | 100 | 0 | 74.00% | 77.67% | 84.00% | 84.67% |
| weblinx | 49 | 0 | 26.53% | 31.63% | 59.18% | 74.83% |
| workarena | 163 | 0 | 100.00% | 100.00% | 100.00% | 100.00% |
| Overall | 1974 | 26 | 74.52% | 78.92% | 85.31% | 88.78% |

<!-- JEV-ROBUST-RESULTS:END -->

## Running the Evaluation

### 1. Environment Setup

Python 3.9+ is required. From the repository root, create a virtual environment and install all dependencies needed for Jev evaluation:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install tqdm
```

### 2. Jev Evaluation Command

```bash
export JEV_API_KEY='replace-with-your-api-key'
python -m src.eval --model jev --input data/final/versions.jsonl
```

### 3. Notes

- Prepare evaluation data separately. `--input` accepts a JSONL file or a directory containing gzip JSONL shards and a `manifest.json`.
- By default, all questions are evaluated with their five perturbations. Use `--limit-bases 10` to evaluate 10 questions first.
- Jev is accessed through an API; no model weights need to be downloaded. The default model is `jev-1.13.0`. Override the endpoint with `JEV_ENDPOINT`; see [configs/models.json](configs/models.json) for configuration.
- Input truncation is disabled by default. Set `"max_input_tokens": 32000` in the model configuration to enable an input budget, removing earlier history while preserving content near the decision point.
- Run `python -m src.eval --help` for additional arguments. Local inference dependencies for other models must be installed separately for their respective adapters.

## Scope and Limitations

- These scores measure adapted candidate-selection decisions, not end-to-end agent success. BFCL measures function selection only; Mind2Web uses candidate sets containing the correct target; WorkArena measures knowledge-value consistency without executing browser tasks.
- LongMemEval uses oracle evidence and does not measure full long-context retrieval. The InjecAgent samples contain known attacks, so they cannot establish false-positive rates on benign inputs.
- Source sizes and label distributions differ. Overall scores are weighted by question count. Multiple questions may share a trajectory or underlying problem and should not be treated as fully independent samples.
- Input truncation was not enabled for this evaluation. Of the 26 questions excluded due to failed requests, 25 belong to web and browser operations. The tables describe performance on the included questions.
- Abstention-Aware Accuracy gives full credit for either a correct answer or low-confidence abstention. Interpret it together with Accuracy, the abstention rate, and the confidence threshold. Results come from one run and do not include uncertainty estimates across repeated runs.

## Citation and Acknowledgments

If you use this project's methods, code, or results, you can cite the repository:

```bibtex
@misc{canitrustujev2026,
  author       = {{CanITrustU-Jev}},
  title        = {CITY-Jev: Evaluating Agent Execution Decisions Under Perturbations},
  year         = {2026},
  howpublished = {\url{https://github.com/JiaQiSJTU/CanITrustU-Jev}},
  note         = {GitHub repository}
}
```

This project builds on the 10 upstream sources listed above. When using their data or adapted examples, also cite the relevant upstream projects or papers and follow their respective licenses and usage terms. Citing this repository does not replace upstream attribution. The confidence-based abstention policy draws on the [TypeSafe Confidence documentation](https://docs.typesafe.ai/confidence).
