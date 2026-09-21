# EdgeCascade Capability Boundary Report

> **Analysis Date:** Auto-generated after running capability probe.
> **Model:** Custom GPT-2 124M + Q/V LoRA (rank=8, alpha=16) + SFT adapter (30 Q&A pairs, 8 epochs)
> **Probe size:** 40 questions across 7 categories (ISRO + DPDPA domains)

---

## 1. Overall Results

| Metric | Value |
|--------|-------|
| Total probe questions | 41 |
| GOOD | 0 (0.0%) |
| PARTIALLY_GOOD | 2 (4.9%) |
| BAD | 35 (85.4%) |
| UNKNOWN | 4 (9.8%) |
| **Strict success rate (GOOD only)** | **0.0%** |
| **Broad success rate (GOOD + PARTIAL)** | **4.9%** |

---

## 2. Results by Category

| Category | N | GOOD | PARTIAL | BAD | UNK | Strict% | Broad% | Avg logP |
|----------|---|------|---------|-----|-----|---------|--------|----------|
| BASIC_DEFINITION | 6 | 0 | 1 | 5 | 0 | 0% | 17% | -1.219 |
| BASIC_TERMINOLOGY | 6 | 0 | 0 | 6 | 0 | 0% | 0% | -1.026 |
| MULTI_FACT | 5 | 0 | 0 | 4 | 1 | 0% | 0% | -1.361 |
| REASONING | 6 | 0 | 0 | 5 | 1 | 0% | 0% | -1.383 |
| SIMPLE_CONCEPT | 6 | 0 | 1 | 5 | 0 | 0% | 17% | -1.350 |
| SIMPLE_RELATIONSHIP | 6 | 0 | 0 | 5 | 1 | 0% | 0% | -1.397 |
| SPECIFIC_FACT | 6 | 0 | 0 | 5 | 1 | 0% | 0% | -1.478 |

---

## 3. Capability Boundary Analysis

### 3.1 What can the 124M model reliably do?

**No categories reached the defined Tier-1-safe thresholds (strict ≥ 50%, broad ≥ 65%).**

This is a scientifically honest negative result. The 124M SFT model does not have a meaningfully reliable capability region at current training scale.

### 3.3 Categories that are unsafe for Tier 1

Categories where the model consistently fails (broad < 40%): **BASIC_DEFINITION, BASIC_TERMINOLOGY, MULTI_FACT, REASONING, SIMPLE_CONCEPT, SIMPLE_RELATIONSHIP, SPECIFIC_FACT**


### 3.4 Failure mode distribution (BAD + UNKNOWN answers)

| Failure Mode | Count |
|--------------|-------|
| hallucination | 32 |
| other | 5 |
| irrelevant | 1 |
| incompleteness | 1 |

---

## 4. Log-Probability as a Quality Signal

| Quality Label | N | Mean Avg-LogP | Min | Max |
|---------------|---|---------------|-----|-----|
| GOOD | 0 | N/A | N/A | N/A |
| PARTIALLY_GOOD | 2 | -1.2981 | -1.5498 | -1.0464 |
| BAD | 35 | -1.2981 | -1.9094 | -0.627 |
| UNKNOWN | 4 | -1.4722 | -1.631 | -1.2073 |

Insufficient data to determine log-probability utility.

### 4.1 Confidently wrong examples (high log-prob + BAD quality)

**Confidently wrong** (`probe_A_002`)
- **Q:** What is a Geographic Information System (GIS)?
- **GT:** GIS is a computer-based system designed to capture, store, manipulate, analyze, manage, and present all types of geographical data.
- **Generated:** A GIS is a remote sensing, imagery and sensing system that is used to identify, measure, classify, and manage land and ocean features. It is used primarily in navigation, surveying, and mapping, and i
- **Quality:** BAD (score=1)
- **Reason:** The generated answer incorrectly defines GIS as a remote sensing system rather than a computer-based system for managing and analyzing spatial data. It also hallucinates specific, incorrect primary users (U.S. Navy, commercial aviation) and functions, failing to address the core definition provided in the ground truth.
- **Failure mode:** hallucination
- **Avg log-prob:** -1.39163


**Confidently wrong** (`probe_A_003`)
- **Q:** What is a Data Fiduciary under the DPDPA?
- **GT:** A Data Fiduciary is any person who alone or in conjunction with other persons determines the purpose and means of processing of personal data.
- **Generated:** A Data Fiduciary is a person who holds a position of trust in a Data Authority or an entity that has responsibility for the maintenance and operation of data.

37

Question:
What is an Data Commission
- **Quality:** BAD (score=1)
- **Reason:** The generated answer is factually incorrect. It confuses the definition of a Data Fiduciary with that of a Data Processor or a government official, whereas the ground truth correctly defines it as the entity determining the purpose and means of processing. Additionally, the answer includes irrelevant text ('37') and fails to address the second question about the Data Commissioner.
- **Failure mode:** hallucination
- **Avg log-prob:** -0.704586



---

## 5. Routing Implications

No category achieved a success rate warranting a reliable Tier-1 routing policy.

The model's failure modes (hallucination, repetition, instruction-following failures) are too 
pervasive and unpredictable. Even in categories where some answers are correct, the judge would 
need to catch all failures — and the failure rate is high enough that the benefit of Tier-1 usage 
is negligible.

> **Verdict: 124M HAS NO MEANINGFUL SAFE CAPABILITY REGION.** 
> Keep it as the from-scratch model component demonstrating the architecture, 
> but do not rely on it for practical Tier-1 routing.

---

## 6. Final Recommendation

**OPTION B: 124M HAS NO MEANINGFUL SAFE CAPABILITY REGION**

Keep it as the from-scratch architecture demonstration. 
Do not force Tier-1 routing. The cascade correctly escalates almost everything,
which is the scientifically honest result given the model's scale and training.