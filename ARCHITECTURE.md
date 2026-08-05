# ORBIT System Architecture

## Overview

ORBIT is a 10-layer autonomous decision intelligence platform built to solve the universal problem across all Big Tech: **"What action should we take for this user right now?"**

## System Design Principles

1. **Causal Over Correlational** — Discover cause-effect, not just patterns
2. **Autonomous Over Manual** — Run 50+ micro-experiments daily without human intervention
3. **Constraint-Aware** — Built for 16GB M5 MacBook; every optimization is a feature
4. **Production-Ready** — Docker, CI/CD, monitoring, drift detection, canary deployments
5. **Explainable** — Every decision cites features, values, and confidence intervals

## Layer-by-Layer Architecture

### Layer 1: Data Ingestion & Processing
- **Source:** YooChoose RecSys Challenge (9M clicks) + synthetic enrichment
- **Format:** Apache Parquet (columnar, compressed)
- **Processing:** Polars (lazy evaluation, 10-50x faster than Pandas)
- **Analytics:** DuckDB (embedded OLAP, queries Parquet directly)
- **Scale Path:** Kafka → Spark Streaming → Delta Lake (documented)

### Layer 2: Feature Engineering (10 Time-Series Features)
| Feature | Description | Business Signal |
|---------|-------------|-----------------|
| click_velocity_5min | Events per 5-min window | Fraud, engagement spike |
| session_recency_hours | Hours since last session | Churn predictor |
| category_entropy_7d | Shannon entropy of categories | Engagement depth |
| conversion_rate_30d | Purchases / sessions | User quality |
| cart_abandonment_streak | Consecutive add-to-cart without purchase | Purchase friction |
| device_fingerprint_risk | Device + geo + time mismatch | Account takeover |
| ltv_trend_slope | Spend slope over 90 days | Revenue trajectory |
| hour_of_day_preference | Most active hour | Personalized timing |
| cross_category_breadth | Distinct categories explored | Exploration signal |
| days_since_first_seen | User tenure | New vs. veteran |

### Layer 3: Feature Store
- **Online:** Redis (<5ms retrieval, TTL eviction, LRU)
- **Offline:** DuckDB (batch compute, backfills)
- **Production Path:** Feast + Redis Cluster + Snowflake

### Layer 4: Model Training
- **Primary:** XGBoost (interpretable, SHAP, <20ms inference)
- **Baseline:** PyTorch Multi-Task NN (churn + engagement + LTV)
- **HPO:** Optuna (Bayesian optimization, early pruning)
- **Tracking:** MLflow (every run logged with params, metrics, artifacts)
- **Registry:** Staging → Production gates (AUC > 0.80, F1 > 0.75, latency < 50ms)

### Layer 5: Causal Uplift Engine (The Differentiator)
- **Meta-Learners:** S-Learner, T-Learner, X-Learner (primary), R-Learner
- **Propensity Score Matching:** Unbiased effect estimation
- **4 Interventions:** Tutorial ($0), Discount ($12), Reminder ($0), No-Action ($0)
- **Decision Logic:** Only act if uplift > cost_threshold with 90%+ confidence

### Layer 6: Decision Optimizer & Serving
- **API:** FastAPI (async, Pydantic v2, auto OpenAPI docs)
- **Cache:** Redis (hot features, <5ms)
- **Optimizer:** Multi-objective (cvxpy) — retention, cost, frequency caps
- **Latency:** p50 <20ms, p99 <50ms
- **Fallback:** Rule-based if model times out; "no action" if uncertain

### Layer 7: Experimentation
- **Assignment:** Hash-based bucketing (deterministic, stratified)
- **Thompson Sampling:** Explore vs. exploit with Beta distributions
- **Sequential Testing:** Early stopping with valid Type-I error control
- **Guardrails:** Auto-shutdown if false_positive > 2% or metric_drop > 5%

### Layer 8: Monitoring & Observability
- **Metrics:** Prometheus (latency p50/p95/p99, throughput, error rate)
- **Dashboards:** Grafana (4 panels: system health, model perf, business impact, data quality)
- **Drift Detection:** PSI > 0.2 or KS-test p < 0.01 → trigger retraining

### Layer 9: Explanation Engine
- **Model:** Qwen2.5-3B-Instruct (3B params, fits in 3GB RAM quantized)
- **Fine-Tuning:** Unsloth + QLoRA (rank=64, 4-bit NF4)
- **Inference:** llama-cpp-python (MPS backend, 25+ tok/sec)
- **Citation:** Every claim MUST cite feature_name + value + model_weight

### Layer 10: MLOps & Infrastructure
- **Containerization:** Docker Compose (API, Redis, DuckDB, Prometheus, Grafana)
- **CI/CD:** GitHub Actions (Lint → Test → Integration → Eval → Deploy)
- **Data Versioning:** DVC (track datasets, features, model artifacts)
- **Deployment:** Canary (10% → 50% → 100%) with auto-rollback

## Scaling Path: 1M → 10B+ Records

| Component | Prototype (Laptop) | Production (10B+/day) |
|-----------|-------------------|----------------------|
| Data | Parquet files | Kafka (256 partitions) → Spark Streaming → Delta Lake |
| Processing | Polars (batch) | Spark Structured Streaming |
| Analytics | DuckDB | Snowflake / BigQuery |
| Feature Store | Custom Redis+DuckDB | Feast + Redis Cluster + Snowflake |
| Training | Single machine | SageMaker / Vertex AI (distributed) |
| Serving | FastAPI single instance | Kubernetes HPA (50+ pods) |
| LLM | llama.cpp local | vLLM on GPU cluster |

**What stays the same:** Feature definitions, model architecture, API contract, experiment logic, causal uplift algorithm.

## 16GB M5 Optimization

| Technique | Memory Saved |
|-----------|--------------|
| Unsloth (QLoRA training) | 50% less RAM |
| 4-bit Quantization (NF4) | ~75% |
| Gradient Checkpointing | ~30% |
| Flash Attention 2 (MPS) | 20-30% speedup |
| ONNX / llama.cpp inference | 60% less RAM |
| Polars over Pandas | 80% less RAM |
| DuckDB over PostgreSQL | Embedded, zero overhead |

**Total runtime RAM: ~10-11GB (comfortable headroom on 16GB)**

## Interview Defense Highlights

**Meta:** "ORBIT causally estimates which notification type has the highest uplift for THIS user — not just predicting dormancy."

**Google:** "The causal engine estimates the effect of featured snippets vs. shopping results on 7-day return rate, not just CTR."

**Amazon:** "Instead of blasting discounts, ORBIT estimates causal uplift per user segment — tutorial for power users (+22%), discount for price-sensitive (+8%)."

**Netflix:** "The multi-objective optimizer weighs re-engagement against unsubscribe risk. Sometimes silence is optimal."

**Apple:** "ORBIT detects that new users respond to tutorials (+31%) while veterans respond to social features (+12%)."

---

*For the complete 1,400-line blueprint, see [ORBIT_Project_Blueprint.md](ORBIT_Project_Blueprint.md)*
