# ORBIT 🛰️
### Autonomous User Lifecycle Intelligence Platform

> **Predict → Estimate Causal Uplift → Decide → Execute → Learn**

ORBIT is an end-to-end ML system that solves the universal problem across all Big Tech companies: **"What action should we take for this user right now?"**

Unlike churn prediction (which only tells you *who* is at risk) or recommendation systems (which only suggest *what* to show), ORBIT **causally estimates the impact of every possible intervention** for each individual user, selects the optimal action, executes it, and learns autonomously.

---

## 🎯 The Universal FAANG Problem

| Company | Their Version |
|---------|--------------|
| **Meta** | Show Reel, ad, or friend post? Notification or silence? |
| **Google** | Featured snippet, shopping result, or AI overview? |
| **Amazon** | Abandoned cart: free shipping, 10% off, or nothing? |
| **Netflix** | Push notification, email, or let subscriber be? |
| **Apple** | Curated playlist or family plan nudge? |
| **Microsoft** | Feature tip, integration suggestion, or ignore? |
| **Uber** | Discount, destination suggestion, or nothing? |
| **Spotify** | Different playlist, genre expansion, or let churn? |

**The gap:** Every company can predict risk. No company can autonomously decide the optimal action with causal rigor.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1: DATA INGESTION (1M+ real events, Polars, DuckDB)  │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  LAYER 2: FEATURE ENGINEERING (10 time-series features)     │
│  click_velocity | session_recency | category_entropy      │
│  conversion_rate | cart_abandonment | device_risk          │
│  ltv_trend | hour_preference | cross_category | tenure     │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  LAYER 3: FEATURE STORE (Redis online + DuckDB offline)     │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  LAYER 4: MODELS (XGBoost primary, PyTorch MTNN baseline) │
│  MLflow tracking | Optuna HPO | SHAP interpretability       │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  LAYER 5: CAUSAL UPLIFT ENGINE (X-Learner, Propensity Match)│
│  4 Interventions: Tutorial | Discount | Reminder | No-Action│
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  LAYER 6: DECISION OPTIMIZER (FastAPI, Multi-Objective)     │
│  p50 <20ms | p99 <50ms | Fallback heuristics                │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  LAYER 7: EXPERIMENTATION (Thompson Sampling, Sequential)   │
│  10% holdout | Auto-shutdown guardrails                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  LAYER 8: MONITORING (Prometheus, Grafana, Drift Detection) │
│  PSI > 0.2 → Auto-retrain trigger                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  LAYER 9: EXPLANATION ENGINE (Qwen2.5-3B, QLoRA, 4-bit)      │
│  Citation-enforced | Runs fully local on 16GB M5            │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  LAYER 10: MLOps (Docker, DVC, GitHub Actions, Canary)        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧠 Why This Is "God Level"

| What Exists Today | Why It Fails |
|-------------------|--------------|
| Churn prediction | Tells you *who* is at risk, not *what to do* |
| Recommendation systems | Optimize CTR, not causal business outcomes |
| A/B testing platforms | Human-designed, slow, not autonomous |
| LLMs (ChatGPT, Claude) | Hallucinate causality, no uplift estimation |
| Reinforcement Learning | Needs billions of interactions, too risky |

**ORBIT closes the loop:** raw signals → causal uplift → optimal decision → execution → outcome logging → model update.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Data Processing** | Polars (lazy eval), DuckDB (embedded OLAP) |
| **Feature Store** | Redis (online), DuckDB (offline) → Feast (prod) |
| **Models** | XGBoost (primary), PyTorch Multi-Task NN (baseline) |
| **HPO** | Optuna (Bayesian) |
| **Causal ML** | Custom X-Learner + econml |
| **API** | FastAPI (async, Pydantic v2) |
| **Experimentation** | Custom Thompson Sampling + Sequential Testing |
| **Monitoring** | Prometheus + Grafana |
| **LLM** | Qwen2.5-3B-Instruct, Unsloth QLoRA, llama.cpp |
| **MLOps** | MLflow, DVC, GitHub Actions, Docker Compose |
| **Hardware** | Apple M5, 16GB RAM (fully local) |

---

## 📅 30-Day Build Roadmap

| Week | Focus | Deliverable |
|------|-------|-------------|
| **Week 1** | Data + Features + Store | 1M+ records, 10 features, Redis+DuckDB store |
| **Week 2** | Models + Causal Uplift + MLflow | XGBoost (0.81 AUC), X-Learner, registry |
| **Week 3** | Serving + Decision + A/B Testing | FastAPI <50ms, optimizer, Thompson Sampling |
| **Week 4** | Monitor + Explain + Polish | Grafana, 3B SLM, CI/CD, blog post |

---

## 📄 Project Documentation

- **[ORBIT_Project_Blueprint.md](ORBIT_Project_Blueprint.md)** — Complete 1,400-line architecture document covering every layer, design decisions, interview defense strategy, and scaling path to 10B+ records.
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — High-level system design with diagrams.
- **[SETUP.md](SETUP.md)** — Environment setup and installation guide.

---

## 👤 Author

**Jay Patel** — Data Scientist targeting FAANG-level ML Engineering roles  
📧 pjay205335@gmail.com | 🔗 [LinkedIn](https://linkedin.com/in/jay-patel-bba86a352) | 🌐 [Portfolio](https://jaypatel-data.netlify.app)

---

*Built for 16GB M5 MacBook. Architected for 10B+ users.*
