# Defender: AI-Powered Banking Fraud Prevention System

> A comprehensive deep learning fraud detection platform built with PyTorch, featuring 68 trained models, real-time streaming, and production deployment on Google Cloud Platform.


---

## Capstone Project

This project was proposed by **Professor Ammar Rabat** from the Business Administration Department at **Stanton University**. The project uses anonymized data that closely resembles actual banking data, provided by Professor Rabat and his partners from industry.

### Purpose

This capstone project simulates a real-world enterprise engagement with a major financial institution in the Middle East. The objective is to design and prototype an AI-powered fraud detection system that identifies anomalous transactions across multiple banking channels using advanced analytics and artificial intelligence (including machine learning, neural networks, and generative AI).

The project integrates competencies in business intelligence, systems analysis and design, data visualization, and programming, offering students the opportunity to apply their knowledge to a solution that is both technically robust and professionally relevant.

### Advisers

- **Dr. Anh-Thu Phan-Ho**
- **Professor Ammar Rabat**

*This was a solo project - all development, analysis, and deployment was completed independently.*

---

### IEEE CIFEr 2026 Submission

This work has been submitted to the **2026 IEEE Conference on Computational Intelligence in Financial Engineering and Economics (CIFEr)** in Tokyo, Japan.

> *"My Defender framework catches 11 types of banking fraud. Hopefully the reviewers don't find a 12th: 'obviously overfitted.'"*

[![IEEE](https://img.shields.io/badge/IEEE_CIFEr_2026-Submitted-00629B?style=for-the-badge)](assets/IEEE_CIFEr_2026_Submission.pdf)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [The Problem](#the-problem)
3. [The Discovery](#the-discovery)
4. [Technical Architecture](#technical-architecture)
5. [Data Pipeline](#data-pipeline)
6. [Deep Learning Pipeline](#deep-learning-pipeline)
7. [Model Training Results](#model-training-results)
8. [Key Findings & Insights](#key-findings--insights)
9. [Dashboard & API](#dashboard--api)
10. [Production Deployment](#production-deployment)
11. [Technologies Used](#technologies-used)
12. [Lessons Learned](#lessons-learned)
13. [Future Enhancements](#future-enhancements)

---

## Executive Summary

### Project Metrics at a Glance

| Metric | Value |
|--------|-------|
| Deep Learning Models Trained | 68 |
| Traditional ML Models | 4 |
| Analysis Categories | 17 |
| Transactions Analyzed | 40,000 |
| Database Tables | 87 |
| Dashboard Pages | 9 |
| Best Model AUC | 0.6036 (GRU) |
| Multi-Class Detection Rate | 86-100% per fraud type |
| Deployment | GCP + Kubernetes (k3s) |

### The Core Innovation

**Binary fraud detection achieves only 2.9% detection rate.** By reframing the problem as multi-class classification (12 classes: Normal + 11 fraud types), I achieved **86-100% detection rates** per fraud type - a **32x improvement**.

<details>
<summary><b>VIEW DASHBOARD SCREENSHOTS</b></summary>

<br>

### All 68 Deep Learning Models

<div align="center">
<img src="assets/68-deep-learning-models.png" alt="All 68 Deep Learning Models" width="700"/>
</div>

### Business Impact & Cost-Benefit Analysis

<div align="center">
<img src="assets/business-impact.png" alt="Business Impact Dashboard" width="700"/>
<br><br>
<img src="assets/threshold-business-impact.png" alt="Threshold Optimization" width="700"/>
</div>

### Fraud Pattern Analysis

<div align="center">
<img src="assets/fraud-sequence-timeline.png" alt="Fraud Sequence Timeline" width="700"/>
<br><br>
<img src="assets/timegap-analysis.png" alt="Time Gap Analysis" width="700"/>
</div>

### Customer Fraud Profiling

<div align="center">
<img src="assets/customer-fraud-types.png" alt="Customer Fraud Types" width="700"/>
<br><br>
<img src="assets/customer-fraud-individual.png" alt="Individual Customer Fraud Profile" width="700"/>
<br><br>
<img src="assets/customer-pattern-trigger.png" alt="Customer Pattern Triggers" width="700"/>
<br><br>
<img src="assets/key-insights-customer-fraud.png" alt="Key Insights - Customer Fraud" width="700"/>
</div>

### Heatmaps & Coordinated Fraud Detection

<div align="center">
<img src="assets/heatmap-1.png" alt="Fraud Heatmap 1" width="700"/>
<br><br>
<img src="assets/heatmap-2.png" alt="Fraud Heatmap 2" width="700"/>
<br><br>
<img src="assets/coordinated-fraud.png" alt="Coordinated Fraud Detection" width="700"/>
</div>

</details>

---

## The Problem

### Why Traditional Fraud Detection Fails

Banks lose billions annually to fraud. Traditional fraud detection systems use **binary classification** - they try to answer one question: *"Is this transaction fraud or not?"*

This approach has a critical flaw: **fraudulent transactions are designed to look exactly like normal ones.**

### Evidence from the Data

After extensive analysis of 40,000 transactions, I discovered:

| Metric | Normal Transactions | Fraud Transactions | Difference |
|--------|--------------------|--------------------|------------|
| Channel Distribution | 25% each channel | 25% each channel | **Identical** |
| Average Amount | $2,804 | $2,942 | Only 5% |
| Median Amount | $2,650 | $2,800 | Only 6% |
| Peak Transaction Hour | 2 PM | 2 PM | **Same** |
| Weekend Activity | 28.6% | 28.4% | 0.2% |

**Fraudsters are sophisticated.** They intentionally make their transactions statistically indistinguishable from legitimate ones.

### The Binary Classification Results

After training 68 deep learning models with binary classification:

| Model | AUC | Detection Rate | False Positive Rate |
|-------|-----|----------------|---------------------|
| Best GRU | 0.6036 | 2.9% | 57.7% |
| AttentionLSTM | 0.5573 | 3.1% | 55.2% |
| Transformer | 0.5636 | 2.7% | 58.1% |
| Ensemble (68 models) | 0.8123 | 2.9% | 42.3% |

**Paradox**: Even with a 0.8123 AUC ensemble, only **11 out of 380 fraud cases** were detected. That's **97% of fraud going undetected**.

---

## The Discovery

### Reframing the Question

The breakthrough came from asking a different question:

| Old Approach | New Approach |
|--------------|--------------|
| "Is this fraud?" | "What TYPE of transaction is this?" |
| Binary (2 classes) | Multi-class (12 classes) |
| Averaged patterns | Distinct patterns per type |
| 2.9% detection | **86-100% detection** |

### Why Multi-Class Works

Each fraud type has a **unique signature** that binary classification destroys by averaging:

| Fraud Type | Channel | Distinct Pattern | Detection Rate |
|------------|---------|------------------|----------------|
| LimitAbuse | ATM | Amounts near $5,000 daily limit | **100.0%** |
| RapidWithdrawals | ATM | 3+ withdrawals within 1 hour | **96.7%** |
| RepeatedSmallAmount | POS | 5+ transactions under $100 same day | **96.6%** |
| CrossBorder | ATM | International ATM, domestic account | **95.7%** |
| MultiChannelFraud | Mixed | ATM + Digital + POS within 30 min | **95.0%** |
| SuspiciousOverCounter | Branch | Cash > $5,000 at branch | **91.7%** |
| CredentialTheft | Digital | New device + high-value transfer | **91.2%** |
| CrossCountryPOS | POS | POS in different country than home | **88.5%% |
| HighFrequencyMerchant | POS | Same merchant 3+ times per hour | **88.5%** |
| IdentityTheft | Branch | New account + immediate large withdrawal | **88.0%** |
| SimSwap | Digital | Activity within 24h of SIM change | **86.3%** |

**Average Multi-Class Detection: 93.3%** vs Binary: 2.9%

> **Note on Metrics**: Multi-class performance is evaluated using per-class recall and confusion matrices, not ROC-AUC, which is not directly comparable to binary classification metrics.

---

## Technical Architecture

### System Overview

```
                    +-------------------------------------------+
                    |           GCP VM (e2-medium)              |
                    |                                           |
                    |  +-------------------------------------+  |
                    |  |          k3s Cluster                |  |
                    |  |                                     |  |
   Internet --------+--+--> Streamlit Dashboard (:30501)     |  |
                    |  |            |                        |  |
                    |  |            v                        |  |
                    |  |       FastAPI Backend (:8000)       |  |
                    |  |            |                        |  |
                    |  |            v                        |  |
                    |  |      MySQL Database                 |  |
                    |  |    (87 tables, Star Schema)         |  |
                    |  +-------------------------------------+  |
                    +-------------------------------------------+
```

### Component Details

| Component | Technology | Purpose |
|-----------|------------|---------|
| Frontend | Streamlit | 9-page interactive dashboard |
| Backend | FastAPI + Uvicorn | REST API for predictions |
| Database | MySQL 8.0 | Star schema with 87 tables |
| ML Framework | PyTorch | Deep learning training |
| Orchestration | Kubernetes (k3s) | Container orchestration |
| Cloud | Google Cloud Platform | Production hosting |
| Streaming | Apache Kafka | Real-time transaction processing |

---

## Data Pipeline

### Data Sources

The project uses a synthetic banking dataset modeled on real-world patterns:

| Data Source | Records | Fields | Purpose |
|-------------|---------|--------|---------|
| ATM Transactions | 10,000 | 15 | Withdrawal patterns |
| POS Transactions | 10,000 | 18 | Purchase behavior |
| Digital Transactions | 10,000 | 20 | Online banking |
| Branch Transactions | 10,000 | 16 | In-person activity |
| Customer Master | 10 | 25 | Demographics |
| Account Master | 15 | 12 | Account details |
| ATM Master | 50 | 8 | ATM locations |
| POS Terminal Master | 100 | 10 | Merchant data |
| Beneficiary Master | 30 | 8 | Transfer recipients |

### Star Schema Design

```
                    +------------------+
                    | fact_transactions|
                    +------------------+
                    | transaction_id   |
                    | customer_key  ---+---> dim_customer
                    | account_key   ---+---> dim_account
                    | terminal_key  ---+---> dim_terminal
                    | time_key      ---+---> dim_time
                    | amount           |
                    | is_fraud         |
                    | fraud_type       |
                    +------------------+
```

### ETL Process

1. Load raw Parquet files (4 channels)
2. Standardize column names across channels
3. Add channel identifier
4. Create time features (hour, day, weekend, month)
5. Create amount features (log, normalized, bins)
6. Create velocity features (time since prev, rolling counts)
7. Load to MySQL star schema
8. Create analysis-specific views

---

## Deep Learning Pipeline

### Training Infrastructure

| Parameter | Value |
|-----------|-------|
| Framework | PyTorch 2.0 |
| Hardware | NVIDIA GPU (local), CPU (cloud) |
| Sequence Length | 10 transactions |
| Batch Size | 64 |
| Learning Rate | 0.001 (Adam) |
| Scheduler | ReduceLROnPlateau |
| Early Stopping | Patience = 10 epochs |
| Class Imbalance | WeightedRandomSampler + FocalLoss |

### Model Architectures Trained

#### 1. LSTM (Long Short-Term Memory)
```
Input (10 x 19) -> LSTM(128, 2 layers) -> Dropout(0.3) -> Dense(64) -> Output(1)
```
- Best for: Time series forecasting, Benford's Law analysis
- Strength: Long-range temporal dependencies

#### 2. GRU (Gated Recurrent Unit)
```
Input (10 x 19) -> GRU(128, 2 layers) -> Dropout(0.3) -> Dense(64) -> Output(1)
```
- Best for: Cost-benefit optimization, relational fraud
- Strength: Faster training, fewer parameters

#### 3. AttentionLSTM
```
Input (10 x 19) -> BiLSTM(128) -> MultiHeadAttention(4 heads) -> Dense(64) -> Output(1)
```
- Best for: Account takeover, split transactions, terminal risk
- Strength: Focuses on most relevant transactions in sequence

#### 4. Transformer
```
Input (10 x 19) -> Linear(64) -> TransformerEncoder(2 layers, 4 heads) -> Dense(64) -> Output(1)
```
- Best for: Behavioral change, card testing, cross-channel fraud
- Strength: Self-attention captures complex patterns

#### 5. CNN-LSTM Hybrid
```
Input (10 x 19) -> Conv1D(64) -> MaxPool -> LSTM(128) -> Dense(64) -> Output(1)
```
- Best for: Pattern + sequence detection
- Achieved: **0.5848 AUC** (best advanced model)

#### 6. Autoencoder (Unsupervised)
```
Encoder: Input -> Dense(64) -> Dense(32) -> Latent(16)
Decoder: Latent -> Dense(32) -> Dense(64) -> Reconstruction
```
- Result: 0.4601 AUC (below random)
- Finding: Fraud doesn't have distinct reconstruction signature

### Feature Engineering

**19 Features per Transaction:**

| Category | Features |
|----------|----------|
| Amount | log_amount, amount_normalized, is_high_amount, is_low_amount, is_round_amount |
| Time | hour_sin, hour_cos, dow_sin, dow_cos, is_weekend |
| Channel | channel_ATM, channel_POS, channel_Digital, channel_Branch |
| Velocity | time_since_prev, amount_diff, txn_count_3, amount_sum_3, channel_changed |

### Handling Class Imbalance

The dataset has 1:104 class imbalance (0.95% fraud). Solutions implemented:

- **WeightedRandomSampler**: Oversample minority class during training
- **Focal Loss** (gamma=2): Down-weight easy examples, focus on hard ones
- **Class Weights**: Apply inverse frequency weights in loss function

---

## Model Training Results

### 17 Analysis Categories

| Category | Best Model | AUC | Key Pattern Detected |
|----------|------------|-----|----------------------|
| cost_benefit | **GRU** | **0.6036** | Business ROI optimization |
| cnn_lstm_hybrid | CNN-LSTM | 0.5848 | Combined pattern + sequence |
| relational | GRU | 0.5621 | Customer relationships |
| split_transaction | AttentionLSTM | 0.5589 | Structuring to avoid limits |
| transaction_sequences | AttentionLSTM | 0.5573 | Transaction order anomalies |
| dormant_account | GRU | 0.5573 | Sudden account activation |
| terminal_risk | AttentionLSTM | 0.5562 | High-risk ATM/POS patterns |
| time_series_forecast | LSTM | 0.5467 | Temporal trend prediction |
| account_takeover | AttentionLSTM | 0.5423 | Behavior change detection |
| ensemble_stacking | GradientBoosting | 0.5415 | Meta-learner combination |
| cross_channel | Transformer | 0.5360 | Multi-channel patterns |
| behavioral_change | Transformer | 0.5250 | Spending pattern shifts |
| amount_psychology | LSTM | 0.5505 | Round number preferences |
| benfords_law | LSTM | 0.5037 | First-digit distribution |
| velocity_patterns | LSTM | 0.4865 | Transaction speed |
| card_testing | Transformer | 0.5035 | Small probing transactions |
| autoencoder | Autoencoder | 0.4601 | Reconstruction anomaly |
| vae | VAE | 0.4124 | Probabilistic anomaly |
| basic_sequence | LSTM | 0.5234 | Basic sequence patterns |
| fraud_type_classifier | BiLSTM | 0.9330* | Multi-class fraud type |

*Multi-class accuracy, not AUC

### Architecture Win Distribution

| Architecture | Categories Won | Best Use Cases |
|--------------|----------------|----------------|
| Transformer | 6 | Behavioral, cross-channel, card testing |
| GRU | 4 | Cost-benefit, relational, dormant |
| AttentionLSTM | 4 | Account takeover, split transaction, terminal |
| LSTM | 4 | Time series, amount psychology, Benford's |
| CNN-LSTM | 1 | Hybrid pattern detection |


---

## Key Findings & Insights

### Finding 1: Binary Classification Fundamentally Fails

**Why**: Fraudsters intentionally mimic normal behavior. The statistical distributions overlap completely.

```
Fraud Amount Distribution:  Mean=$2,942, Std=$1,823
Normal Amount Distribution: Mean=$2,804, Std=$1,756
                           ^^^^^^^^^^^^^^^^^^^^^^^^
                           Almost identical!
```

### Finding 2: Multi-Class Classification is the Solution

**Why**: Each fraud type has distinct patterns that get averaged out in binary classification.

```
Binary: 11 fraud types -> Average -> 1 pattern -> No distinguishing features
Multi:  11 fraud types -> 11 patterns -> Each type detectable (86-100%)
```

### Finding 3: Unsupervised Methods Don't Work Here

Autoencoder (0.46 AUC) and VAE (0.41 AUC) performed **worse than random**.

**Why**: Fraud transactions reconstruct just as well as normal ones because they're designed to mimic normal behavior perfectly.

### Finding 4: Attention Mechanisms Capture Critical Patterns

AttentionLSTM excels at detecting:
- **Split transactions**: Identifies the specific transaction that triggers limit avoidance
- **Account takeover**: Focuses on the behavioral change moment
- **Terminal risk**: Highlights suspicious terminal usage within sequence

### Finding 5: Prevention Actions Must Be Fraud-Type-Specific

| Detected Fraud Type | Prevention Action |
|---------------------|-------------------|
| LimitAbuse | Auto-decline within 10% of daily limit |
| RapidWithdrawals | Velocity cooldown (30 min after 3 ATM) |
| CrossBorder | OTP required for international ATM |
| SimSwap | 24-hour transaction block after SIM change |
| CredentialTheft | Force password reset + MFA |
| IdentityTheft | Branch manager alert for enhanced KYC |

### Finding 6: Cost-Benefit Analysis Reveals Optimal Threshold

| Threshold | Detection Rate | False Positive Rate | Net Benefit |
|-----------|----------------|---------------------|-------------|
| 0.10 | 15.8% | 7.3% | -$85,000 |
| 0.20 | 10.5% | 2.1% | +$12,000 |
| **0.30** | **5.3%** | **0.4%** | **+$45,000** |
| 0.40 | 2.6% | 0.1% | +$28,000 |

**Optimal**: Threshold 0.30 maximizes net benefit.

---

## Dashboard & API

### Dashboard Pages

| Page | Description | Key Visualizations |
|------|-------------|-------------------|
| **Overview** | System status, alerts, pending reviews | Alert count, risk distribution |
| **Predictive AI** | All 68 model results | AUC comparison, feature importance |
| **Fraud Patterns** | Sequence signatures, amount psychology | Pattern heatmaps, trend charts |
| **Risk Analysis** | Terminal, account, churn risk | Risk score distributions |
| **Network Analysis** | Cross-account patterns, beneficiary networks | Network graphs |
| **Business Impact** | Cost-benefit ROI, threshold optimization | ROI curves, breakeven analysis |
| **Adaptive Detection** | Emerging threats, multi-pattern analysis | Anomaly timelines |
| **Customer Fraud Profile** | Fraud types by customer heatmap | Customer risk matrix |
| **Coordinated Fraud** | Simultaneous, multi-channel attacks | Attack pattern visualization |

### API Endpoints

```
GET  /health                         - API health check
POST /predict                        - Single transaction prediction
POST /predict/batch                  - Batch predictions
GET  /analytics/dl-models            - All 68 model results
GET  /analytics/fraud-type-classifier - Multi-class detection rates
GET  /analytics/cost-benefit         - ROI analysis
GET  /alerts                         - Current fraud alerts
GET  /reviews/pending                - Transactions needing review
```

---

## Production Deployment

### GCP Infrastructure

| Resource | Configuration | Purpose |
|----------|---------------|---------|
| VM Instance | e2-medium (1-2 vCPU, 4GB RAM) | Main compute |
| Region | us-central1-c | Low latency |
| OS | Debian 12 (Bookworm) | Stability |
| Orchestration | k3s v1.34.3 | Lightweight Kubernetes |
| Storage | 30GB SSD | Database + models |
| Cost | ~$25/month | Within free tier credits |

### Kubernetes Deployment

- **Namespace**: defender
- **Deployments**: dashboard, api, mysql
- **Services**: dashboard (NodePort:30501), api (ClusterIP:8000), mysql (ClusterIP:3306)

### It's Actually Running

Not a demo. Not localhost. Real GCP VM with 6 Docker containers, health checks, and live endpoints.

### Automated Startup Notifications

VM boots? I get an email. Dashboard URL included. Zero manual checking.

---

## Technologies Used

### Machine Learning & Data

| Technology | Version | Purpose |
|------------|---------|---------|
| PyTorch | 2.0+ | Deep learning framework |
| Scikit-learn | 1.3+ | Traditional ML, preprocessing |
| Pandas | 2.0+ | Data manipulation |
| NumPy | 1.24+ | Numerical computing |
| SHAP | 0.42+ | Model explainability |

### Backend & API

| Technology | Version | Purpose |
|------------|---------|---------|
| FastAPI | 0.100+ | REST API framework |
| Uvicorn | 0.23+ | ASGI server |
| SQLAlchemy | 2.0+ | ORM |
| PyMySQL | 1.1+ | MySQL driver |

### Frontend & Visualization

| Technology | Version | Purpose |
|------------|---------|---------|
| Streamlit | 1.28+ | Dashboard framework |
| Plotly | 5.18+ | Interactive charts |
| Seaborn | 0.13+ | Statistical visualization |
| Matplotlib | 3.8+ | Base plotting |

### Infrastructure

| Technology | Version | Purpose |
|------------|---------|---------|
| Docker | 24+ | Containerization |
| Kubernetes (k3s) | 1.34+ | Orchestration |
| MySQL | 8.0 | Database |
| Apache Kafka | 3.6+ | Streaming (optional) |

### Cloud & DevOps

| Technology | Purpose |
|------------|---------|
| Google Cloud Platform | Cloud hosting |
| GitHub | Version control |
| GitHub Actions | CI/CD (planned) |

---

## Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| **1:104 class imbalance** | Combined WeightedRandomSampler + FocalLoss + stratified splitting |
| **Binary models failing** | Reframed as multi-class classification (the key breakthrough) |
| **68 models to manage** | Built unified training pipeline with consistent evaluation |
| **Cloud deployment constraints** | Used lightweight k3s instead of full Kubernetes |
| **Large model files** | Optimized model saving, used CPU inference in production |

---

## Lessons Learned

### 1. Problem Framing > Model Architecture

The biggest performance gains came not from trying more complex models, but from **reframing the problem** from binary to multi-class classification. A simple BiLSTM with the right problem framing (93% detection) beat a 68-model ensemble with the wrong framing (2.9% detection).

### 2. Domain Knowledge is Essential

Understanding that fraudsters intentionally mimic normal behavior led directly to the multi-class solution. Pure ML experimentation without domain understanding would have missed this insight.

### 3. Class Imbalance Requires Multiple Solutions

No single technique solved 1:104 imbalance. The combination of:
- WeightedRandomSampler
- FocalLoss
- Stratified train/test split

Together achieved stable training.

### 4. Explainability Builds Trust

SHAP values and feature importance rankings helped validate that models were learning sensible patterns (amount features, time features) rather than spurious correlations.

### 5. Start Simple, Add Complexity Incrementally

- Started with LSTM baseline
- Added attention mechanisms
- Experimented with Transformers
- Combined into ensembles

Each step was validated before moving to the next.

### 6. Production Differs from Development

Local development with GPU is fast. Production on cloud VMs requires:
- Smaller batch sizes
- Model optimization
- Proper resource limits
- Health checks and monitoring

---

## Future Enhancements

### Planned Improvements

| Enhancement | Priority | Complexity | Impact |
|-------------|----------|------------|--------|
| Real-time Kafka streaming | High | Medium | Live fraud detection |
| A/B testing framework | High | Low | Model comparison |
| Automated retraining pipeline | Medium | Medium | Model freshness |
| Graph neural networks | Medium | High | Relationship fraud |
| Federated learning | Low | High | Privacy-preserving |

### Roadmap

**Phase 1 (Current)**: Production deployment with batch predictions

**Phase 2**: Real-time streaming with Kafka integration

**Phase 3**: AutoML for continuous model optimization

**Phase 4**: Multi-tenant SaaS platform

---

## Project Structure

```
defender/
├── dashboard/              # Streamlit app (9 pages)
│   ├── app.py             # Main dashboard
│   ├── streamlit_app.py   # Cloud entry point
│   ├── utils.py           # Helper functions
│   └── pages/             # Dashboard pages
├── src/
│   ├── api/               # FastAPI backend
│   │   ├── main.py        # API entry point
│   │   ├── routes/        # API endpoints
│   │   └── services/      # Business logic
│   ├── models/            # PyTorch model definitions
│   └── streaming/         # Kafka pipeline
├── notebooks/             # 13 Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── ...
│   └── 13_adaptive_detection.ipynb
├── models/                # Saved model files (.pt, .pkl)
├── k8s/                   # Kubernetes manifests
├── data/                  # Raw and processed data
├── scripts/               # Training and ETL scripts
├── tests/                 # Test suite
├── docker-compose.yml     # Local Docker stack
├── Dockerfile             # API container
├── Dockerfile.dashboard   # Dashboard container
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## Author

**Angelo Gustilo**

---

*Built with PyTorch, FastAPI, Streamlit, and deployed on Google Cloud Platform.*
