# Problem Statement: Why Standard Fraud Models Fail Cross-Border Payments

## Executive Summary

Cross-border payment fraud detection presents unique challenges that standard machine learning approaches fail to address. This document outlines the fundamental problem, its business impact, and the rationale for a context-aware solution.

## The Core Problem

### Behavioural Heterogeneity Across Corridors

International payments exhibit corridor-specific patterns that violate the assumptions of traditional fraud models:

**Transaction Size Distributions**

Consider two common UK remittance corridors:

| Corridor | Median Transaction | 95th Percentile | Typical Use Case |
|----------|-------------------|-----------------|------------------|
| UK → Nigeria | £350 | £2,500 | Family support, school fees |
| UK → Poland | £180 | £800 | Worker remittances, monthly transfers |
| UK → India | £500 | £5,000 | Property payments, family events |

A £2,000 transaction is unremarkable for UK→Nigeria but highly unusual for UK→Poland. Global thresholds cannot capture this nuance.

**Velocity Patterns**

Legitimate transaction velocity varies dramatically:

- **UK → Philippines**: Often clustered around month-end (domestic worker payment cycles)
- **UK → Pakistan**: Spikes during Eid periods (5-10x normal volume)
- **UK → EU**: More evenly distributed throughout the month

A sudden increase in transaction frequency might indicate fraud in one corridor but is entirely normal in another during specific periods.

**Sender Behaviour Profiles**

Different corridors attract different sender demographics:

- Average account age before first transaction
- Typical number of beneficiaries per sender
- Device and channel preferences
- Time-of-day patterns

### The False Positive Problem

When models trained on global data encounter corridor-specific legitimate behaviour, they generate false positives at unacceptable rates.

**Business Impact of False Positives:**

1. **Customer Friction**: Blocked legitimate transactions require manual review, delaying time-sensitive transfers (medical emergencies, school fee deadlines)

2. **Customer Attrition**: Repeated false blocks drive customers to competitors; acquiring a new remittance customer costs 5-7x retention

3. **Operational Cost**: Each false positive requires analyst review (15-30 minutes average), creating unsustainable operational burden at scale

4. **Regulatory Risk**: Excessive blocking of specific corridors may constitute discriminatory practice under financial services regulations

### The Missed Fraud Problem

Conversely, fraud patterns that are distinctive within a corridor may appear normal globally.

**Example**: Account takeover in UK→Nigeria corridor

- Fraudster gains access to established account
- Initiates £400 transfer (below global alert threshold)
- Transaction size is normal for corridor
- But: new beneficiary + unusual time + device change = strong corridor-specific signal

Global models miss this because £400 with a new beneficiary isn't inherently suspicious. Corridor-aware models recognise the pattern.

## Why Standard Approaches Fall Short

### Approach 1: Global Threshold Models

**Method**: Single set of rules/thresholds applied uniformly

**Failure Mode**: Cannot balance precision across corridors with different baseline behaviours

**Result**: Either too many false positives in high-value corridors or missed fraud in low-value corridors

### Approach 2: Corridor-Specific Models

**Method**: Train separate models for each origin-destination pair

**Failure Mode**: Data sparsity in low-volume corridors; maintenance burden of N×M models

**Result**: Overfitting in sparse corridors; operational complexity

### Approach 3: Corridor as a Feature

**Method**: Include corridor identifier as input feature to global model

**Failure Mode**: Model learns average effect, not interaction effects with other signals

**Result**: Modest improvement but still misses corridor-specific signal combinations

## The Opportunity

A dynamic weighting system offers a middle path:

- **Single model architecture** (maintainable)
- **Corridor-specific parameters** (accurate)
- **Continuous learning** from new corridor data (adaptive)

The key insight is that the *signals* for fraud are largely universal (velocity spikes, new beneficiaries, unusual amounts), but their *weights* should vary by context.

## Success Criteria

Any solution must achieve:

| Metric | Target | Rationale |
|--------|--------|-----------|
| Recall | ≥90% | Regulatory expectation; fraud must be caught |
| Precision | Improve by ≥15% | Operational sustainability |
| Fraud Loss Reduction | ≥10% | Business case justification |
| Review Volume | Reduce by ≥20% | Analyst capacity constraint |

## Constraints

The solution must operate within:

1. **Latency**: Fraud scoring must complete in <200ms for real-time decisioning
2. **Explainability**: Regulatory requirement to explain why transactions were blocked
3. **Fairness**: Cannot systematically disadvantage legitimate users of specific corridors
4. **Data Privacy**: Must work with aggregated corridor statistics, not individual customer profiles from other institutions

## Next Steps

The following documents detail:

- `methodology.md`: Technical approach to dynamic signal weighting
- `results.md`: Production performance and lessons learned

---

*This problem statement reflects patterns observed in production cross-border payment systems. Specific figures are illustrative.*
