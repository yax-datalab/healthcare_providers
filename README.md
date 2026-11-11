# Healthcare Provider Fraud Detection: Exploratory Data Analysis

An exploratory analysis of how healthcare providers flagged as potentially fraudulent differ from non-flagged providers in their billing, operational and clinical patterns. The analysis is descriptive: it does not build predictive models (see [Next steps](#next-steps)).

**Notebook:** [`fraud_detection.ipynb`](fraud_detection.ipynb) (a static export is available in [`fraud_detection.html`](fraud_detection.html))

## Business questions

1. **Financial patterns:** How do reimbursed amounts and reimbursement per beneficiary differ between flagged and non-flagged providers?
2. **Operational patterns:** Is there evidence of over-utilization, such as longer hospital stays or more claims per beneficiary?
3. **Patient and diagnostic complexity:** Do flagged providers treat more complex patients (more diagnoses per claim), or are some diagnosis codes over-represented in their claims?
4. **Concentration and variety:** Is billing more concentrated or more diverse across diagnoses and attending physicians, once provider size is taken into account?

## Data

[Healthcare Provider Fraud Detection Analysis](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis) (Kaggle). Only the labeled training files are used:

| Source | Rows | Description |
|---|---|---|
| Inpatient | 40,474 | One row per claim for admitted patients |
| Outpatient | 517,737 | One row per claim for non-admitted patients |
| Beneficiary | 138,556 | Demographics, chronic conditions, annual amounts |
| Provider | 5,410 | Provider-level fraud label (flagged / non-flagged) |

The label applies to providers, not to individual claims, and indicates *potential* fraud, not confirmed fraud.

## Approach

- **Data integration:** inpatient and outpatient claims are stacked, then left-joined to beneficiary data and the provider label (joins validated as many-to-one).
- **Unit of analysis:** the provider. Flagged providers are a minority (9.35%) but much larger than the rest, so comparing individual claims would mix provider size with billing behavior.
- **Claim types analyzed separately:** reimbursement scales and available variables differ between inpatient and outpatient.
- **Robust summaries:** provider-level medians (providers with very few claims are excluded from the median-based comparisons).
- **Size-aware comparison:** provider-level features are compared within three provider-size ranges using the AUC (probability that a flagged provider has a higher value than a non-flagged one of similar size).
- **Diagnosis codes:** lift of each primary diagnosis in flagged vs. non-flagged claims, ranked by the lower bound of its 95% confidence interval.

## Key findings

- Flagged providers are 9.35% of providers but submit about five times as many claims each, and account for 57.8% of inpatient and 36.6% of outpatient claims.
- Typical (median) claim amounts are nearly identical between groups.
- Within comparable size ranges, flagged **inpatient** providers show higher mean reimbursement and reimbursement per beneficiary (AUC 0.62–0.76).
- Flagged **outpatient** providers depend less on a single attending physician (AUC 0.27–0.35 for the top physician's share of claims) and show no difference in amounts.
- Diagnoses per claim do not differ in either claim type.
- No inpatient diagnosis code stands out; a few outpatient codes are over-represented among flagged providers and are candidates for review.

These are moderate, descriptive associations, not evidence of fraudulent intent. Measures tied to provider size (number of physicians, diversity indices) may still partly reflect volume rather than billing behavior.

## Limitations

- `Provider_flag` is a provider-level label of potential fraud, not confirmed fraud.
- Differences are associations, not evidence of causality or intent.
- Medians use only providers with at least as many claims as the 25th percentile of their claim type; the feature comparisons by size range use all providers with at least 10 per class in each range.
- Size control is approximate (three ranges), and AUC values are reported without confidence intervals or significance tests.
- Diagnosis-code lift is not corrected for multiple comparisons.
- Beneficiary annual reimbursement columns contain negative values, which suggest adjustments; they are not used in the provider-level analysis.

## Next steps

A supervised model to predict `Provider_flag` from provider-level features, prioritizing ratio-type features (reimbursement per beneficiary, claims per beneficiary, diagnostic diversity, physician concentration) and treating size variables explicitly. The class imbalance calls for stratified splits at the provider level, class weights or tree-based models, and evaluation with recall and precision–recall metrics, since missed fraud is the costly error.

## Repository structure

```
├── fraud_detection.ipynb   # Analysis notebook
├── fraud_detection.html    # Static export of the notebook
├── utils.py                # Helper functions used by the notebook
├── requirements.txt        # Python dependencies
└── README.md
```

## How to run

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook fraud_detection.ipynb
```

The notebook downloads the dataset with [`kagglehub`](https://github.com/Kaggle/kagglehub). If prompted, authenticate with your Kaggle account (see the kagglehub documentation).
