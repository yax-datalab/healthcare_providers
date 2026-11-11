import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
###################          Load kagle data   ############################
def load_subset(pattern, files_path):
    file = next(files_path.glob(pattern))
    return pd.read_csv(file)
    
###################         Compare metrics    ############################   
def compare_metric(df, metric, group_col="Provider_flag", n_boot=1000):
    results = {}
    for grp in ["Non-flagged", "Flagged"]:
        vals = df.loc[df[group_col] == grp, metric].dropna().values
        boots = [np.mean(np.random.choice(vals, len(vals), replace=True))
                 for _ in range(n_boot)]
        results[grp] = {
            "mean":   np.mean(vals),
            "median": np.median(vals),
            "ci_low": np.percentile(boots, 2.5),
            "ci_high":np.percentile(boots, 97.5),
            "n":      len(vals)
        }
    # Mann-Whitney (no asume normalidad)
    no_vals  = df.loc[df[group_col] == "Non-flagged",  metric].dropna()
    yes_vals = df.loc[df[group_col] == "Flagged", metric].dropna()
    stat, p = mannwhitneyu(no_vals, yes_vals, alternative="two-sided")
    return results, p


################     Plot histograms and violin function         ###################
def plot_hist_violin(
    data,ax_hist,ax_violin,x,
    title_hist=None,title_violin=None,
    xlabel_hist=None,ylabel_hist=None,
    xlabel_violin=None, ylabel_violin=None,
    hue="Provider_flag",order=("Flagged", "Non-flagged"),
):
    sns.histplot(
        data=data,x=x,hue=hue,hue_order=order,
        bins=30,ax=ax_hist,stat="probability",
        common_norm=False,
        )
    
    sns.violinplot(
        data=data, x=hue, y=x, hue=hue, 
        order=order, hue_order=order, ax=ax_violin,
        )

    for ax, opts in (
        (ax_hist,   dict(title=title_hist,   xlabel=xlabel_hist,   ylabel=ylabel_hist)),
        (ax_violin, dict(title=title_violin, xlabel=xlabel_violin, ylabel=ylabel_violin)),
    ):
        ax.set(**{k: v for k, v in opts.items() if v is not None})
################     Balance samples function         ###################
def balance_samples(df, group_col="Provider_flag", random_state=111):
    n_min = df[group_col].value_counts().min()
    return (df.groupby(group_col, group_keys=True)
              .apply(lambda x: x.sample(n_min, random_state=random_state))
              .reset_index(drop=True))

################     Set plots theme and aesthetics         ###################
def plot_settings():
    sns.set_theme(style="whitegrid",
                  palette="colorblind",
                  context="notebook")
    
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.titlelocation":"left",
        "figure.subplot.hspace": 0.4})

################     Plot labels dictionary        ###################
plot_lbls = {
    "InscClaimAmtReimbursed":"Reimbursed Amount($)",
    "OPAnnualReimbursementAmt":"Annual Outpatient Reimbursement ($)",
    "OPAnnualDeductibleAmt":"Annual Outpatient Deductible ($)",
    "IPAnnualReimbursementAmt":"Annual Inpatient Reimbursement ($)",
    "IPAnnualDeductibleAmt":"Annual Inpatient Deductible ($)",
    "days_in_hospital":"Hospital Stay (days)", 
    "claim_days":"Claim Duration (days)", 
    "daily_cost":"Daily Reimbursed Cost ($/Day)",
    "n_chronic_cond":"Chronic Conditions per Beneficiary", 
    "n_claim_dx":"Diagnosis Codes per Claim",
    "median_reimbursed":"Median Reimbursed Amount ($)",
    "median_days_in_hospital":"Median Hospital Stay (median)",
    "median_claim_days": "Median Claim Duration (Days)",
    "median_daily_cost": "Median Daily Reimbursed Cost ($/Day)",
    "median_IPAnnualReimbursementAmt": "Median Annual Inpatient Reimbursement ($)",
    "median_OPAnnualReimbursementAmt": "Median Annual Outpatient Reimbursement ($)",
    "median_InscClaimAmtReimbursed": "Median Reimbursed Claim Amount ($)",
    "median_IPAnnualDeductibleAmt": "Median Annual Inpatient Deductible ($)",
    "median_OPAnnualDeductibleAmt": "Median Annual Outpatient Deductible ($)",
    "median_DeductibleAmtPaid": "Median Deductible Paid per Claim ($)",
    "median_n_claim_dx": "Mean Diagnosis Codes per Claim",
    "n_bene": "Beneficiaries",
    "n_claims": "Total Claims",
    "tot_claims": "Claims",
    "n_prov": "Providers",
    "mean_n_dx": "Diagnoses per claim",
    "total_claims": "Total claims",
    "n_bene": "Beneficiaries",
    "reimb_amount_sd": "Reimbursement SD",
    "reimb_amount_mean": "Mean reimbursement",
    "reimb_amount_tot": "Total reimbursement",
    "n_attending": "Distinct attending physicians",
    "top_attphy_share": "Top physician's share of claims",
    "top_dx_share": "Top diagnosis' share of claims",
    "attphy_entropy": "Physician diversity",
    "dx_entropy": "Diagnosis diversity",
    "cv_reimb_amount": "Reimbursement CV",
    "reimb_per_bene": "Reimbursement per beneficiary",
    "n_claims_per_bene": "Claims per beneficiary",
}
####################              Distribution stats    ###################33
def distribution_shape(df):
    distr_shape = np.select(
        [
            df["nunique"] <= 1,       
            df["skew"].abs() > 1,
            df["kurt"].abs() > 3,
            ],
        ["n.a.", "skewed", "heavy-tailed"],
        default="approx. symmetric"
    )
    return(distr_shape)

#####################         Stratified AUC         ########################
def stratified_auc(df, feats, size_col, n_bins, min_per_class):
    df = df.copy()
    df["size_bin"] = pd.qcut(df[size_col], n_bins, labels=False, duplicates="drop")
    rows = []
    for b, d in df.groupby("size_bin"):
        for c in feats:
            y = d.loc[d.Provider_flag == "Flagged", c].dropna()
            n = d.loc[d.Provider_flag == "Non-flagged", c].dropna()
            if len(y) < min_per_class or len(n) < min_per_class:
                continue
            u, _ = mannwhitneyu(y, n)
            rows.append({"size_bin": b, "feature": c,
                         "auc": u / (len(y) * len(n)), "n_flagged": len(y), "n_nonflagged": len(n)})
    return pd.DataFrame(rows)

#####################         Topshare         ########################
def topshare(d):
    vcounts=d.value_counts(normalize=True)
    return vcounts.iloc[0] if len(vcounts) else np.nan
    
#####################         Sharonns entropy         ########################
def shanentropy(df):
    p=df.value_counts(normalize=True)
    return stats.entropy(p, base=2) if len(p) else np.nan
####################### Compare AUCs         #################################
def compare(df, feats):
    rows = []
    for c in feats:
        y = df.loc[df.PotentialFraud == "Flagged", c].dropna()
        n = df.loc[df.PotentialFraud == "Non-flagged", c].dropna()
        u, p = mannwhitneyu(y, n)
        rows.append({"feature": c, "auc": u / (len(y) * len(n)), "p": p,
                     "median_flagged": y.median(), "median_nonflagged": n.median()})
    out = pd.DataFrame(rows)
    out["q"] = multipletests(out["p"], method="fdr_bh")[1]
    return out.assign(dist_from_05=lambda d: (d.auc - 0.5).abs()).sort_values("dist_from_05", ascending=False)

