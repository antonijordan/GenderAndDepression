"""
================================================================================
  Linguistic Markers of Depression — Full Analysis Pipeline
================================================================================
  Research Questions:
    RQ1: Do depressed and non-depressed people differ in marker use?
    RQ2: Do depressed men and women differ in marker use?

  Sections:
    0. Configuration
    1. Load & Inspect Data
    2. Descriptive Statistics
    3. Visualisations
    4. RQ1 — t-tests / Mann-Whitney U + FDR correction + effect sizes
    5. RQ2 — MANOVA + follow-up ANOVAs (depressed participants only)
    6. Logistic Regression (per gender + full sample)
    7. Radar Chart (group profiles)
================================================================================
"""

# ── 0. CONFIGURATION ──────────────────────────────────────────────────────────

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, shapiro
from statsmodels.stats.multitest import multipletests
from statsmodels.multivariate.manova import MANOVA
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import statsmodels.api as sm
import os

# ── File paths ──
DATA_FILE  = "X"
OUTPUT_DIR = "X"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Marker columns ──
MARKERS = [
    "pronoun", "ppron", "i", "we", "you", "shehe", "they", "ipron",
    "tone_pos", "tone_neg", "emo_pos", "emo_neg",
    "emo_anx", "emo_anger", "emo_sad",
    "Absolutist"
]

MARKER_LABELS = {
    "pronoun":   "Pronouns (total)",
    "ppron":     "Personal pronouns",
    "i":         "1st person singular (I)",
    "we":        "1st person plural (we)",
    "you":       "2nd person (you)",
    "shehe":     "3rd person singular (she/he)",
    "they":      "3rd person plural (they)",
    "ipron":     "Impersonal pronouns",
    "tone_pos":  "Positive tone",
    "tone_neg":  "Negative tone",
    "emo_pos":   "Positive emotion",
    "emo_neg":   "Negative emotion",
    "emo_anx":   "Anxiety",
    "emo_anger": "Anger",
    "emo_sad":   "Sadness",
    "Absolutist":"Absolutist words",
}

GROUP_COL  = "PHQ8_Binary"  # 0 = not depressed, 1 = depressed
GENDER_COL = "Gender"       # 0 = female, 1 = male

# ── Plotting style ──
plt.rcParams.update({
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "sans-serif",
})
PALETTE        = {0: "#4C9BE8", 1: "#E8624C"}
GENDER_PALETTE = {0: "#C2678D", 1: "#5B8DB8"}


# ── 1. LOAD & INSPECT DATA ────────────────────────────────────────────────────

df = pd.read_csv(DATA_FILE)

df.replace("n/a", np.nan, inplace=True)
df.dropna(subset=[GROUP_COL, GENDER_COL], inplace=True)

for col in MARKERS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[GROUP_COL]  = pd.to_numeric(df[GROUP_COL],  errors="coerce").astype(int)
df[GENDER_COL] = pd.to_numeric(df[GENDER_COL], errors="coerce").astype(int)

print(f"\nDataset shape:  {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nDepression group counts:\n{df[GROUP_COL].value_counts().rename({0:'Not depressed', 1:'Depressed'})}")
print(f"\nGender counts:\n{df[GENDER_COL].value_counts().rename({0:'Female', 1:'Male'})}")
print(f"\nGroup x Gender:\n{pd.crosstab(df[GENDER_COL].map({0:'Female',1:'Male'}), df[GROUP_COL].map({0:'Not depressed', 1:'Depressed'}))}")


# ── 2. DESCRIPTIVE STATISTICS ─────────────────────────────────────────────────

desc = df.groupby(GROUP_COL)[MARKERS].agg(["mean", "std"]).round(3)
desc.index = ["Not depressed", "Depressed"]
print("\nMean +/- SD by depression status:")
print(desc.to_string())

desc_gender = df.groupby([GENDER_COL, GROUP_COL])[MARKERS].mean().round(3)
desc_gender.index = [
    f"{'Female' if g==0 else 'Male'} - {'Not dep' if d==0 else 'Dep'}"
    for g, d in desc_gender.index
]
print("\nMeans by Gender x Depression:")
print(desc_gender.to_string())

# Depressed only descriptives for RQ2
desc_dep = df[df[GROUP_COL] == 1].groupby(GENDER_COL)[MARKERS].agg(["mean", "std"]).round(3)
desc_dep.index = ["Female (depressed)", "Male (depressed)"]
print("\nMean +/- SD — Depressed participants only (RQ2):")
print(desc_dep.to_string())

desc.to_csv(f"{OUTPUT_DIR}/descriptives_depression.csv")
desc_gender.to_csv(f"{OUTPUT_DIR}/descriptives_gender_depression.csv")
desc_dep.to_csv(f"{OUTPUT_DIR}/descriptives_depressed_only.csv")
print(f"\n✓ Descriptives saved to {OUTPUT_DIR}/")


# ── 3. VISUALISATIONS ─────────────────────────────────────────────────────────

n_cols = 4
n_rows = int(np.ceil(len(MARKERS) / n_cols))

## 3a. Violin plots — depressed vs. not depressed
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 3.5))
axes = axes.flatten()

for idx, marker in enumerate(MARKERS):
    ax = axes[idx]
    data_plot = [
        df[df[GROUP_COL] == 0][marker].dropna(),
        df[df[GROUP_COL] == 1][marker].dropna(),
    ]
    valid = [(i, d) for i, d in enumerate(data_plot) if len(d) > 0]
    if not valid:
        continue
    positions, valid_data = zip(*valid)
    parts = ax.violinplot(valid_data, positions=list(positions), showmedians=True)
    for pc, color in zip(parts["bodies"], [PALETTE[0], PALETTE[1]]):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Not\nDepressed", "Depressed"], fontsize=9)
    ax.set_title(MARKER_LABELS.get(marker, marker), fontsize=10, fontweight="bold")
    ax.set_ylabel("Score", fontsize=8)

for i in range(len(MARKERS), len(axes)):
    axes[i].set_visible(False)

fig.suptitle("Linguistic Markers: Depressed vs. Not Depressed", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/violin_depression.png", bbox_inches="tight")
plt.close()
print("✓ Saved: violin_depression.png")

## 3b. Violin plots — 4 groups (gender x depression)
df["Group"] = df.apply(
    lambda r: f"{'Male' if r[GENDER_COL]==1 else 'Female'} - {'Dep' if r[GROUP_COL]==1 else 'Non-dep'}",
    axis=1
)
GROUP_ORDER  = ["Female - Non-dep", "Female - Dep", "Male - Non-dep", "Male - Dep"]
GROUP_COLORS = ["#C2678D", "#8B2252", "#5B8DB8", "#1E4D7B"]

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3.5))
axes = axes.flatten()

for idx, marker in enumerate(MARKERS):
    ax = axes[idx]
    data_plot = [df[df["Group"] == g][marker].dropna() for g in GROUP_ORDER]
    valid = [(i, d) for i, d in enumerate(data_plot) if len(d) > 0]
    if not valid:
        continue
    positions, valid_data = zip(*valid)
    parts = ax.violinplot(valid_data, positions=list(positions), showmedians=True)
    for pc, color in zip(parts["bodies"], GROUP_COLORS):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["F\nNon-dep", "F\nDep", "M\nNon-dep", "M\nDep"], fontsize=8)
    ax.set_title(MARKER_LABELS.get(marker, marker), fontsize=10, fontweight="bold")

for i in range(len(MARKERS), len(axes)):
    axes[i].set_visible(False)

fig.suptitle("Linguistic Markers by Gender x Depression", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/violin_gender_depression.png", bbox_inches="tight")
plt.close()
print("✓ Saved: violin_gender_depression.png")

## 3c. Violin plots — depressed only, female vs male (RQ2)
df_dep = df[df[GROUP_COL] == 1].copy()

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 3.5))
axes = axes.flatten()

for idx, marker in enumerate(MARKERS):
    ax = axes[idx]
    data_plot = [
        df_dep[df_dep[GENDER_COL] == 0][marker].dropna(),
        df_dep[df_dep[GENDER_COL] == 1][marker].dropna(),
    ]
    valid = [(i, d) for i, d in enumerate(data_plot) if len(d) > 0]
    if not valid:
        continue
    positions, valid_data = zip(*valid)
    parts = ax.violinplot(valid_data, positions=list(positions), showmedians=True)
    for pc, color in zip(parts["bodies"], [GENDER_PALETTE[0], GENDER_PALETTE[1]]):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Female", "Male"], fontsize=9)
    ax.set_title(MARKER_LABELS.get(marker, marker), fontsize=10, fontweight="bold")
    ax.set_ylabel("Score", fontsize=8)

for i in range(len(MARKERS), len(axes)):
    axes[i].set_visible(False)

fig.suptitle("Linguistic Markers: Depressed Women vs. Depressed Men (RQ2)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/violin_depressed_gender.png", bbox_inches="tight")
plt.close()
print("✓ Saved: violin_depressed_gender.png")


# ── 4. RQ1 — GROUP COMPARISONS (t-test / Mann-Whitney) ───────────────────────

def cohens_d(a, b):
    pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return (np.mean(a) - np.mean(b)) / pooled_std if pooled_std != 0 else 0

def normality_ok(x, alpha=0.05):
    if len(x) > 5000:
        return True
    _, p = shapiro(x[:5000])
    return p > alpha

results_rq1 = []

for marker in MARKERS:
    g0 = df[df[GROUP_COL] == 0][marker].dropna()
    g1 = df[df[GROUP_COL] == 1][marker].dropna()

    use_parametric = normality_ok(g0) and normality_ok(g1)

    if use_parametric:
        stat, p = stats.ttest_ind(g0, g1)
        test = "t-test"
    else:
        stat, p = mannwhitneyu(g0, g1, alternative="two-sided")
        test = "Mann-Whitney U"

    d = cohens_d(g1, g0)

    results_rq1.append({
        "Marker":      MARKER_LABELS.get(marker, marker),
        "column":      marker,
        "Test":        test,
        "Statistic":   round(stat, 3),
        "p_raw":       round(p, 4),
        "Cohen_d":     round(d, 3),
        "Mean_NonDep": round(g0.mean(), 3),
        "SD_NonDep":   round(g0.std(), 3),
        "Mean_Dep":    round(g1.mean(), 3),
        "SD_Dep":      round(g1.std(), 3),
    })

rq1_df = pd.DataFrame(results_rq1)

reject_fdr, p_fdr, _, _ = multipletests(rq1_df["p_raw"], method="fdr_bh")
rq1_df["p_fdr"]   = p_fdr.round(4)
rq1_df["sig_fdr"] = reject_fdr

print("\nRQ1 Results (sorted by Cohen's d):")
display_cols = ["Marker", "Test", "Mean_NonDep", "SD_NonDep", "Mean_Dep", "SD_Dep", "p_raw", "p_fdr", "Cohen_d"]
print(rq1_df[display_cols].sort_values("Cohen_d", key=abs, ascending=False).to_string(index=False))

rq1_df.to_csv(f"{OUTPUT_DIR}/RQ1_group_comparisons.csv", index=False)
print(f"\n✓ Saved: RQ1_group_comparisons.csv")

fig, ax = plt.subplots(figsize=(10, 8))
sorted_df = rq1_df.sort_values("Cohen_d")
colors = ["#E8624C" if sig else "#AAAAAA" for sig in sorted_df["sig_fdr"]]
ax.barh(sorted_df["Marker"], sorted_df["Cohen_d"], color=colors, edgecolor="white", height=0.6)
ax.axvline(0,    color="black",   linewidth=0.8, linestyle="--")
ax.axvline(0.2,  color="#cccccc", linewidth=0.6, linestyle=":")
ax.axvline(-0.2, color="#cccccc", linewidth=0.6, linestyle=":")
ax.axvline(0.5,  color="#aaaaaa", linewidth=0.6, linestyle=":")
ax.axvline(-0.5, color="#aaaaaa", linewidth=0.6, linestyle=":")
ax.set_xlabel("Cohen's d  (positive = higher in depressed group)", fontsize=10)
ax.set_title("Effect Sizes: Depressed vs. Non-Depressed\n(red = significant after FDR correction)",
             fontsize=12, fontweight="bold")
sig_patch   = mpatches.Patch(color="#E8624C", label="Significant (FDR)")
insig_patch = mpatches.Patch(color="#AAAAAA", label="Not significant")
ax.legend(handles=[sig_patch, insig_patch], loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/RQ1_effect_sizes.png", bbox_inches="tight")
plt.close()
print("✓ Saved: RQ1_effect_sizes.png")



# ── 5. RQ2 — MANOVA + FOLLOW-UP ANOVAs (DEPRESSED ONLY) ─────────────────────

df_manova = df[(df[GROUP_COL] == 1) & (df[GENDER_COL].isin([0, 1]))].dropna(subset=MARKERS).copy()

print(f"\nDepressed-only subset: {df_manova.shape[0]} participants")
print(f"  Female (0): {(df_manova[GENDER_COL]==0).sum()}")
print(f"  Male   (1): {(df_manova[GENDER_COL]==1).sum()}")

marker_formula = " + ".join([f"Q('{m}')" for m in MARKERS])
manova_formula = f"{marker_formula} ~ C({GENDER_COL})"

try:
    manova = MANOVA.from_formula(manova_formula, data=df_manova)
    result = manova.mv_test()
    print("\nMANOVA Results (depressed only, Gender effect):")
    print(result)
    with open(f"{OUTPUT_DIR}/MANOVA_results.txt", "w") as f:
        f.write(str(result))
    print("✓ Saved: MANOVA_results.txt")
except Exception as e:
    print(f"  ⚠ MANOVA failed: {e}")

print("\nFollow-up ANOVAs (Gender effect, depressed participants only):")
anova_results = []

for marker in MARKERS:
    try:
        formula = f"Q('{marker}') ~ C({GENDER_COL})"
        model   = ols(formula, data=df_manova).fit()
        table   = anova_lm(model, typ=2)
        gen_f   = table.loc[f"C({GENDER_COL})", "F"]
        gen_p   = table.loc[f"C({GENDER_COL})", "PR(>F)"]

        female  = df_manova[df_manova[GENDER_COL] == 0][marker].dropna()
        male    = df_manova[df_manova[GENDER_COL] == 1][marker].dropna()

        d = cohens_d(male, female)

        anova_results.append({
            "Marker":   MARKER_LABELS.get(marker, marker),
            "column":   marker,
            "F":        round(gen_f, 3),
            "p_gender": round(gen_p, 4),
            "Cohen_d":  round(d, 3),
            "Mean_F":   round(female.mean(), 3),
            "SD_F":     round(female.std(), 3),
            "Mean_M":   round(male.mean(), 3),
            "SD_M":     round(male.std(), 3),
        })
        print(f"  ✓ {marker}")
    except Exception as e:
        print(f"  ⚠ ANOVA failed for {marker}: {e}")

anova_df = pd.DataFrame(anova_results)

if not anova_df.empty:
    reject_fdr, p_fdr, _, _ = multipletests(anova_df["p_gender"], method="fdr_bh")
    anova_df["p_fdr"]   = p_fdr.round(4)
    anova_df["sig_fdr"] = reject_fdr

    print("\nRQ2 Results (sorted by p-value):")
    print(anova_df[["Marker", "F", "p_gender", "p_fdr", "sig_fdr",
                    "Cohen_d", "Mean_F", "SD_F", "Mean_M", "SD_M"]
                   ].sort_values("p_gender").to_string(index=False))

    anova_df.to_csv(f"{OUTPUT_DIR}/RQ2_depressed_gender_ANOVAs.csv", index=False)
    print(f"\n✓ Saved: RQ2_depressed_gender_ANOVAs.csv")

    fig, ax = plt.subplots(figsize=(10, 8))
    sorted_anova = anova_df.sort_values("Cohen_d")
    colors = ["#E8624C" if sig else "#AAAAAA" for sig in sorted_anova["sig_fdr"]]
    ax.barh(sorted_anova["Marker"], sorted_anova["Cohen_d"], color=colors, edgecolor="white", height=0.6)
    ax.axvline(0,    color="black",   linewidth=0.8, linestyle="--")
    ax.axvline(0.2,  color="#cccccc", linewidth=0.6, linestyle=":")
    ax.axvline(-0.2, color="#cccccc", linewidth=0.6, linestyle=":")
    ax.axvline(0.5,  color="#aaaaaa", linewidth=0.6, linestyle=":")
    ax.axvline(-0.5, color="#aaaaaa", linewidth=0.6, linestyle=":")
    ax.set_xlabel("Cohen's d  (positive = higher in depressed males)", fontsize=10)
    ax.set_title("Effect Sizes: Depressed Men vs. Depressed Women\n(red = significant after FDR correction)",
                 fontsize=12, fontweight="bold")
    sig_patch   = mpatches.Patch(color="#E8624C", label="Significant (FDR)")
    insig_patch = mpatches.Patch(color="#AAAAAA", label="Not significant")
    ax.legend(handles=[sig_patch, insig_patch], loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/RQ2_effect_sizes.png", bbox_inches="tight")
    plt.close()
    print("✓ Saved: RQ2_effect_sizes.png")

    sig_markers = anova_df[anova_df["sig_fdr"] == True]
    if not sig_markers.empty:
        fig, axes = plt.subplots(1, len(sig_markers), figsize=(5 * len(sig_markers), 5))
        if len(sig_markers) == 1:
            axes = [axes]
        for ax, (_, row) in zip(axes, sig_markers.iterrows()):
            ax.bar(["Female", "Male"], [row["Mean_F"], row["Mean_M"]],
                   color=[GENDER_PALETTE[0], GENDER_PALETTE[1]], width=0.5, edgecolor="white")
            ax.set_title(f"{row['Marker']}\np={row['p_fdr']} (FDR)", fontweight="bold", fontsize=10)
            ax.set_ylabel("Mean score")
        fig.suptitle("Significant Markers: Depressed Women vs. Men", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/RQ2_significant_markers.png", bbox_inches="tight")
        plt.close()
        print("✓ Saved: RQ2_significant_markers.png")
    else:
        print("\nNo significant gender differences found among depressed participants (after FDR correction).")

# ── 6. LOGISTIC REGRESSION ────────────────────────────────────────────────────
# ── LOGISTIC REGRESSION (REDUCED — collinear pronouns removed) ────────────────

# Reduced marker set — keeping 'i' as the theoretically motivated pronoun,
# dropping 'pronoun' and 'ppron' which are aggregates that cause multicollinearity
MARKERS_REDUCED = [
    "i", "we", "you", "shehe", "they", "ipron",
    "tone_pos", "tone_neg", "emo_pos", "emo_neg",
    "emo_anx", "emo_anger", "emo_sad",
    "Absolutist"
]

MARKER_LABELS_REDUCED = {k: v for k, v in MARKER_LABELS.items() if k in MARKERS_REDUCED}

def run_logistic_reduced(data, label):
    X = data[MARKERS_REDUCED].dropna()
    y = data.loc[X.index, GROUP_COL]

    if y.nunique() < 2:
        print(f"\n  ⚠ Skipping {label} — only one class present.")
        return None, None

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model     = LogisticRegression(max_iter=1000, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="roc_auc")
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)

    print(f"\n--- {label} ---")
    print(f"  Cross-validated AUC (5-fold): {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print(f"  Classification report (train):\n{classification_report(y, y_pred, target_names=['Non-dep','Depressed'])}")

    X_sm     = sm.add_constant(pd.DataFrame(X_scaled, columns=MARKERS_REDUCED))
    sm_model = sm.Logit(y.values, X_sm).fit(disp=0)

    coef_df = pd.DataFrame({
        "Marker":  ["intercept"] + MARKERS_REDUCED,
        "Coef":    sm_model.params.values,
        "OR":      np.exp(sm_model.params.values),
        "p":       sm_model.pvalues.values,
        "CI_low":  np.exp(sm_model.conf_int().iloc[:, 0].values),
        "CI_high": np.exp(sm_model.conf_int().iloc[:, 1].values),
    }).iloc[1:]

    coef_df = coef_df.sort_values("OR", ascending=False)
    print(coef_df[["Marker", "OR", "p"]].to_string(index=False))
    coef_df.to_csv(f"{OUTPUT_DIR}/logistic_reduced_{label.replace(' ','_')}.csv", index=False)

    # Coefficient plot
    fig, ax = plt.subplots(figsize=(9, 7))
    y_pos = range(len(coef_df))
    ax.barh(y_pos, np.log(coef_df["OR"]),
            xerr=[np.log(coef_df["OR"]) - np.log(coef_df["CI_low"]),
                  np.log(coef_df["CI_high"]) - np.log(coef_df["OR"])],
            color=["#E8624C" if p < 0.05 else "#AAAAAA" for p in coef_df["p"]],
            align="center", height=0.6, capsize=3)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([MARKER_LABELS_REDUCED.get(m, m) for m in coef_df["Marker"]])
    ax.set_xlabel("Log Odds Ratio (standardised predictors)", fontsize=10)
    ax.set_title(f"Logistic Regression Coefficients (Reduced)\n{label}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fname = f"{OUTPUT_DIR}/logistic_reduced_coefs_{label.replace(' ','_')}.png"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {os.path.basename(fname)}")

    return model, cv_scores

run_logistic_reduced(df,                       "Full Sample")
run_logistic_reduced(df[df[GENDER_COL] == 0], "Females Only")
run_logistic_reduced(df[df[GENDER_COL] == 1], "Males Only")



# ── 7. RADAR CHART ────────────────────────────────────────────────────────────

from sklearn.preprocessing import MinMaxScaler

mm      = MinMaxScaler()
df_norm = df.copy()
df_norm[MARKERS] = mm.fit_transform(df[MARKERS].fillna(df[MARKERS].mean()))

groups = {
    "Female - Depressed":     df_norm[(df_norm[GENDER_COL]==0) & (df_norm[GROUP_COL]==1)],
    "Female - Not Depressed": df_norm[(df_norm[GENDER_COL]==0) & (df_norm[GROUP_COL]==0)],
    "Male - Depressed":       df_norm[(df_norm[GENDER_COL]==1) & (df_norm[GROUP_COL]==1)],
    "Male - Not Depressed":   df_norm[(df_norm[GENDER_COL]==1) & (df_norm[GROUP_COL]==0)],
}
radar_colors = ["#C2678D", "#F2A0C0", "#1E4D7B", "#5B8DB8"]

labels = [MARKER_LABELS.get(m, m) for m in MARKERS]
N      = len(MARKERS)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(11, 11), subplot_kw=dict(polar=True))

for (group_label, group_df), color in zip(groups.items(), radar_colors):
    values  = group_df[MARKERS].mean().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle="solid", label=group_label, color=color)
    ax.fill(angles, values, alpha=0.08, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, size=9)
ax.set_yticklabels([])
ax.set_title("Linguistic Profiles by Gender x Depression\n(normalised 0-1)",
             size=14, fontweight="bold", pad=25)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/radar_chart.png", bbox_inches="tight")
plt.close()
print("✓ Saved: radar_chart.png")


# ── DONE ──────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  ALL STEPS COMPLETE")
print(f"  All outputs saved to: {OUTPUT_DIR}/")
print("=" * 70)
print("""
  Files produced:
  ├── descriptives_depression.csv
  ├── descriptives_gender_depression.csv
  ├── descriptives_depressed_only.csv
  ├── violin_depression.png
  ├── violin_gender_depression.png
  ├── violin_depressed_gender.png         (RQ2 - depressed only)
  ├── RQ1_group_comparisons.csv
  ├── RQ1_effect_sizes.png
  ├── MANOVA_results.txt
  ├── RQ2_depressed_gender_ANOVAs.csv     (columns: Marker, F, p_gender, Cohen_d, Mean_F, Mean_M, p_fdr, sig_fdr)
  ├── RQ2_effect_sizes.png
  ├── RQ2_significant_markers.png         (only if significant results exist)
  ├── logistic_Full_Sample.csv/png
  ├── logistic_Females_Only.csv/png
  ├── logistic_Males_Only.csv/png
  └── radar_chart.png
""")