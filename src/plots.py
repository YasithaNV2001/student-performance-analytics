"""The 16 figures of the analysis. Each function saves its figure to `out_dir` and returns it."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from .clustering import K_RANGE
from .config import FIGURES_DIR, RANDOM_STATE
from .features import BEHAV_COLS

PALETTE = sns.color_palette("Set2")
SUBJECT_COLORS = {'Math': PALETTE[0], 'Portuguese': PALETTE[1]}

# Numeric columns shown in the correlation heatmaps (Figure 4)
CORR_COLS = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
             'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']


def set_style():
    """Shared Matplotlib style for every figure."""
    plt.rcParams.update({
        'figure.dpi': 120, 'font.size': 11,
        'axes.titlesize': 13, 'axes.labelsize': 11,
        'axes.spines.top': False, 'axes.spines.right': False,
    })


def _save(fig, filename, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, bbox_inches='tight')
    return fig


def fig1_grade_distributions(df_mat_p, df_por_p, out_dir=FIGURES_DIR):
    """Figure 1: G1, G2 and G3 histograms for Math and Portuguese, with the pass mark."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for row, (df, subj) in enumerate([(df_mat_p, 'Math'), (df_por_p, 'Portuguese')]):
        color = SUBJECT_COLORS[subj]
        for col, (grade, title) in enumerate([('G1', 'Period 1'), ('G2', 'Period 2'), ('G3', 'Final')]):
            ax = axes[row][col]
            ax.hist(df[grade], bins=20, color=color, edgecolor='white', alpha=0.85)
            ax.axvline(10, color='red', linestyle='--', lw=1.5, label='Pass (10)')
            ax.set_title(f'{subj} — {title} Grade')
            ax.set_xlabel('Grade (0–20)')
            ax.set_ylabel('Number of Students')
            ax.legend(fontsize=8)
    plt.suptitle('Grade Distributions Across Periods: Math vs Portuguese', fontweight='bold', y=1.01)
    plt.tight_layout()
    return _save(fig, 'fig1_grade_distributions.png', out_dir)


def fig2_cross_subject(df_both_p, out_dir=FIGURES_DIR):
    """Figure 2: Math vs Portuguese grades, grade-gap histogram and pass/fail cross-tabulation."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].scatter(df_both_p['G3_math'], df_both_p['G3_port'],
                    alpha=0.45, color=PALETTE[4], s=40, edgecolors='white')
    r, _ = stats.pearsonr(df_both_p['G3_math'], df_both_p['G3_port'])
    m_coef, b = np.polyfit(df_both_p['G3_math'], df_both_p['G3_port'], 1)
    x_line = np.linspace(0, 20, 100)
    axes[0].plot(x_line, m_coef * x_line + b, 'r--', lw=1.5)
    axes[0].set_xlabel('Math Final Grade (G3)')
    axes[0].set_ylabel('Portuguese Final Grade (G3)')
    axes[0].set_title(f'Math vs Portuguese G3\n(r = {r:.3f}, p < 0.001)')

    axes[1].hist(df_both_p['grade_gap'], bins=25, color=PALETTE[5], edgecolor='white', alpha=0.85)
    axes[1].axvline(0, color='red', linestyle='--', lw=1.5)
    axes[1].set_xlabel('Grade Gap (Math minus Portuguese)')
    axes[1].set_ylabel('Number of Students')
    axes[1].set_title(f'Grade Gap Distribution\nMean = {df_both_p.grade_gap.mean():.2f}')

    ct = pd.crosstab(df_both_p['pass_math'], df_both_p['pass_port'])
    ct.index = ['Fail Math', 'Pass Math']
    ct.columns = ['Fail Portuguese', 'Pass Portuguese']
    sns.heatmap(ct, annot=True, fmt='d', cmap='YlOrRd', ax=axes[2], linewidths=1, cbar=False)
    axes[2].set_title('Pass/Fail Cross-tabulation\n(Dual-Enrolled Students)')

    plt.suptitle('Cross-Subject Analysis — Students Enrolled in Both Courses', fontweight='bold', y=1.02)
    plt.tight_layout()
    return _save(fig, 'fig2_cross_subject.png', out_dir)


def fig3_feature_distributions(df_mat_p, df_por_p, out_dir=FIGURES_DIR):
    """Figure 3: distributions of six key features for passing (filled) and failing (outline) students."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    feats = ['studytime', 'failures', 'absences', 'avg_alcohol', 'parent_edu_avg', 'grade_trend']
    titles = ['Study Time', 'Past Failures', 'Absences', 'Avg Alcohol', 'Parent Edu Avg', 'Grade Trend (G2-G1)']
    for i, (feat, title) in enumerate(zip(feats, titles)):
        ax = axes[i // 3][i % 3]
        for subj, df, color in [('Math', df_mat_p, SUBJECT_COLORS['Math']),
                                ('Portuguese', df_por_p, SUBJECT_COLORS['Portuguese'])]:
            pass_vals = df[df.pass_fail == 1][feat].dropna()
            fail_vals = df[df.pass_fail == 0][feat].dropna()
            ax.hist(pass_vals, bins=15, alpha=0.5, color=color, label=f'{subj} Pass', density=True)
            ax.hist(fail_vals, bins=15, alpha=0.3, color=color, linestyle='--',
                    histtype='step', lw=1.5, label=f'{subj} Fail', density=True)
        ax.set_title(title)
        ax.set_xlabel(feat)
        ax.set_ylabel('Density')
        if i == 0:
            ax.legend(fontsize=7)
    plt.suptitle('Feature Distributions by Pass/Fail — Math vs Portuguese', fontweight='bold', y=1.01)
    plt.tight_layout()
    return _save(fig, 'fig3_feature_distributions.png', out_dir)


def fig4_correlation_heatmaps(df_mat_p, df_por_p, out_dir=FIGURES_DIR):
    """Figure 4: lower-triangle correlation matrices for Math and Portuguese."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, (df, subj) in zip(axes, [(df_mat_p, 'Math'), (df_por_p, 'Portuguese')]):
        cols = [c for c in CORR_COLS if c in df.columns]
        corr = df[cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))  # the upper triangle is a mirror image
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                    center=0, ax=ax, linewidths=0.4, annot_kws={'size': 7},
                    cbar_kws={'shrink': 0.7}, square=True)
        ax.set_title(f'{subj} — Feature Correlation Matrix', fontweight='bold')
    plt.tight_layout()
    return _save(fig, 'fig4_correlation_heatmaps.png', out_dir)


def fig5_xapi_eda(df_x, out_dir=FIGURES_DIR):
    """Figure 5: xAPI behaviour by class (violins), mean engagement and absence pattern."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for i, col in enumerate(BEHAV_COLS):
        ax = axes[0][i] if i < 3 else axes[1][i - 3]
        sns.violinplot(data=df_x, x='Class', y=col, hue='Class', legend=False, order=['L', 'M', 'H'],
                       palette={'L': PALETTE[3], 'M': PALETTE[1], 'H': PALETTE[2]},
                       inner='box', ax=ax)
        ax.set_title(f'{col} by Performance Class')
        ax.set_xlabel('Class  (L=Low, M=Medium, H=High)')

    ax = axes[1][1]
    means = df_x.groupby('Class')['engagement_score'].mean()[['L', 'M', 'H']]
    bars = ax.bar(['Low', 'Medium', 'High'], means.values,
                  color=[PALETTE[3], PALETTE[1], PALETTE[2]], edgecolor='white')
    for bar, val in zip(bars, means.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_title('Mean Engagement Score by Class')
    ax.set_ylabel('Engagement Score (0-100)')

    ax = axes[1][2]
    ct = df_x.groupby(['Class', 'StudentAbsenceDays']).size().unstack(fill_value=0)
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_pct.loc[['L', 'M', 'H']].plot(kind='bar', ax=ax,
                                     color=[PALETTE[0], PALETTE[3]], edgecolor='white', width=0.6)
    ax.set_title('Absence Pattern by Class')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title='Absences', fontsize=9)

    plt.suptitle('xAPI Behavioral Engagement Patterns by Performance Class', fontweight='bold', y=1.01)
    plt.tight_layout()
    return _save(fig, 'fig5_xapi_eda.png', out_dir)


def fig6_rf_comparison(res_math, res_por, out_dir=FIGURES_DIR):
    """Figure 6: confusion matrices and ROC curves of the Math and Portuguese Random Forests."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, res, subj, cmap in [(axes[0], res_math, 'Math', 'Blues'),
                                (axes[1], res_por, 'Portuguese', 'Greens')]:
        cm = confusion_matrix(res['y_test'], res['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                    xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'],
                    cbar=False, linewidths=1)
        ax.set_title(f"{subj} RF  (Acc={res['accuracy']*100:.1f}%, AUC={res['auc']:.3f})", fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

    fpr_m, tpr_m, _ = roc_curve(res_math['y_test'], res_math['y_prob'])
    fpr_p, tpr_p, _ = roc_curve(res_por['y_test'], res_por['y_prob'])
    axes[2].plot(fpr_m, tpr_m, color=SUBJECT_COLORS['Math'], lw=2, label=f"Math (AUC={res_math['auc']:.3f})")
    axes[2].plot(fpr_p, tpr_p, color=SUBJECT_COLORS['Portuguese'], lw=2,
                 label=f"Portuguese (AUC={res_por['auc']:.3f})")
    axes[2].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC=0.500)')
    axes[2].fill_between(fpr_m, tpr_m, alpha=0.08, color=SUBJECT_COLORS['Math'])
    axes[2].fill_between(fpr_p, tpr_p, alpha=0.08, color=SUBJECT_COLORS['Portuguese'])
    axes[2].set_title('ROC Curves — Math vs Portuguese', fontweight='bold')
    axes[2].set_xlabel('False Positive Rate')
    axes[2].set_ylabel('True Positive Rate')
    axes[2].legend()
    plt.tight_layout()
    return _save(fig, 'fig6_rf_comparison.png', out_dir)


def fig7_feature_importance(fi_mat, fi_por, out_dir=FIGURES_DIR):
    """Figure 7: top 15 Random Forest feature importances, Math next to Portuguese."""
    fi_df = pd.DataFrame({'Math': fi_mat, 'Portuguese': fi_por}).fillna(0)
    fi_df = fi_df.sort_values('Math', ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(fi_df))
    w = 0.38
    ax.barh(x + w / 2, fi_df['Math'], w, label='Math', color=SUBJECT_COLORS['Math'], alpha=0.85, edgecolor='white')
    ax.barh(x - w / 2, fi_df['Portuguese'], w, label='Portuguese', color=SUBJECT_COLORS['Portuguese'],
            alpha=0.85, edgecolor='white')
    ax.set_yticks(x)
    ax.set_yticklabels(fi_df.index)
    ax.set_title('Top 15 Feature Importances — Math vs Portuguese\n(longer bar = model relied on it more)',
                 fontweight='bold')
    ax.set_xlabel('Feature Importance Score')
    ax.legend()
    plt.tight_layout()
    return _save(fig, 'fig7_feature_importance_comparison.png', out_dir)


def fig8_gb_xapi(res_gb, feature_names, out_dir=FIGURES_DIR):
    """Figure 8: confusion matrix and feature importances of the xAPI Gradient Boosting model."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cm_gb = confusion_matrix(res_gb['y_test'], res_gb['y_pred'])
    sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Purples', ax=axes[0],
                xticklabels=['Low', 'Med', 'High'], yticklabels=['Low', 'Med', 'High'],
                cbar=False, linewidths=1)
    axes[0].set_title(f"xAPI Gradient Boosting\nAcc={res_gb['accuracy']*100:.1f}%, AUC={res_gb['auc']:.3f}",
                      fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    gb_fi = pd.DataFrame({'feature': feature_names,
                          'importance': res_gb['model'].feature_importances_}).sort_values('importance', ascending=True)
    axes[1].barh(gb_fi['feature'], gb_fi['importance'], color=PALETTE[4], edgecolor='white')
    axes[1].set_title('Feature Importances — xAPI Gradient Boosting', fontweight='bold')
    axes[1].set_xlabel('Importance Score')
    plt.tight_layout()
    return _save(fig, 'fig8_gb_xapi.png', out_dir)


def fig9_all_models_summary(res_math, res_por, res_gb, out_dir=FIGURES_DIR):
    """Figure 9: accuracy, CV accuracy and AUC of all three models, and all ROC curves."""
    names = ['Math RF', 'Portuguese RF', 'xAPI GB']
    results = [res_math, res_por, res_gb]
    accs = [r['accuracy'] * 100 for r in results]
    aucs = [r['auc'] for r in results]
    cv_mean = [r['cv'].mean() * 100 for r in results]
    cv_std = [r['cv'].std() * 100 for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(names))
    w = 0.3
    axes[0].bar(x - w, accs, w, label='Test Accuracy (%)', color=PALETTE[0], alpha=0.85, edgecolor='white')
    axes[0].bar(x, cv_mean, w, label='CV Accuracy (%)', color=PALETTE[1], alpha=0.85, edgecolor='white')
    axes[0].errorbar(x, cv_mean, yerr=cv_std, fmt='none', color='black', capsize=4)
    axes[0].bar(x + w, [a * 100 for a in aucs], w, label='AUC x 100', color=PALETTE[2], alpha=0.85, edgecolor='white')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    axes[0].set_ylabel('Score')
    axes[0].set_ylim(0, 115)
    axes[0].set_title('All Models — Performance Comparison\n(error bars = 5-fold CV std)', fontweight='bold')
    axes[0].legend(fontsize=9)
    for i, (a, cv, auc) in enumerate(zip(accs, cv_mean, aucs)):
        axes[0].text(i - w, a + 1, f'{a:.1f}', ha='center', fontsize=8)
        axes[0].text(i, cv + 1, f'{cv:.1f}', ha='center', fontsize=8)
        axes[0].text(i + w, auc * 100 + 1, f'{auc:.3f}', ha='center', fontsize=8)

    fpr_m, tpr_m, _ = roc_curve(res_math['y_test'], res_math['y_prob'])
    fpr_p, tpr_p, _ = roc_curve(res_por['y_test'], res_por['y_prob'])
    axes[1].plot(fpr_m, tpr_m, color=SUBJECT_COLORS['Math'], lw=2, label=f"Math RF (AUC={res_math['auc']:.3f})")
    axes[1].plot(fpr_p, tpr_p, color=SUBJECT_COLORS['Portuguese'], lw=2,
                 label=f"Portuguese RF (AUC={res_por['auc']:.3f})")
    y_bin, y_prob_gb = res_gb['y_bin'], res_gb['y_prob']
    for i, (cls, col) in enumerate(zip(['Low', 'Med', 'High'], [PALETTE[3], PALETTE[4], PALETTE[2]])):
        fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_prob_gb[:, i])
        auc_i = roc_auc_score(y_bin[:, i], y_prob_gb[:, i])
        axes[1].plot(fpr_i, tpr_i, color=col, lw=1.5, linestyle='--', label=f'xAPI {cls} (AUC={auc_i:.3f})')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='Random baseline')
    axes[1].set_title('ROC Curves — All Models', fontweight='bold')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    return _save(fig, 'fig9_all_models_summary.png', out_dir)


def fig10_uci_clustering(uci_cl, out_dir=FIGURES_DIR):
    """Figure 10: elbow curve, silhouette scores and PCA view of the UCI clusters."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    axes[0].plot(K_RANGE, uci_cl['inertia'], 'o-', color=PALETTE[0], lw=2, ms=7)
    axes[0].axvline(3, color='red', linestyle='--', alpha=0.7, label='k=3 (chosen)')
    axes[0].set_title('Elbow Method — UCI Combined', fontweight='bold')
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Inertia (within-cluster variance)')
    axes[0].legend()

    axes[1].plot(K_RANGE, uci_cl['silhouette_by_k'], 's-', color=PALETTE[1], lw=2, ms=7)
    axes[1].axvline(uci_cl['best_k'], color='red', linestyle='--', alpha=0.7, label=f"Best k={uci_cl['best_k']}")
    axes[1].set_title('Silhouette Scores — UCI Combined', fontweight='bold')
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Silhouette Score  (higher = better)')
    axes[1].legend()

    # PCA only for the picture; the clustering itself used all 8 features
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(uci_cl['X'])
    colors = {'Low Achiever': PALETTE[3], 'Mid Achiever': PALETTE[1], 'High Achiever': PALETTE[2]}
    for name, color in colors.items():
        mask = uci_cl['data']['cluster_name'] == name
        axes[2].scatter(X_pca[mask, 0], X_pca[mask, 1], c=[color], label=name, alpha=0.6, s=35, edgecolors='white')
    axes[2].set_title(f"PCA Projection — UCI Clusters\nSilhouette={uci_cl['silhouette']:.3f}", fontweight='bold')
    axes[2].set_xlabel(f'PC1  ({pca.explained_variance_ratio_[0]*100:.1f}% variance explained)')
    axes[2].set_ylabel(f'PC2  ({pca.explained_variance_ratio_[1]*100:.1f}% variance explained)')
    axes[2].legend()
    plt.tight_layout()
    return _save(fig, 'fig10_uci_clustering.png', out_dir)


def fig11_xapi_clustering(xapi_cl, out_dir=FIGURES_DIR):
    """Figure 11: PCA view of the xAPI clusters and their normalised behaviour profiles."""
    data = xapi_cl['data']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(xapi_cl['X'])
    colors = {'Passive': PALETTE[3], 'Moderate': PALETTE[1], 'Active': PALETTE[2]}
    for name, color in colors.items():
        mask = data['cluster_name'] == name
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], c=[color], label=name, alpha=0.65, s=40, edgecolors='white')
    axes[0].set_title(f"xAPI Clusters — PCA Projection\nSilhouette={xapi_cl['silhouette']:.3f}", fontweight='bold')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[0].legend()

    profile = data.groupby('cluster_name')[BEHAV_COLS].mean()
    norm_p = (profile - profile.min()) / (profile.max() - profile.min()) * 100
    x_ = np.arange(len(BEHAV_COLS))
    w_ = 0.25
    for i, (name, color) in enumerate(colors.items()):
        axes[1].bar(x_ + i * w_, norm_p.loc[name], w_, label=name, color=color, edgecolor='white', alpha=0.85)
    axes[1].set_xticks(x_ + w_)
    axes[1].set_xticklabels(BEHAV_COLS, rotation=15)
    axes[1].set_title('Behavioral Profile by Cluster\n(normalised 0-100)', fontweight='bold')
    axes[1].set_ylabel('Normalised Score')
    axes[1].legend()
    plt.tight_layout()
    return _save(fig, 'fig11_xapi_clustering.png', out_dir)


def fig12_association_rules(xapi_outcome_rules, uci_outcome_rules, out_dir=FIGURES_DIR):
    """Figure 12: support vs confidence of the class-predicting rules, coloured by lift."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, rules_df, title in [
        (axes[0], xapi_outcome_rules, 'xAPI Behavioral Rules\n(consequent = class outcome)'),
        (axes[1], uci_outcome_rules, 'UCI Academic Rules\n(consequent = pass or fail)'),
    ]:
        if len(rules_df) == 0:
            ax.text(0.5, 0.5, 'No rules found at current thresholds',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontweight='bold')
            continue
        sc = ax.scatter(rules_df['support'], rules_df['confidence'],
                        c=rules_df['lift'], cmap='YlOrRd', s=70, alpha=0.8,
                        edgecolors='grey', linewidth=0.3)
        plt.colorbar(sc, ax=ax, label='Lift')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Support  (fraction of students with this pattern)')
        ax.set_ylabel('Confidence  (reliability when pattern occurs)')
        best = rules_df.loc[rules_df['lift'].idxmax()]
        ax.annotate(f"Best lift={best['lift']:.2f}", (best['support'], best['confidence']),
                    textcoords='offset points', xytext=(6, 6),
                    fontsize=8, color='darkred', fontweight='bold')
    plt.tight_layout()
    return _save(fig, 'fig12_association_rules.png', out_dir)


def fig13_studytime_vs_grade(df_mat_p, df_por_p, out_dir=FIGURES_DIR):
    """Figure 13: final grade by weekly study time, Math and Portuguese."""
    labels = {1: '<2h', 2: '2-5h', 3: '5-10h', 4: '>10h'}
    order = ['<2h', '2-5h', '5-10h', '>10h']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (df, subj) in zip(axes, [(df_mat_p, 'Math'), (df_por_p, 'Portuguese')]):
        df_tmp = df.copy()
        df_tmp['study_label'] = df_tmp['studytime'].map(labels)
        order_present = [o for o in order if o in df_tmp['study_label'].unique()]
        counts = df_tmp['study_label'].value_counts()
        sns.boxplot(data=df_tmp, x='study_label', y='G3', order=order_present,
                    color=SUBJECT_COLORS[subj], ax=ax, width=0.5)
        ax.axhline(10, color='red', linestyle='--', lw=1.2, label='Pass threshold (10)')
        ax.set_title(f'{subj} — Final Grade by Weekly Study Time', fontweight='bold')
        ax.set_xlabel('Weekly Study Time')
        ax.set_ylabel('Final Grade G3 (0-20)')
        ax.legend(fontsize=9)
        for i, cat in enumerate(order_present):
            ax.text(i, ax.get_ylim()[0] + 0.2, f'n={counts.get(cat, 0)}', ha='center', fontsize=8, color='grey')
    plt.suptitle('Study Time vs Final Grade: Math and Portuguese', fontweight='bold', y=1.02)
    plt.tight_layout()
    return _save(fig, 'fig13_studytime_vs_grade.png', out_dir)


def fig14_baselines_early_warning(baseline_results, early_results, out_dir=FIGURES_DIR):
    """Figure 14: accuracy and fail recall of the baselines and the early-warning models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    labels = ['Majority\nclass', 'Rule\nG2 >= 10', 'LogReg\n(all)', 'RF\n(all)',
              'RF\n(G1 only)', 'LogReg\n(no grades)', 'RF\n(no grades)']
    for ax, subj in zip(axes, ['Math', 'Portuguese']):
        b = baseline_results[subj].set_index('Model')
        e = early_results[subj].set_index('Model')
        order = [(b, 'Majority class'), (b, 'Rule: G2 >= 10'), (b, 'Logistic Regression'),
                 (b, 'Random Forest'), (e, 'Random Forest (G1 only)'),
                 (e, 'Logistic Regression'), (e, 'Random Forest')]
        acc = [src.loc[m, 'Accuracy'] for src, m in order]
        rec = [src.loc[m, 'Fail recall'] for src, m in order]
        x = np.arange(len(labels))
        w = 0.38
        ax.bar(x - w / 2, acc, w, label='Test accuracy (%)', color=PALETTE[0], edgecolor='white')
        ax.bar(x + w / 2, rec, w, label='Fail recall (%)', color=PALETTE[1], edgecolor='white')
        for i, (a, r) in enumerate(zip(acc, rec)):
            ax.text(i - w / 2, a + 1, f'{a:.0f}', ha='center', fontsize=8)
            ax.text(i + w / 2, r + 1, f'{r:.0f}', ha='center', fontsize=8)
        ax.axvspan(4.5, 6.5, color='grey', alpha=0.08)
        ax.text(5.5, 106, 'no grades', ha='center', fontsize=9, style='italic', color='dimgrey')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 112)
        ax.set_title(f'{subj} — Baselines vs Early-Warning Models', fontweight='bold')
    axes[0].set_ylabel('Score (%)')
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc='lower center', ncol=2, fontsize=10, frameon=False)
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    return _save(fig, 'fig14_baselines_early_warning.png', out_dir)


def fig15_shap_summary(shap_fail, out_dir=FIGURES_DIR):
    """Figure 15: SHAP beeswarm for the Portuguese early-warning model (Fail class)."""
    np.random.seed(RANDOM_STATE)  # the beeswarm jitters dots with NumPy's global generator
    plt.figure()
    shap.plots.beeswarm(shap_fail, max_display=12, show=False)
    plt.title('What Drives Fail Risk Before Any Grades Exist?\n'
              '(Portuguese, early-warning Random Forest — SHAP for the Fail class)',
              fontweight='bold', fontsize=11)
    return _save(plt.gcf(), 'fig15_shap_summary.png', out_dir)


def fig16_shap_single_student(shap_fail, index, fail_prob, actual='Fail', out_dir=FIGURES_DIR):
    """Figure 16: SHAP waterfall showing why one student was flagged."""
    plt.figure()
    shap.plots.waterfall(shap_fail[index], max_display=10, show=False)
    plt.title(f'Why This Student Was Flagged\n(predicted fail probability {fail_prob:.2f}, actual: {actual})',
              fontweight='bold', fontsize=11)
    return _save(plt.gcf(), 'fig16_shap_single_student.png', out_dir)
