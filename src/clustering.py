"""K-Means clustering of UCI and xAPI students (notebook section 7)."""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE
from .features import BEHAV_COLS

UCI_CLUSTER_FEATURES = ['studytime', 'failures', 'absences', 'avg_alcohol',
                        'parent_edu_avg', 'G1', 'G2', 'G3']
K_RANGE = range(2, 9)
N_CLUSTERS = 3  # chosen for interpretability: three named learner profiles


def pool_uci(df_mat_p, df_por_p):
    """Stack Math and Portuguese students into one table with a `subject` column."""
    cols = UCI_CLUSTER_FEATURES + ['pass_fail']
    return pd.concat([df_mat_p[cols].assign(subject='Math'),
                      df_por_p[cols].assign(subject='Portuguese')], ignore_index=True)


def k_sweep(X, k_range=K_RANGE):
    """Inertia (elbow method) and silhouette score for each k."""
    inertia, sil = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(X, labels))
    return inertia, sil


def name_clusters(labels, score, names):
    """Map cluster ids to `names`, from the lowest to the highest mean `score`."""
    order = score.groupby(labels).mean().sort_values().index
    return dict(zip(order, names))


def _fit(X, data, rank_by, names):
    """Sweep k, fit k = 3 and add `cluster` and `cluster_name` columns to `data`."""
    inertia, sil = k_sweep(X)
    data['cluster'] = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE,
                             n_init=10).fit_predict(X)
    data['cluster_name'] = data['cluster'].map(name_clusters(data['cluster'], data[rank_by], names))
    return {'data': data, 'X': X, 'inertia': inertia, 'silhouette_by_k': sil,
            'best_k': list(K_RANGE)[sil.index(max(sil))],
            'silhouette': silhouette_score(X, data['cluster']),
            'davies_bouldin': davies_bouldin_score(X, data['cluster'])}


def cluster_uci(df_mat_p, df_por_p):
    """Cluster all UCI students on standardised grades and habits; name clusters by mean G3."""
    data = pool_uci(df_mat_p, df_por_p)
    X = StandardScaler().fit_transform(data[UCI_CLUSTER_FEATURES].fillna(0))
    return _fit(X, data, 'G3', ['Low Achiever', 'Mid Achiever', 'High Achiever'])


def cluster_xapi(df_x):
    """Cluster xAPI students on the four behaviour counts only; name clusters by engagement."""
    data = df_x.copy()
    X = StandardScaler().fit_transform(data[BEHAV_COLS])
    return _fit(X, data, 'engagement_score', ['Passive', 'Moderate', 'Active'])
