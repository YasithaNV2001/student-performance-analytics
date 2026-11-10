"""Train the early-warning model used by the Streamlit demo (app.py).

    python train_early_warning.py

Target: fail = final Math grade (G3) below 10. Features: every column of
data/processed/student-mat-cleaned.csv except G2 and G3, so the model can be
used after the first term (G1 is allowed). The text columns are one-hot
encoded and a Random Forest (300 trees, max depth 10, balanced class weights,
random_state 42) is trained on a stratified 80% split and tested on the other 20%.
The fitted pipeline is saved to models/early_warning.joblib.
"""
from src import early_warning


def main():
    pipe, auc = early_warning.train()
    path = early_warning.save(pipe)
    cv = early_warning.cv_auc()
    n_features = len(pipe.named_steps['prep'].feature_names_in_)
    print(f'Features: {n_features} (everything except G2 and G3)')
    print(f'Test AUC: {auc:.3f}')
    print(f'5-fold CV AUC: {cv.mean():.3f} +/- {cv.std():.3f}')
    print(f'Model saved to {path}')


if __name__ == '__main__':
    main()
