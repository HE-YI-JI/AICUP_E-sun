"""
Positive–Unlabeled (PU) learning with iterative self-training using XGBoost.

This module implements a PU-learning workflow that identifies reliable
negative accounts, iteratively expands the labeled dataset using model
confidence, and trains a sequence of XGBoost models to refine the decision
boundary. It operates on preprocessed node-level features (final_data.csv)
and produces a prediction file (upload.csv) for downstream evaluation.

The core component is the PU2S class, which repeatedly:
    1. Trains a base estimator on known positives and reliable negatives.
    2. Predicts probabilities on the full dataset.
    3. Adds new pseudo-positive and pseudo-negative samples using adaptive
       confidence thresholds.
    4. Clones and initializes the next estimator to continue refinement.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.base import clone

class PU2S:
    """
    Iterative positive–unlabeled (PU) learning with self-training.

    This class implements a multi-iteration training procedure where a base
    estimator (e.g., XGBoost) is repeatedly fitted on:
        * All confirmed positive samples.
        * A user-provided set of reliable negative samples.
        * Newly inferred pseudo-positive and pseudo-negative samples.

    At each iteration, the current estimator is trained, probabilities are
    computed for all samples, and new training points are added based on
    adaptive confidence thresholds. A new cloned estimator is appended to
    the model list for the next iteration.
    """
    def __init__(self, base_estimator, reliable_negative, iterations, positive_threshold=0.95, negative_threshold=0.05):
        self.base_estimator = base_estimator
        self.model = [clone(base_estimator)]
        self.eval_score = []
        self.train_size = []
        self.reliable_negative = reliable_negative
        self.iterations = iterations
        self.trained = 0
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold

    def _fit(self, model, X, y, eval_set: tuple = None, **fit_params):
        """Fit the given model with training data and optional evaluation set."""
        model.fit(X, y, eval_set=eval_set, **fit_params)

    def _predict_proba(self, model, X):
        """Return predicted positive-class probabilities for X."""
        return model.predict_proba(X)[:, 1]
    
    def _predict(self, model, X, threshold=0.5):
        """Return binary predictions for X using the given probability threshold."""
        return (model.predict_proba(X)[:, 1] > threshold).astype(bool)

    def fit(self, X, y, eval_set: tuple = None, **fit_params):
        """
        Fit the PU-learning model over multiple self-training iterations.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix including an 'acct' column. The feature matrix is
            internally split into training/expansion sets based on reliable
            negatives and positive labels.
        y : pd.Series
            Binary labels where 1 indicates known positives and 0 indicates
            unlabeled samples.
        eval_set : tuple, optional
            A pair (X_eval, y_eval) used for monitoring F1 score across
            iterations.
        fit_params : dict
            Additional parameters passed to the base estimator's fit() method.

        Notes
        -----
        At each iteration:
            1. Fit model on current labeled dataset.
            2. Predict probabilities on all samples.
            3. Add pseudo-positive samples above adaptive positive threshold.
            4. Add pseudo-negative samples below adaptive negative threshold.
            5. Deduplicate and continue training with a cloned estimator.
        Training stops early when the training size and evaluation score become
        stable over recent iterations.
        """
        reliable_negative_idx = X['acct'].isin(self.reliable_negative).values
        reliable_positive_idx = y == 1
        X = X.drop(columns=['acct'])
        X_train = X[reliable_negative_idx | reliable_positive_idx]
        y_train = y[reliable_negative_idx | reliable_positive_idx]
        print(f"Initial training size: {len(y_train)}, Positives: {y_train.sum()}, Negatives: {len(y_train)-y_train.sum()}")
        self.pbar = tqdm(total=self.iterations, desc="Training")
        eval_x, eval_y = eval_set if eval_set is not None else (None, None)

        try:
            while self.trained < self.iterations:
                self._fit(self.model[self.trained], X_train, y_train, eval_set=[(eval_x.drop(columns=['acct']), eval_y)] if eval_set is not None else None, **fit_params)
                result = self._predict_proba(self.model[self.trained], X)
                positive_threshold = self.positive_threshold if result.max() > self.positive_threshold else result.max() if result.max() > 0.8 else 1
                negative_threshold = self.negative_threshold if result.min() < self.negative_threshold else result.min() if result.min() < 0.2 else 0
                self.eval_score.append(f1_score(eval_y, self._predict(self.model[self.trained], eval_x.drop(columns=['acct']), threshold=positive_threshold)) if eval_set is not None else 0)
                self.train_size.append(len(y_train))
                self.pbar.set_postfix_str(f"Eval Score: {self.eval_score[-1]:.4f}, train_size: {self.train_size[-1]}")
                self.pbar.update(1)
                self.trained = self.trained + 1
                if np.std(self.train_size[-10:]) <= 0 and self.trained > 10 and np.std(self.eval_score[-10:]) < 0.01:
                    break
                X_train = pd.concat([X_train, X[result > positive_threshold]])
                y_train = pd.concat([y_train, pd.Series([1]*sum(result > positive_threshold))])
                X_train = pd.concat([X_train, X[result < negative_threshold]]).reset_index(drop=True)
                y_train = pd.concat([y_train, pd.Series([0]*sum(result < negative_threshold))]).reset_index(drop=True)
                X_train = X_train.drop_duplicates()
                y_train = y_train[X_train.index].reset_index(drop=True)
                X_train = X_train.reset_index(drop=True)
                self.model.append(clone(self.base_estimator))
        except Exception as e:
            print(e)
        finally:
            self.pbar.close()

    def predict_proba(self, X):
        """
        Predict positive-class probabilities using the final trained estimator.

        Raises
        ------
        ValueError
            If called before any estimator has been trained.
        """
        if self.trained < 1:
            raise ValueError("Model is not trained yet!")
        return self.model[-1].predict_proba(X)[:, 1]

if __name__ == "__main__":

    # Prepare training, validation, and test datasets for PU-learning.
    # Train iterative PU model and generate output predictions for submission
    all_data = pd.read_csv('final_data.csv').drop(columns=['0.2'])
    test_y = pd.read_csv(r'acct_predict.csv')
    safe_acct = pd.read_csv('safe_acct.csv')
    eval_set = pd.concat([all_data[all_data['acct'].isin(safe_acct['acct'])].sample(10), all_data[all_data['is_alert'] == 1].sample(10)])
    train = all_data.drop(index=eval_set.index).reset_index(drop=True)
    eval_set = eval_set.reset_index(drop=True)

    base = XGBClassifier(
        n_estimators=3000,
        learning_rate=0.003,
        max_depth=12,
        scale_pos_weight= train['is_alert'].value_counts()[0] / train['is_alert'].value_counts()[1],
        subsample=0.8,
        device='cuda',
        verbosity=0,
    )
    pu_estimator = PU2S(base, reliable_negative=safe_acct['acct'], iterations=100, positive_threshold=0.95, negative_threshold=0.05)
    pu_estimator.fit(
        train.drop(columns=['is_alert']),
        train['is_alert'],
        eval_set=(eval_set.drop(columns=['is_alert']), eval_set['is_alert']),
        verbose=0
    )

    test = all_data[all_data['acct'].isin(test_y['acct'])].reset_index(drop=True).set_index('acct').reindex(test_y['acct']).reset_index()
    assert not test[test.isna().any(axis=1)].empty

    predict = pu_estimator.model[1].predict_proba(test.drop(columns=['is_alert', 'acct']))[:, 1]
    test_y['label'] = (predict>0.9999995).astype(int)
    test_y.to_csv('upload.csv', index=False)

    print("Done!")
    print("save to upload.csv")