import pandas as pd
import numpy as np
import pandas as pd
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.base import clone

class PU2S:
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
        model.fit(X, y, eval_set=eval_set, **fit_params)

    def _predict_proba(self, model, X):
        return model.predict_proba(X)[:, 1]
    
    def _predict(self, model, X, threshold=0.5):
        return (model.predict_proba(X)[:, 1] > threshold).astype(bool)

    def fit(self, X, y, eval_set: tuple = None, **fit_params):
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
        if self.trained < 1:
            raise ValueError("Model is not trained yet!")
        
        return self.model[-1].predict_proba(X)[:, 1]

if __name__ == "__main__":
    
    # Read file.
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
