from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one match in {path}, found {count}: {old[:80]!r}")
    file_path.write_text(text.replace(old, new, 1), encoding="utf-8")


data_loader = "LogReg/LASSO/module/data_loader.py"
replace_once(data_loader, "import json\n", "import json\nimport os\n")

replace_once(
    data_loader,
    '''def extract_base_scores(json_data: List[List[Dict[str, Any]]]) -> Dict[Tuple[str, int], float]:
    """ベースモデルのlogit_clarificationを抽出"""
    scores = {}
    for conversation in json_data:
        for turn in conversation:
            # conv_idとturn_idを文字列/整数に統一
            conv_id = str(turn['conv_id'])
            turn_id = int(turn['turn_id'])
            key = (conv_id, turn_id)
            
            logit = turn.get('logit_clarification')
            if logit is not None:
                scores[key] = float(logit)
    
    return scores
''',
    '''PLM_SCORE_MODE_ENV = 'QPP4SIP_PLM_SCORE_MODE'
DEFAULT_PLM_SCORE_MODE = 'logit_difference'
VALID_PLM_SCORE_MODES = {'logit_difference', 'positive_logit'}


def resolve_plm_score_mode(score_mode: str = None) -> str:
    """PLMスコア方式を解決し、妥当性を検証する。"""
    resolved_mode = score_mode or os.getenv(PLM_SCORE_MODE_ENV, DEFAULT_PLM_SCORE_MODE)
    resolved_mode = resolved_mode.strip().lower().replace('-', '_')
    if resolved_mode not in VALID_PLM_SCORE_MODES:
        valid_modes = ', '.join(sorted(VALID_PLM_SCORE_MODES))
        raise ValueError(f"Unknown PLM score mode: {resolved_mode}. Choose one of: {valid_modes}")
    return resolved_mode


def _get_plm_score(turn: Dict[str, Any], score_mode: str):
    positive_logit = turn.get('logit_clarification')
    if positive_logit is None:
        return None
    if score_mode == 'positive_logit':
        return float(positive_logit)

    negative_logit = turn.get('logit_not_clarification')
    if negative_logit is None:
        raise ValueError(
            "logit_difference requires logit_not_clarification. "
            f"Use {PLM_SCORE_MODE_ENV}=positive_logit for legacy prediction files."
        )
    return float(positive_logit) - float(negative_logit)


def extract_base_scores(
    json_data: List[List[Dict[str, Any]]],
    score_mode: str = None,
) -> Dict[Tuple[str, int], float]:
    """PLMスコアを抽出する。既定値は正例logit - 負例logit。"""
    resolved_mode = resolve_plm_score_mode(score_mode)
    scores = {}
    for conversation in json_data:
        for turn in conversation:
            conv_id = str(turn['conv_id'])
            turn_id = int(turn['turn_id'])
            key = (conv_id, turn_id)
            score = _get_plm_score(turn, resolved_mode)
            if score is not None:
                scores[key] = score
    return scores
''',
)

replace_once(
    data_loader,
    "def load_base_scores(split: str, dataset: str, base_dir: Path, base_experiment_names: List[str] = None, use_bert: bool = True, use_roberta: bool = True, use_transfer: bool = False, use_full_train_model_for_dev: bool = False) -> Dict[str, Dict[Tuple[str, int], float]]:\n",
    "def load_base_scores(split: str, dataset: str, base_dir: Path, base_experiment_names: List[str] = None, use_bert: bool = True, use_roberta: bool = True, use_transfer: bool = False, use_full_train_model_for_dev: bool = False, score_mode: str = None) -> Dict[str, Dict[Tuple[str, int], float]]:\n",
)
replace_once(
    data_loader,
    "        use_full_train_model_for_dev: devデータの場合、train全体でFTしたモデルを使用するか（デフォルト: False）\n                                      Trueの場合、_kfold5を削除した実験名を参照\n    \n    Returns:\n        ベーススコアの辞書 {prefix_logit_clarification: {key: score}}\n",
    "        use_full_train_model_for_dev: devデータの場合、train全体でFTしたモデルを使用するか（デフォルト: False）\n                                      Trueの場合、_kfold5を削除した実験名を参照\n        score_mode: PLMスコア方式。logit_difference（既定）またはpositive_logit\n    \n    Returns:\n        ベーススコアの辞書 {prefix_logit_clarification: {key: score}}\n",
)
replace_once(
    data_loader,
    "    base_scores = {}\n    \n    # base_experiment_namesが指定されていない場合は、configから読み込む\n",
    "    base_scores = {}\n    resolved_score_mode = resolve_plm_score_mode(score_mode)\n    \n    # base_experiment_namesが指定されていない場合は、configから読み込む\n",
)
replace_once(
    data_loader,
    '''                logit = turn.get('logit_clarification')
                if logit is not None:
                    scores[key] = float(logit)
''',
    '''                score = _get_plm_score(turn, resolved_score_mode)
                if score is not None:
                    scores[key] = score
''',
)
replace_once(
    data_loader,
    '        print(f"Loaded {len(scores)} {feature_name} scores for {split} (experiment: {exp_name})")\n',
    '        print(f"Loaded {len(scores)} {feature_name} scores for {split} (experiment: {exp_name}, mode: {resolved_score_mode})")\n',
)

models = "LogReg/LASSO/module/models.py"
replace_once(
    models,
    "from sklearn.model_selection import GridSearchCV, StratifiedKFold\n",
    "from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold\n",
)
random_forest_class = '''class RandomForestCVModel:
    """層化CVでハイパーパラメータを選択するRandomForest分類器。"""

    def __init__(self, cv=5, scoring='roc_auc', random_state=42,
                 class_weight='balanced', n_jobs=-1, n_iter=32,
                 param_distributions=None):
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.param_distributions = param_distributions
        self.model_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.best_train_score_ = None

    def _default_param_distributions(self):
        class_weights = [self.class_weight]
        if self.class_weight == 'balanced':
            class_weights.append('balanced_subsample')
        return {
            'n_estimators': [200, 500, 800],
            'max_depth': [3, 5, 8, 12, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', 0.5],
            'class_weight': class_weights,
        }

    def fit(self, X, y, print_and_save_func=None):
        labels, counts = np.unique(np.asarray(y), return_counts=True)
        if len(labels) < 2:
            raise ValueError("RandomForest tuning requires at least two classes")
        effective_cv = min(self.cv, int(counts.min()))
        if effective_cv < 2:
            raise ValueError("RandomForest tuning requires at least two samples in every class")

        param_distributions = self.param_distributions or self._default_param_distributions()
        total_candidates = int(np.prod([len(values) for values in param_distributions.values()]))
        n_iter = min(self.n_iter, total_candidates)
        cv = StratifiedKFold(n_splits=effective_cv, shuffle=True, random_state=self.random_state)
        base_model = RandomForestClassifier(
            random_state=self.random_state,
            class_weight=self.class_weight,
            n_jobs=1,
        )
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=self.scoring,
            cv=cv,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            refit=True,
            return_train_score=True,
            error_score='raise',
        )
        search.fit(X, y)

        self.search_ = search
        self.model_ = search.best_estimator_
        self.best_estimator_ = search.best_estimator_
        self.best_params_ = search.best_params_
        self.best_score_ = float(search.best_score_)
        self.best_train_score_ = float(search.cv_results_['mean_train_score'][search.best_index_])
        self.feature_importances_ = self.model_.feature_importances_
        self.classes_ = self.model_.classes_
        self.n_features_in_ = self.model_.n_features_in_
        if hasattr(self.model_, 'feature_names_in_'):
            self.feature_names_in_ = self.model_.feature_names_in_

        log = print_and_save_func or print
        log("RandomForest hyperparameter search completed:")
        log(f"  best params: {self.best_params_}")
        log(f"  best mean CV {self.scoring}: {self.best_score_:.4f}")
        log(f"  corresponding mean train score: {self.best_train_score_:.4f}")
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def score(self, X, y):
        return self.model_.score(X, y)


'''
replace_once(models, "def create_model(model_type, max_iter=1000", random_forest_class + "def create_model(model_type, max_iter=1000")
replace_once(
    models,
    '''    elif model_type == 'randomforest':
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
            class_weight='balanced',
            n_jobs=-1
        )
''',
    '''    elif model_type == 'randomforest':
        return RandomForestCVModel(
            cv=cv,
            scoring=scoring,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=n_jobs,
        )
''',
)

module_init = "LogReg/LASSO/module/__init__.py"
replace_once(
    module_init,
    "    LARSCVModel,\n    create_model,\n",
    "    LARSCVModel,\n    RandomForestCVModel,\n    create_model,\n",
)
replace_once(
    module_init,
    "    'LARSCVModel',\n    'create_model',\n",
    "    'LARSCVModel',\n    'RandomForestCVModel',\n    'create_model',\n",
)

rf_main = "LogReg/RandomForest/main.py"
replace_once(
    rf_main,
    "from typing import Dict, List, Tuple, Any\n\n\n# パス設定（main関数内で動的に設定）\n",
    "from typing import Dict, List, Tuple, Any\n\nLASSO_DIR = Path(__file__).resolve().parents[1] / 'LASSO'\nsys.path.insert(0, str(LASSO_DIR))\nfrom module.data_loader import extract_base_scores as extract_shared_base_scores\nfrom module.models import RandomForestCVModel\n\n\n# パス設定（main関数内で動的に設定）\n",
)
replace_once(
    rf_main,
    '''def extract_base_scores(json_data: List[List[Dict[str, Any]]]) -> Dict[Tuple[str, int], float]:
    """ベースモデルのlogit_clarificationを抽出"""
    scores = {}
    for conversation in json_data:
        for turn in conversation:
            # conv_idとturn_idを文字列/整数に統一
            conv_id = str(turn['conv_id'])
            turn_id = int(turn['turn_id'])
            key = (conv_id, turn_id)
            
            logit = turn.get('logit_clarification')
            if logit is not None:
                scores[key] = float(logit)
    
    return scores
''',
    '''def extract_base_scores(
    json_data: List[List[Dict[str, Any]]],
    score_mode: str = 'logit_difference',
) -> Dict[Tuple[str, int], float]:
    """共有ローダーを使ってPLMスコアを抽出する。"""
    return extract_shared_base_scores(json_data, score_mode=score_mode)
''',
)
replace_once(
    rf_main,
    '        help="ベーススコア（logit_clarification）も特徴量として使用する（デフォルト: False、QPPスコアのみ使用）"\n',
    '        help="PLMスコアも特徴量として使用する（デフォルト: False、QPPスコアのみ使用）"\n',
)
use_base_block = '''    parser.add_argument(
        "--use-base-score",
        action="store_true",
        help="PLMスコアも特徴量として使用する（デフォルト: False、QPPスコアのみ使用）"
    )
'''
replace_once(
    rf_main,
    use_base_block,
    use_base_block + '''
    parser.add_argument(
        "--plm-score-mode",
        type=str,
        default="logit_difference",
        choices=["logit_difference", "positive_logit"],
        help="PLMスコア方式（デフォルト: logit_difference=正例logit-負例logit。positive_logitで従来方式）"
    )
''',
)
replace_once(
    rf_main,
    '        help="決定木の数（デフォルト: 100）"\n',
    '        help="固定RFで使用する決定木数（--no-rf-search時のみ、デフォルト: 100）"\n',
)
replace_once(
    rf_main,
    '        help="決定木の最大深度（デフォルト: None、制限なし）"\n',
    '        help="固定RFで使用する最大深度（--no-rf-search時のみ、デフォルト: None）"\n',
)
max_depth_block = '''    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="固定RFで使用する最大深度（--no-rf-search時のみ、デフォルト: None）"
    )
'''
replace_once(
    rf_main,
    max_depth_block,
    max_depth_block + '''
    parser.add_argument(
        "--no-rf-search",
        action="store_false",
        dest="use_rf_search",
        default=True,
        help="RandomForestのハイパーパラメータ探索を無効化し、従来の固定設定を使用"
    )

    parser.add_argument(
        "--rf-search-iterations",
        type=int,
        default=32,
        help="RandomizedSearchCVの試行数（デフォルト: 32）"
    )

    parser.add_argument(
        "--rf-cv-folds",
        type=int,
        default=5,
        help="RandomForestハイパーパラメータ探索の層化CV fold数（デフォルト: 5）"
    )

    parser.add_argument(
        "--rf-scoring",
        type=str,
        default="roc_auc",
        choices=["roc_auc", "average_precision", "f1", "accuracy"],
        help="RandomForestハイパーパラメータ探索の評価指標（デフォルト: roc_auc）"
    )
''',
)
replace_once(
    rf_main,
    '    print_and_save(f"使用する特徴量: {\'ベーススコア + QPPスコア（post + nsp）\' if use_base_score else \'QPPスコア（post + nsp）\'}")\n    print_and_save(f"モデルパラメータ: n_estimators={args.n_estimators}, max_depth={args.max_depth}, random_state={args.random_state}\\n")\n',
    '    print_and_save(f"使用する特徴量: {\'PLMスコア + QPPスコア（post + nsp）\' if use_base_score else \'QPPスコア（post + nsp）\'}")\n    if use_base_score:\n        print_and_save(f"PLMスコア方式: {args.plm_score_mode}")\n    if args.use_rf_search:\n        print_and_save(f"RandomForest探索: RandomizedSearchCV (iterations={args.rf_search_iterations}, cv={args.rf_cv_folds}, scoring={args.rf_scoring}, random_state={args.random_state})\\n")\n    else:\n        print_and_save(f"RandomForest固定設定: n_estimators={args.n_estimators}, max_depth={args.max_depth}, random_state={args.random_state}\\n")\n',
)
replace_once(
    rf_main,
    "        train_base_scores = extract_base_scores(train_pred_data)\n",
    "        train_base_scores = extract_base_scores(train_pred_data, score_mode=args.plm_score_mode)\n",
)
replace_once(
    rf_main,
    "        dev_base_scores = extract_base_scores(dev_pred_data)\n",
    "        dev_base_scores = extract_base_scores(dev_pred_data, score_mode=args.plm_score_mode)\n",
)
replace_once(
    rf_main,
    '''    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        class_weight='balanced',
        n_jobs=-1  # 並列処理を有効化
    )
    
    model.fit(X_train_norm, y_train)
    print_and_save("  - 学習完了")
''',
    '''    if args.use_rf_search:
        model = RandomForestCVModel(
            cv=args.rf_cv_folds,
            scoring=args.rf_scoring,
            random_state=args.random_state,
            class_weight='balanced',
            n_jobs=-1,
            n_iter=args.rf_search_iterations,
        )
        model.fit(X_train_norm, y_train, print_and_save_func=print_and_save)
    else:
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        model.fit(X_train_norm, y_train)
    print_and_save("  - 学習完了")
''',
)

print("Applied DEIM PLM-score and RandomForest tuning changes")
