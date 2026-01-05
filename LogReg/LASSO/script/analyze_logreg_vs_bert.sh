#!/bin/bash
# LogRegが正解し、BERT/RoBERTaが外したクエリを分析するスクリプト
# 複数のバージョン（roberta, bert, bm25, pre post, pre, post）に対応

# プロジェクトのルートディレクトリ
BASE_DIR="/home/daiki_shibata/pj/QPP4SIP"
cd "$BASE_DIR" || exit

# データセット名
DATASET="AmbigNQ"

# データセットJSONのパス
DATASET_JSON="${BASE_DIR}/dataset/${DATASET}/dev.json"

# 分析を実行する関数
run_analysis() {
    local feature_type=$1      # "pre_post_bert", "pre_post", "pre", "post"
    local base_model=$2        # "RoBERTa", "BERT", または ""（ベースモデルなし）
    local retrieval_method=$3  # "dpr" または "bm25"
    
    # ディレクトリ名を構築
    local feature_dir=""
    case "$feature_type" in
        "pre_post_bert")
            if [ "$base_model" = "RoBERTa" ]; then
                feature_dir="ictf_idf_maxidf_scq_scs_clarity_ns50_nqc_smv_wig_rob"
            elif [ "$base_model" = "BERT" ]; then
                feature_dir="ictf_idf_maxidf_scq_scs_clarity_ns50_nqc_smv_wig_bert"
            else
                echo "エラー: pre_post_bertにはbase_modelが必要です"
                return 1
            fi
            ;;
        "pre_post")
            feature_dir="ictf_idf_maxidf_scq_scs_clarity_ns50_nqc_smv_wig"
            ;;
        "pre")
            feature_dir="ictf_idf_maxidf_scq_scs"
            ;;
        "post")
            feature_dir="clarity_ns50_nqc_smv_wig"
            ;;
        *)
            echo "エラー: 不明なfeature_type: $feature_type"
            return 1
            ;;
    esac
    
    # BM25の場合はサフィックスを追加
    if [ "$retrieval_method" = "bm25" ]; then
        feature_dir="${feature_dir}_bm25"
    fi
    
    # パスを構築
    local results_csv="${BASE_DIR}/LogReg/LASSO/outputs/${DATASET}/${feature_type}/${feature_dir}/csv/results.csv"
    local output_dir="${BASE_DIR}/LogReg/LASSO/outputs/${DATASET}/${feature_type}/${feature_dir}/analysis"
    
    # results.csvが存在するか確認
    if [ ! -f "$results_csv" ]; then
        echo "警告: results.csvが見つかりません: $results_csv"
        echo "スキップします"
        return 1
    fi
    
    # ベースモデルが存在するか確認（ベースモデルが必要な場合）
    if [ -n "$base_model" ]; then
        # results.csvにベースモデルのカラムが存在するか確認
        if ! grep -q "^[^,]*,$base_model," "$results_csv" 2>/dev/null && ! head -1 "$results_csv" | grep -q ",$base_model,"; then
            echo "警告: results.csvに${base_model}カラムが見つかりません: $results_csv"
            echo "スキップします"
            return 1
        fi
    fi
    
    echo ""
    echo "=========================================="
    echo "分析開始: ${feature_type} / ${base_model:-"なし"} / ${retrieval_method}"
    echo "=========================================="
    echo "results.csv: $results_csv"
    echo "output_dir: $output_dir"
    echo ""
    
    # 出力ディレクトリを作成
    mkdir -p "$output_dir"
    
    # ベースモデルがある場合のみ分析を実行
    if [ -n "$base_model" ]; then
        # スクリプトを実行
        python "${BASE_DIR}/LogReg/LASSO/analyze_model_differences.py" \
            --results-csv "$results_csv" \
            --dataset-json "$DATASET_JSON" \
            --output-dir "$output_dir" \
            --threshold 0.5 2>&1 | grep -E "(分析|完了|エラー|LogReg|${base_model})" || true
        
        # ベースモデル名を小文字に変換（bash互換性のため）
        local base_model_lower=$(echo "$base_model" | tr '[:upper:]' '[:lower:]')
        
        echo ""
        echo "=== 可視化を実行 ==="
        python "${BASE_DIR}/LogReg/LASSO/visualize_model_differences.py" \
            --results-csv "$results_csv" \
            --analysis-csv "${output_dir}/logreg_correct_${base_model_lower}_wrong.csv" \
            --output-dir "$output_dir" \
            --dataset-json "$DATASET_JSON" \
            --base-model "$base_model" 2>&1 | grep -E "(可視化|保存|エラー|完了)" || true
        
        echo ""
        echo "=== 高度な証拠分析を実行 ==="
        python "${BASE_DIR}/LogReg/LASSO/analyze_qpp_evidence.py" \
            --results-csv "$results_csv" \
            --dataset-json "$DATASET_JSON" \
            --output-dir "$output_dir" \
            --base-model "$base_model" \
            --analysis-csv "${output_dir}/logreg_correct_${base_model_lower}_wrong.csv" 2>&1 | grep -E "(保存|分析|エラー|完了)" || true
        
        echo ""
        echo "=== 高度なAUC分析を実行 ==="
        python "${BASE_DIR}/LogReg/LASSO/analyze_advanced_evidence.py" \
            --results-csv "$results_csv" \
            --output-dir "$output_dir" \
            --base-model "$base_model" \
            --qpp-feature "WIG" 2>&1 | grep -E "(保存|分析|エラー|完了|改善)" || true
        
        echo ""
        echo "=== 閾値に依存しないFP/FN分析を実行 ==="
        python "${BASE_DIR}/LogReg/LASSO/analyze_threshold_free_fp_fn.py" \
            --results-csv "$results_csv" \
            --output-dir "$output_dir" \
            --base-model "$base_model" \
            --method "roc" \
            --n-points 20 2>&1 | grep -E "(保存|分析|エラー|完了|FP|FN)" || true
        
        echo ""
        echo "=== ${base_model}が弱い部分の共通特徴を探索 ==="
        python "${BASE_DIR}/LogReg/LASSO/explore_roberta_weakness_patterns.py" \
            --results-csv "$results_csv" \
            --output-dir "$output_dir" \
            --base-model "$base_model" \
            --error-threshold 0.0 \
            --improvement-threshold 0.01 \
            --n-clusters 3 2>&1 | grep -E "(保存|分析|エラー|完了|弱い|クラスタ)" || true
        
        echo ""
        echo "=== 一般的な語の多さと${base_model}の性能の関係を可視化 ==="
        python "${BASE_DIR}/LogReg/LASSO/visualize_common_words_evidence.py" \
            --results-csv "$results_csv" \
            --output-dir "$output_dir" \
            --base-model "$base_model" 2>&1 | grep -E "(保存|可視化|エラー|完了|Plotting)" || true
        
        echo ""
        echo "=== ${base_model}が誤分類したがQPP統合で正解したケースを探索的に分析 ==="
        python "${BASE_DIR}/LogReg/LASSO/explore_qpp_success_cases.py" \
            --results-csv "$results_csv" \
            --output-dir "$output_dir" \
            --base-model "$base_model" \
            --n-thresholds 20 2>&1 | grep -E "(保存|分析|エラー|完了|成功|特徴|統計)" || true
        
        echo ""
        echo "=== 発表用：因果関係を示すグラフを作成 ==="
        python "${BASE_DIR}/LogReg/LASSO/plot_causal_evidence_for_presentation.py" \
            --results-csv "$results_csv" \
            --output-dir "$output_dir" \
            --base-model "$base_model" 2>&1 | grep -E "(保存|作成|エラー|完了|Saved)" || true
    else
        echo "ベースモデルがないため、ベースモデル比較分析をスキップします"
    fi
    
    echo ""
    echo "=== 分析完了: ${feature_type} / ${base_model:-"なし"} / ${retrieval_method} ==="
    echo "結果は以下に保存されました:"
    echo "  - ${output_dir}/"
    echo ""
}

# 各バージョンを実行
echo "=========================================="
echo "LogReg vs BERT/RoBERTa 分析スクリプト"
echo "=========================================="
echo ""

# pre_post_bert + RoBERTa + DPR
run_analysis "pre_post_bert" "RoBERTa" "dpr"

# pre_post_bert + RoBERTa + BM25
run_analysis "pre_post_bert" "RoBERTa" "bm25"

# pre_post_bert + BERT + DPR
run_analysis "pre_post_bert" "BERT" "dpr"

# pre_post_bert + BERT + BM25
run_analysis "pre_post_bert" "BERT" "bm25"

# pre_post + DPR
run_analysis "pre_post" "" "dpr"

# pre_post + BM25
run_analysis "pre_post" "" "bm25"

# pre + DPR (pre-retrievalは検索手法に依存しないため、DPRのみ実行)
run_analysis "pre" "" "dpr"

# post + DPR
run_analysis "post" "" "dpr"

# post + BM25
run_analysis "post" "" "bm25"

echo ""
echo "=========================================="
echo "全ての分析が完了しました"
echo "=========================================="
