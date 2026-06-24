import logging
import re
from typing import List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def get_flexible_col(df: pd.DataFrame, possible_names: List[str]) -> pd.Series:
    """健壮的列名查找器，兼容多表合并后的后缀"""
    for name in possible_names:
        if name in df.columns:
            return df[name]
        for col in df.columns:
            if str(col).lower().strip().startswith(name.lower().strip()):
                return df[col]
    return pd.Series(name=possible_names[0], index=df.index)

def one_hot_terms(series: pd.Series, prefix: str, target_index: pd.Index) -> pd.DataFrame:
    """
    终极修复版：用强大的正则搜索器，精准捞取文本中隐藏的 EC 数字
    """
    terms = series.fillna("None").astype(str)
    rows = []
    
    MAX_EC_LEVEL = 3  # 截取到前三位

    for value in terms:
        if value == "None" or not value.strip() or "unknown" in value.lower():
            rows.append({})
            continue
            
        tokens = []
        if prefix.lower() == "ec":
            # 1. 兼容各种分隔符切开多属性
            raw_splits = re.split(r'[;,/\s]+', value)
            for item in raw_splits:
                # 2. 【核心修复】：直接在字符串里“搜寻”符合 X.X.X.X 格式的数字片段
                # \d+(?:\.\d+)+ 匹配类似 7.1 或 1.1.1.45 这样的纯数字带点的结构
                match = re.search(r'\d+(?:\.\d+)+', item)
                if match:
                    clean_item = match.group(0) # 拿到了干净的 "7.1" 或 "7.1.1.2"
                    parts = clean_item.split('.')
                    # 3. 按层级截断
                    truncated_parts = parts[:MAX_EC_LEVEL]
                    truncated_ec = ".".join(truncated_parts)
                    tokens.append(truncated_ec)
        else:
            # 其他非 EC 特征保持原样
            if "." in value and ";" not in value:
                tokens = [t.strip() for t in value.split() if t.strip()]
            else:
                tokens = [t.strip() for t in value.split(";") if t.strip()]
        
        # 构建当前行的特征字典
        row = {f"{prefix}_EC:{term}" if prefix.lower() == "ec" else f"{prefix}_{term}": 1 
               for term in set(tokens) if term}
        rows.append(row)
        
    feature_df = pd.DataFrame(rows, index=target_index).fillna(0).astype(int)
    return feature_df

def bin_redox_potential(values: pd.Series) -> pd.Series:
    bins = [-np.inf, -300, -100, 0, 100, np.inf]
    labels = ["very_reductive", "reductive", "near_neutral", "oxidative", "very_oxidative"]
    return pd.cut(values, bins=bins, labels=labels).astype(str).fillna("Unknown")

def filter_top_terms(series: pd.Series, prefix: str, target_index: pd.Index, top_n: int = 50) -> pd.DataFrame:
    terms = series.fillna("None").astype(str)
    term_counts = {}
    for value in terms:
        tokens = [t.strip() for t in value.split(";") if t.strip()]
        for term in tokens:
            if term and term != "None":
                term_counts[term] = term_counts.get(term, 0) + 1

    top_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_term_set = {term for term, _ in top_terms}

    rows = []
    for value in terms:
        tokens = [t.strip() for t in value.split(";") if t.strip()]
        row = {
            f"{prefix}_{term}": 1
            for term in tokens
            if term in top_term_set and term != "None"
        }
        rows.append(row)

    # 【核心修复】显式传入 target_index
    feature_df = pd.DataFrame(rows, index=target_index).fillna(0).astype(int)
    logger.info(f"Extracted top {min(len(top_term_set), top_n)} terms for '{prefix}'")
    return feature_df

def build_functional_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Building functional features with index preservation")
    feature_frames = []
    
    # 抓取弹性列
    ec_series = get_flexible_col(df, ["EC number", "EC Number"])
    catalytic_series = get_flexible_col(df, ["Catalytic activity", "Catalytic"])
    cofactor_series = get_flexible_col(df, ["Cofactor"])
    redox_series = get_flexible_col(df, ["Redox potential", "Redox"])

# ─── 👈 在这里强行插入你的【高光探针】 ───
    print("\n" + "="*50)
    print(f"【DEBUG 探针】当前程序实际抓取到的 EC 列名是: '{ec_series.name}'")
    print(f"【DEBUG 探针】总表 df 里面所有包含 'ec' 的列名有: {[c for c in df.columns if 'ec' in str(c).lower()]}")
    print(f"【DEBUG 探针】该列前 15 行的原始数据切片为:")
    print(df[ec_series.name].head(15))
    print("="*50 + "\n")
    # ─────────────────────────────────────────

    # 1. EC Number 特征
    if ec_series.notna().any():
        ec_df = one_hot_terms(ec_series, prefix="ec", target_index=df.index)
        feature_frames.append(ec_df)
        logger.info(f"Added EC features with shape {ec_df.shape}")
        # DEBUG: ec_df columns right after one_hot_terms
        print(f"[DEBUG one_hot_terms→ec_df] {ec_df.shape[1]} cols: {sorted(ec_df.columns.tolist())[:15]}")

    # 2. Catalytic Activity 特征
    if catalytic_series.notna().any():
        catalysis_df = filter_top_terms(catalytic_series, prefix="catalytic", target_index=df.index, top_n=50)
        feature_frames.append(catalysis_df)

    # 3. Cofactor 特征
    if cofactor_series.notna().any():
        cofactor_df = filter_top_terms(cofactor_series, prefix="cofactor", target_index=df.index, top_n=30)
        feature_frames.append(cofactor_df)

    # 4. Redox Potential 特征
    if redox_series.notna().any():
        numeric_redox = pd.to_numeric(redox_series, errors="coerce")
        missing_rate = numeric_redox.isna().mean()
        low_info_flag = int(missing_rate > 0.9)
        
        redox_low_df = pd.DataFrame(
            {"redox_low_information": [low_info_flag] * len(df)}, index=df.index
        )
        feature_frames.append(redox_low_df)

        redox_bins = bin_redox_potential(numeric_redox)
        # 强制保持索引
        redox_bins.index = df.index
        redox_df = one_hot_terms(redox_bins, prefix="redox", target_index=df.index)
        feature_frames.append(redox_df)
# 1. 正常合并所有生成的特征帧
    if feature_frames:
        all_features = pd.concat(feature_frames, axis=1)
    else:
        all_features = pd.DataFrame(index=df.index)
        
    all_features = all_features.fillna(0).astype(int)
    
    # 2. ─── 🧽 【新增：全表 EC 列终极净化与聚合逻辑】 ───
    logger.info("Starting global EC column normalization and aggregation...")

    # DEBUG: all_features before EC aggregation
    ec_before = [c for c in all_features.columns if str(c).lower().startswith("ec")]
    dirty_before = [c for c in ec_before if "EC:" not in str(c) or " " in str(c)]
    print(f"[DEBUG pre-aggregation] EC cols: {len(ec_before)}, dirty: {len(dirty_before)}")
    if dirty_before:
        print(f"[DEBUG pre-aggregation] dirty samples: {dirty_before[:5]}")

    ec_cols = [col for col in all_features.columns if str(col).lower().startswith("ec")]
    non_ec_cols = [col for col in all_features.columns if not str(col).lower().startswith("ec")]
    
    # 用一个字典来归拢合并后的数据 { "7.1.1": pd.Series }
    aggregated_ec_data = {}
    
    for col in ec_cols:
        # 去除列名里的一切空格
        col_clean = str(col).replace(" ", "")
        # 提取里面隐藏的 X.X.X 数字核心
        match = re.search(r'\d+(?:\.\d+)+', col_clean)
        
        if match:
            core_ec = match.group(0)
            standard_col_name = f"ec_EC:{core_ec}"
            
            if standard_col_name not in aggregated_ec_data:
                aggregated_ec_data[standard_col_name] = all_features[col].copy()
            else:
                # 【核心】：如果是由于空格、脏后缀拆开的同一类酶，用逻辑或 (max) 强行合并！
                aggregated_ec_data[standard_col_name] = np.maximum(aggregated_ec_data[standard_col_name], all_features[col])
        else:
            # 如果是 ec_Unknown 或者没匹配到数字的，统一收拢到特殊的未知列
            if "unknown" in col_clean.lower() or "none" in col_clean.lower():
                if "ec_EC:Unknown" not in aggregated_ec_data:
                    aggregated_ec_data["ec_EC:Unknown"] = all_features[col].copy()
                else:
                    aggregated_ec_data["ec_EC:Unknown"] = np.maximum(aggregated_ec_data["ec_EC:Unknown"], all_features[col])

    # 将清洗规整后的 EC 特征转化为 DataFrame
    if aggregated_ec_data:
        clean_ec_df = pd.DataFrame(aggregated_ec_data, index=df.index)
        # 按列名自然排序，让表头排得整整齐齐（如 1.1.1 在前，7.1.1 在后）
        clean_ec_df = clean_ec_df.reindex(columns=sorted(clean_ec_df.columns))
        # 与其他非 EC 特征（如 catalytic, cofactor）拼接
        all_features = pd.concat([clean_ec_df, all_features[non_ec_cols]], axis=1)
    
    # DEBUG: final EC columns after aggregation
    ec_after = [c for c in all_features.columns if str(c).lower().startswith("ec")]
    dirty_after = [c for c in ec_after if "EC:" not in str(c) or " " in str(c)]
    print(f"[DEBUG post-aggregation] EC cols: {len(ec_after)}, dirty: {len(dirty_after)}")
    if dirty_after:
        print(f"[DEBUG post-aggregation] dirty samples: {dirty_after[:5]}")
    print(f"[DEBUG post-aggregation] clean EC sample: {sorted(ec_after)[:5]}")

    logger.info(f"Final safe functional feature matrix shape {all_features.shape}")
    return all_features