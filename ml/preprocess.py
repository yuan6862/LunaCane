import argparse
import glob
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    MaxPooling1D,
)
from tensorflow.keras.models import Sequential


FEATURE_COLS = ["ax", "ay", "az", "gx", "gy", "gz", "acc_mag", "gyro_mag"]
RAW_SENSOR_COLS = ["ax", "ay", "az", "gx", "gy", "gz"]
LABEL_COL = "label"
FALL_LABEL = "fall"
STANDARD_LABELS = ("stand", "walk", "sit_down", "put_down", "tap", "fall")
ACC_MAG_INDEX = FEATURE_COLS.index("acc_mag")
GYRO_MAG_INDEX = FEATURE_COLS.index("gyro_mag")
SAMPLE_RATE_HZ = 50
DEFAULT_DECISION_THRESHOLD = 0.5


def normalize_label(value):
    label = str(value).strip().lower()
    aliases = {
        "sitdown": "sit_down",
        "sit-down": "sit_down",
        "putdown": "put_down",
        "put-down": "put_down",
    }
    return aliases.get(label, label)


def load_labeled_csvs(data_dir, fall_label=FALL_LABEL):
    """Load bmi_data_*.csv files that contain labels and prepare feature columns."""
    print(f"正在读取 {data_dir} 下的 CSV 数据...")
    all_files = sorted(glob.glob(os.path.join(data_dir, "bmi_data_*.csv")))
    df_list = []
    skipped = []

    for file_path in all_files:
        df = pd.read_csv(file_path)
        missing = [col for col in RAW_SENSOR_COLS + [LABEL_COL] if col not in df.columns]
        if missing:
            skipped.append({"file": file_path, "reason": f"missing columns: {missing}"})
            continue

        df = df.copy()
        df["source_file"] = os.path.basename(file_path)
        df[LABEL_COL] = df[LABEL_COL].map(normalize_label)
        for col in RAW_SENSOR_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if "esp_time_ms" in df.columns:
            df["esp_time_ms"] = pd.to_numeric(df["esp_time_ms"], errors="coerce")

        if "acc_mag" not in df.columns:
            df["acc_mag"] = np.sqrt(df["ax"] ** 2 + df["ay"] ** 2 + df["az"] ** 2)
        if "gyro_mag" not in df.columns:
            df["gyro_mag"] = np.sqrt(df["gx"] ** 2 + df["gy"] ** 2 + df["gz"] ** 2)

        df["numeric_label"] = (df[LABEL_COL] == fall_label).astype(int)
        df_list.append(df)

    if skipped:
        print("以下 CSV 未参与训练:")
        for item in skipped:
            print(f"  - {item['file']}: {item['reason']}")

    if not df_list:
        raise ValueError(f"在 {data_dir} 没有找到带 label 的 bmi_data_*.csv。请先采集标注数据。")

    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df = merged_df.dropna(subset=FEATURE_COLS + ["numeric_label"])
    if merged_df.empty:
        raise ValueError("所有数据行都存在空值或非法数值，无法训练。")

    _print_dataset_summary(merged_df)
    return merged_df


def _print_dataset_summary(df):
    total = len(df)
    fall_count = int(df["numeric_label"].sum())
    print(
        f"数据加载完成，共 {total} 行 | 跌倒帧: {fall_count} "
        f"({fall_count / total * 100:.1f}%) | 正常帧: {total - fall_count}"
    )
    print("标签分布:")
    print(df[LABEL_COL].value_counts().to_string())
    print("文件分布:")
    print(df["source_file"].value_counts().to_string())


def write_quality_report(df, output_path, fall_label=FALL_LABEL):
    """Write a lightweight data quality report for later experiment records."""
    label_counts = df[LABEL_COL].value_counts().to_dict()
    file_counts = df["source_file"].value_counts().to_dict()
    feature_stats = df[FEATURE_COLS].describe().to_dict()
    sampling_stats = build_sampling_quality(df)
    session_columns = [col for col in ("session_id", "participant", "mount_position", "note") if col in df.columns]

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rows": int(len(df)),
        "fall_rows": int(df["numeric_label"].sum()),
        "normal_rows": int(len(df) - df["numeric_label"].sum()),
        "label_counts": label_counts,
        "label_mapping": build_label_mapping(df, fall_label),
        "file_counts": file_counts,
        "feature_columns": FEATURE_COLS,
        "feature_stats": feature_stats,
        "sampling_quality": sampling_stats,
    }
    if session_columns:
        report["session_summary"] = (
            df[["source_file"] + session_columns]
            .drop_duplicates()
            .sort_values("source_file")
            .to_dict(orient="records")
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"数据质检报告已保存至: {output_path}")


def build_label_mapping(df, fall_label):
    observed_labels = sorted(df[LABEL_COL].dropna().unique().tolist())
    unknown_standard_labels = [label for label in observed_labels if label not in STANDARD_LABELS]
    return {
        "fall_label": fall_label,
        "positive_class": [fall_label],
        "normal_class": [label for label in observed_labels if label != fall_label],
        "standard_labels": STANDARD_LABELS,
        "non_standard_observed_labels": unknown_standard_labels,
    }


def build_sampling_quality(df):
    """Summarize ESP sampling interval stability when esp_time_ms is available."""
    if "esp_time_ms" not in df.columns:
        return {"available": False, "reason": "missing esp_time_ms column"}

    rows = []
    for source_file, group in df.groupby("source_file", sort=False):
        times = group["esp_time_ms"].dropna().astype(float).values
        if len(times) < 2:
            rows.append({"source_file": source_file, "available": False, "reason": "less than 2 timestamps"})
            continue

        intervals = np.diff(times)
        valid_intervals = intervals[intervals > 0]
        if len(valid_intervals) == 0:
            rows.append({"source_file": source_file, "available": False, "reason": "no positive intervals"})
            continue

        rows.append(
            {
                "source_file": source_file,
                "available": True,
                "sample_count": int(len(times)),
                "mean_interval_ms": float(np.mean(valid_intervals)),
                "median_interval_ms": float(np.median(valid_intervals)),
                "max_interval_ms": float(np.max(valid_intervals)),
                "estimated_hz": float(1000.0 / np.mean(valid_intervals)),
                "gap_count_over_60ms": int(np.sum(valid_intervals > 60)),
            }
        )
    return {"available": True, "files": rows}


def normalize_data(df, scaler_path, fit=True):
    """Standardize sensor features and export scaler parameters for ESP32 inference."""
    sensor_data = df[FEATURE_COLS].values

    if fit:
        scaler = StandardScaler()
        normalized = scaler.fit_transform(sensor_data)
        joblib.dump(scaler, scaler_path)
        _export_scaler_to_header(scaler, os.path.dirname(scaler_path))
        print(f"Scaler 已保存至: {scaler_path}")
    else:
        scaler = joblib.load(scaler_path)
        normalized = scaler.transform(sensor_data)

    normalized_df = df.copy()
    normalized_df[FEATURE_COLS] = normalized
    return normalized_df


def fit_transform_window_scaler(X_train, X_test, scaler_path):
    """Fit scaler on training windows only, then transform train and test windows."""
    scaler = StandardScaler()
    feature_count = X_train.shape[-1]

    X_train_flat = X_train.reshape(-1, feature_count)
    X_test_flat = X_test.reshape(-1, feature_count)

    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape).astype(np.float32)

    joblib.dump(scaler, scaler_path)
    _export_scaler_to_header(scaler, os.path.dirname(scaler_path))
    print(f"Scaler 已保存至: {scaler_path}")
    return X_train_scaled, X_test_scaled


def _export_scaler_to_header(scaler, output_dir):
    """Export scaler parameters as a C header for ESP32 inference."""
    lines = [
        "// Auto-generated by ml/preprocess.py. Do not edit manually.",
        "// Usage: normalized = (raw_value - FEATURE_MEANS[i]) / FEATURE_STDS[i]",
        "#ifndef SCALER_PARAMS_H",
        "#define SCALER_PARAMS_H",
        "",
        f"const float FEATURE_MEANS[8] = {{{', '.join(f'{v:.6f}f' for v in scaler.mean_)}}};",
        f"const float FEATURE_STDS[8]  = {{{', '.join(f'{v:.6f}f' for v in scaler.scale_)}}};",
        "",
        "// Feature order: " + ", ".join(FEATURE_COLS),
        "#endif",
    ]
    header_path = os.path.join(output_dir, "scaler_params.h")
    with open(header_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"ESP32 归一化头文件已导出至: {header_path}")


def write_deployment_metadata(output_dir, window_size, step_size, threshold, positive_label=FALL_LABEL):
    """Export the preprocessing and decision contract consumed by firmware."""
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_file": "lunacane_model.tflite",
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "window_size": int(window_size),
        "step_size": int(step_size),
        "feature_columns": FEATURE_COLS,
        "input_shape": [1, int(window_size), len(FEATURE_COLS)],
        "output_shape": [1, 1],
        "decision_threshold": float(threshold),
        "positive_label": normalize_label(positive_label),
        "normalization": "standard_scaler",
        "scaler_header": "scaler_params.h",
    }
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "model_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    header = [
        "// Auto-generated by ml/preprocess.py. Do not edit manually.",
        "#ifndef LUNACANE_MODEL_CONFIG_H",
        "#define LUNACANE_MODEL_CONFIG_H",
        "",
        f"constexpr int MODEL_SAMPLE_RATE_HZ = {SAMPLE_RATE_HZ};",
        f"constexpr int MODEL_WINDOW_SIZE = {int(window_size)};",
        f"constexpr int MODEL_STEP_SIZE = {int(step_size)};",
        f"constexpr int MODEL_FEATURE_COUNT = {len(FEATURE_COLS)};",
        f"constexpr float MODEL_FALL_THRESHOLD = {float(threshold):.6f}f;",
        "",
        "// Feature order: " + ", ".join(FEATURE_COLS),
        "#endif",
    ]
    header_path = os.path.join(output_dir, "model_config.h")
    with open(header_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(header))
    print(f"部署元数据已保存至: {metadata_path}")
    print(f"ESP32 模型配置已保存至: {header_path}")
    return metadata


def create_sliding_windows(df, window_size=100, step_size=50, fall_ratio_threshold=0.5):
    """
    Build windows per source file so windows never cross recording boundaries.

    At 50Hz, window_size=100 means a 2 second window, and step_size=50 means 50%
    overlap. A window is labeled as fall only when enough frames inside it are fall.
    """
    print(
        "开始滑动窗口切割... "
        f"(窗口大小: {window_size}, 步长: {step_size}, 跌倒阈值: {fall_ratio_threshold * 100:.0f}%)"
    )

    features, labels, groups = [], [], []
    for source_file, group in df.groupby("source_file", sort=False):
        sensor_data = group[FEATURE_COLS].values
        label_data = group["numeric_label"].values

        if len(group) < window_size:
            print(f"跳过 {source_file}: 行数 {len(group)} 小于窗口大小 {window_size}")
            continue

        for i in range(0, len(group) - window_size + 1, step_size):
            window_features = sensor_data[i : i + window_size]
            fall_ratio = np.mean(label_data[i : i + window_size])
            window_label = 1 if fall_ratio >= fall_ratio_threshold else 0
            features.append(window_features)
            labels.append(window_label)
            groups.append(source_file)

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups)

    if len(y) == 0:
        raise ValueError("没有生成任何训练窗口，请增加数据量或减小 window_size。")

    fall_windows = int(y.sum())
    print(
        f"切割完毕，共 {len(y)} 个窗口 | 跌倒窗口: {fall_windows} "
        f"({fall_windows / len(y) * 100:.1f}%) | 正常窗口: {len(y) - fall_windows}"
    )
    return X, y, groups


def build_tinyml_cnn(input_shape):
    """
    Lightweight 1D-CNN for ESP32-oriented fall detection.

    This is the default deployment candidate because it learns temporal motion
    patterns directly from IMU windows and exports cleanly to TFLite.
    """
    model = Sequential(
        [
            Input(shape=input_shape),
            Conv1D(filters=16, kernel_size=5, padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"),
            GlobalAveragePooling1D(),
            Dense(32, activation="relu"),
            Dropout(0.4),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
        ],
    )
    return model


def build_statistical_window_features(X):
    """Create compact window statistics for a classical baseline model."""
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    min_value = X.min(axis=1)
    max_value = X.max(axis=1)
    peak_to_peak = max_value - min_value
    return np.concatenate([mean, std, min_value, max_value, peak_to_peak], axis=1)


def build_binary_classification_metrics(y_true, y_pred):
    """Return a stable two-class report even when one class is absent."""
    labels = [0, 1]
    target_names = ["正常动作", "跌倒"]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    printable_report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )
    return report, matrix, printable_report


def train_random_forest_baseline(X_train, y_train, X_test, y_test, output_dir):
    """Train a classical baseline for comparison; it is not the deployment target."""
    baseline = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    baseline.fit(build_statistical_window_features(X_train), y_train)
    y_pred = baseline.predict(build_statistical_window_features(X_test))

    report, matrix, printable_report = build_binary_classification_metrics(y_test, y_pred)
    result = {"model": "random_forest_baseline", "confusion_matrix": matrix, "classification_report": report}

    model_path = os.path.join(output_dir, "random_forest_baseline.pkl")
    report_path = os.path.join(output_dir, "random_forest_report.json")
    joblib.dump(baseline, model_path)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\nRandomForest 对照模型:")
    print("混淆矩阵:")
    print(np.asarray(matrix))
    print(printable_report)
    print(f"RandomForest 对照模型已保存至: {model_path}")
    print(f"RandomForest 评估报告已保存至: {report_path}")
    return result


def predict_rule_baseline(X, impact_acc_threshold, gyro_threshold, quiet_gyro_threshold):
    """
    Interpretable fall rule for comparison:
    a fall-like window should contain impact, strong rotation, and a quiet tail.
    """
    acc_mag = X[:, :, ACC_MAG_INDEX]
    gyro_mag = X[:, :, GYRO_MAG_INDEX]
    tail_len = max(1, X.shape[1] // 4)

    impact = acc_mag.max(axis=1) >= impact_acc_threshold
    rotation = gyro_mag.max(axis=1) >= gyro_threshold
    quiet_tail = gyro_mag[:, -tail_len:].mean(axis=1) <= quiet_gyro_threshold
    return (impact & rotation & quiet_tail).astype(int)


def evaluate_rule_baseline(X_test, y_test, output_dir, impact_acc_threshold, gyro_threshold, quiet_gyro_threshold):
    y_pred = predict_rule_baseline(X_test, impact_acc_threshold, gyro_threshold, quiet_gyro_threshold)
    report, matrix, printable_report = build_binary_classification_metrics(y_test, y_pred)
    result = {
        "model": "interpretable_rule_baseline",
        "description": "impact + rotation + quiet tail",
        "thresholds": {
            "impact_acc_mag": impact_acc_threshold,
            "gyro_mag": gyro_threshold,
            "quiet_tail_gyro_mean": quiet_gyro_threshold,
        },
        "confusion_matrix": matrix,
        "classification_report": report,
    }

    report_path = os.path.join(output_dir, "rule_baseline_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n规则基线测试集表现:")
    print("规则: acc_mag 冲击 + gyro_mag 旋转 + 窗口尾部趋于安静")
    print("混淆矩阵:")
    print(np.asarray(matrix))
    print(printable_report)
    print(f"规则基线评估报告已保存至: {report_path}")
    return result


def choose_recall_weighted_threshold(y_true, y_prob):
    """Choose a CNN decision threshold with F2 so fall recall is weighted higher."""
    candidates = np.round(np.arange(0.20, 0.81, 0.05), 2)
    best = None
    candidate_rows = []

    for threshold in candidates:
        y_pred = (y_prob >= threshold).astype(int)
        precision, recall, f2, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            beta=2.0,
            labels=[1],
            average="binary",
            zero_division=0,
        )
        row = {
            "threshold": float(threshold),
            "fall_precision": float(precision),
            "fall_recall": float(recall),
            "fall_f2": float(f2),
        }
        candidate_rows.append(row)
        if best is None or (row["fall_f2"], row["fall_recall"]) > (best["fall_f2"], best["fall_recall"]):
            best = row

    return best["threshold"], candidate_rows


def train_cnn_model(X_train, y_train, X_test, y_test, output_dir, epochs, batch_size):
    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"类别权重: 正常={class_weights[0]:.2f}, 跌倒={class_weights[1]:.2f}")

    model = build_tinyml_cnn((X_train.shape[1], X_train.shape[2]))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", patience=8, restore_best_weights=True, mode="max"),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]

    print("\n开始训练 1D-CNN...")
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        class_weight=class_weight_dict,
        callbacks=callbacks,
    )

    train_pred_prob = model.predict(X_train).flatten()
    selected_threshold, threshold_candidates = choose_recall_weighted_threshold(y_train, train_pred_prob)
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob >= selected_threshold).astype(int)
    report, matrix, printable_report = build_binary_classification_metrics(y_test, y_pred)

    print("\n1D-CNN 测试集表现:")
    print("混淆矩阵:")
    print(np.asarray(matrix))
    print(printable_report)

    keras_model_path = os.path.join(output_dir, "lunacane_model.keras")
    tflite_model_path = os.path.join(output_dir, "lunacane_model.tflite")
    report_path = os.path.join(output_dir, "cnn_report.json")

    model.save(keras_model_path)
    convert_to_tflite(model, tflite_model_path)

    result = {
        "model": "tinyml_1d_cnn",
        "threshold": selected_threshold,
        "threshold_policy": "selected on training windows by fall-class F2 score",
        "threshold_candidates": threshold_candidates,
        "confusion_matrix": matrix,
        "classification_report": report,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Keras 模型已保存至: {keras_model_path}")
    print(f"1D-CNN 判定阈值: {selected_threshold:.2f} (按跌倒类别 F2 选择，优先召回)")
    print(f"1D-CNN 评估报告已保存至: {report_path}")
    return result


def convert_to_tflite(model, save_path):
    """Convert Keras model to dynamically quantized TensorFlow Lite."""
    print("\n正在将模型转换为 TensorFlow Lite 格式...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(save_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite 模型已保存至: {save_path} (大小: {len(tflite_model) / 1024:.2f} KB)")


def split_windows(X, y, groups, test_size):
    if len(np.unique(y)) < 2:
        raise ValueError("窗口标签只有一个类别。至少需要正常动作和跌倒两类数据才能训练。")

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError(
            "至少需要 2 个采集文件才能按 source_file 划分训练集和测试集。"
            f"当前采集文件数={len(unique_groups)}。"
        )

    splitter = GroupShuffleSplit(n_splits=50, test_size=test_size, random_state=42)
    fallback_split = None
    for train_idx, test_idx in splitter.split(X, y, groups):
        if fallback_split is None:
            fallback_split = (train_idx, test_idx)

        train_labels = np.unique(y[train_idx])
        test_labels = np.unique(y[test_idx])
        if len(train_labels) == 2 and len(test_labels) == 2:
            return (
                X[train_idx],
                X[test_idx],
                y[train_idx],
                y[test_idx],
                groups[train_idx],
                groups[test_idx],
            )

    train_idx, test_idx = fallback_split
    if len(np.unique(y[train_idx])) < 2:
        raise ValueError(
            "按采集文件分组后，训练集缺少正常或跌倒类别。"
            "请增加采集文件数量，确保正常和跌倒数据分布在多个文件中。"
        )

    print("警告: 按 source_file 划分后测试集没有同时包含正常和跌倒类别，请补充更多采集文件。")
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx], groups[train_idx], groups[test_idx]


def run_training(args):
    os.makedirs(args.output_dir, exist_ok=True)
    fall_label = normalize_label(args.fall_label)

    df = load_labeled_csvs(args.data_dir, fall_label=fall_label)
    write_quality_report(df, os.path.join(args.output_dir, "data_quality_report.json"), fall_label=fall_label)
    X, y, groups = create_sliding_windows(
        df,
        window_size=args.window_size,
        step_size=args.step_size,
        fall_ratio_threshold=args.fall_ratio_threshold,
    )

    X_train, X_test, y_train, y_test, train_groups, test_groups = split_windows(X, y, groups, args.test_size)
    X_train_raw, X_test_raw = X_train.copy(), X_test.copy()
    X_train, X_test = fit_transform_window_scaler(X_train, X_test, os.path.join(args.output_dir, "scaler.pkl"))
    print(f"训练集: {len(X_train)} 窗口，测试集: {len(X_test)} 窗口")
    print(f"训练文件数: {len(np.unique(train_groups))}，测试文件数: {len(np.unique(test_groups))}")

    results = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "window_size": args.window_size,
        "step_size": args.step_size,
        "fall_ratio_threshold": args.fall_ratio_threshold,
        "label_mapping": build_label_mapping(df, fall_label),
        "test_size": args.test_size,
        "train_groups": sorted(np.unique(train_groups).tolist()),
        "test_groups": sorted(np.unique(test_groups).tolist()),
    }

    if args.baseline:
        results["random_forest"] = train_random_forest_baseline(X_train, y_train, X_test, y_test, args.output_dir)

    results["rule_baseline"] = evaluate_rule_baseline(
        X_test_raw,
        y_test,
        args.output_dir,
        args.rule_impact_acc,
        args.rule_gyro,
        args.rule_quiet_gyro,
    )
    results["cnn"] = train_cnn_model(X_train, y_train, X_test, y_test, args.output_dir, args.epochs, args.batch_size)
    write_deployment_metadata(
        args.output_dir,
        args.window_size,
        args.step_size,
        results["cnn"]["threshold"],
        positive_label=fall_label,
    )

    summary_path = os.path.join(args.output_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n训练汇总已保存至: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train LunaCane fall detection models.")
    parser.add_argument("--data-dir", default="../data/raw", help="Directory containing bmi_data_*.csv files.")
    parser.add_argument("--output-dir", default="../models", help="Directory for model and report outputs.")
    parser.add_argument("--window-size", type=int, default=100, help="Number of samples per window. 100 = 2s at 50Hz.")
    parser.add_argument("--step-size", type=int, default=50, help="Sliding window step. 50 = 50%% overlap at 50Hz.")
    parser.add_argument("--fall-ratio-threshold", type=float, default=0.5, help="Fall frame ratio needed for a fall window.")
    parser.add_argument("--fall-label", default=FALL_LABEL, help="CSV label that should be treated as the fall class.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum CNN training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="CNN batch size.")
    parser.add_argument("--rule-impact-acc", type=float, default=2.5, help="Rule baseline impact threshold on acc_mag.")
    parser.add_argument("--rule-gyro", type=float, default=180.0, help="Rule baseline rotation threshold on gyro_mag.")
    parser.add_argument("--rule-quiet-gyro", type=float, default=30.0, help="Rule baseline quiet-tail gyro mean threshold.")
    parser.add_argument("--no-baseline", action="store_true", help="Skip RandomForest comparison model.")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    cli_args.baseline = not cli_args.no_baseline

    try:
        run_training(cli_args)
        print("\n训练完成。")
    except Exception as exc:
        import traceback

        print(f"\n运行出错: {exc}")
        traceback.print_exc()
        raise
