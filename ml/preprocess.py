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
from sklearn.metrics import classification_report, confusion_matrix
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


def load_labeled_csvs(data_dir):
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
        df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip().str.lower()
        for col in RAW_SENSOR_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if "acc_mag" not in df.columns:
            df["acc_mag"] = np.sqrt(df["ax"] ** 2 + df["ay"] ** 2 + df["az"] ** 2)
        if "gyro_mag" not in df.columns:
            df["gyro_mag"] = np.sqrt(df["gx"] ** 2 + df["gy"] ** 2 + df["gz"] ** 2)

        df["numeric_label"] = (df[LABEL_COL] == FALL_LABEL).astype(int)
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


def write_quality_report(df, output_path):
    """Write a lightweight data quality report for later experiment records."""
    label_counts = df[LABEL_COL].value_counts().to_dict()
    file_counts = df["source_file"].value_counts().to_dict()
    feature_stats = df[FEATURE_COLS].describe().to_dict()

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rows": int(len(df)),
        "fall_rows": int(df["numeric_label"].sum()),
        "normal_rows": int(len(df) - df["numeric_label"].sum()),
        "label_counts": label_counts,
        "file_counts": file_counts,
        "feature_columns": FEATURE_COLS,
        "feature_stats": feature_stats,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"数据质检报告已保存至: {output_path}")


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

    report = classification_report(
        y_test,
        y_pred,
        target_names=["正常动作", "跌倒"],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, y_pred).tolist()
    result = {"model": "random_forest_baseline", "confusion_matrix": matrix, "classification_report": report}

    model_path = os.path.join(output_dir, "random_forest_baseline.pkl")
    report_path = os.path.join(output_dir, "random_forest_report.json")
    joblib.dump(baseline, model_path)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\nRandomForest 对照模型:")
    print("混淆矩阵:")
    print(np.asarray(matrix))
    print(classification_report(y_test, y_pred, target_names=["正常动作", "跌倒"], zero_division=0))
    print(f"RandomForest 对照模型已保存至: {model_path}")
    print(f"RandomForest 评估报告已保存至: {report_path}")
    return result


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

    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    matrix = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(
        y_test,
        y_pred,
        target_names=["正常动作", "跌倒"],
        output_dict=True,
        zero_division=0,
    )

    print("\n1D-CNN 测试集表现:")
    print("混淆矩阵:")
    print(np.asarray(matrix))
    print(classification_report(y_test, y_pred, target_names=["正常动作", "跌倒"], zero_division=0))

    keras_model_path = os.path.join(output_dir, "lunacane_model.keras")
    tflite_model_path = os.path.join(output_dir, "lunacane_model.tflite")
    report_path = os.path.join(output_dir, "cnn_report.json")

    model.save(keras_model_path)
    convert_to_tflite(model, tflite_model_path)

    result = {
        "model": "tinyml_1d_cnn",
        "threshold": 0.5,
        "confusion_matrix": matrix,
        "classification_report": report,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Keras 模型已保存至: {keras_model_path}")
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

    df = load_labeled_csvs(args.data_dir)
    write_quality_report(df, os.path.join(args.output_dir, "data_quality_report.json"))
    X, y, groups = create_sliding_windows(
        df,
        window_size=args.window_size,
        step_size=args.step_size,
        fall_ratio_threshold=args.fall_ratio_threshold,
    )

    X_train, X_test, y_train, y_test, train_groups, test_groups = split_windows(X, y, groups, args.test_size)
    X_train, X_test = fit_transform_window_scaler(X_train, X_test, os.path.join(args.output_dir, "scaler.pkl"))
    print(f"训练集: {len(X_train)} 窗口，测试集: {len(X_test)} 窗口")
    print(f"训练文件数: {len(np.unique(train_groups))}，测试文件数: {len(np.unique(test_groups))}")

    results = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "window_size": args.window_size,
        "step_size": args.step_size,
        "fall_ratio_threshold": args.fall_ratio_threshold,
        "test_size": args.test_size,
        "train_groups": sorted(np.unique(train_groups).tolist()),
        "test_groups": sorted(np.unique(test_groups).tolist()),
    }

    if args.baseline:
        results["random_forest"] = train_random_forest_baseline(X_train, y_train, X_test, y_test, args.output_dir)

    results["cnn"] = train_cnn_model(X_train, y_train, X_test, y_test, args.output_dir, args.epochs, args.batch_size)

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
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum CNN training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="CNN batch size.")
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
