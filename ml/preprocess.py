import os
import glob
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, Input, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix


# ==========================================
# 1. 数据加载与预处理模块 (适配硬件组员的数据格式)
# ==========================================
def load_and_merge_csvs(data_dir):
    """读取目录下所有队友生成的 bmi_data_*.csv 并合并"""
    print(f"正在读取 {data_dir} 下的 CSV 数据...")
    all_files = glob.glob(os.path.join(data_dir, "bmi_data_*.csv"))
    df_list = []

    for file in all_files:
        df = pd.read_csv(file)
        df_list.append(df)

    if not df_list:
        raise ValueError(f"在 {data_dir} 目录下没有找到任何 CSV 文件！请让组员先发数据过来。")

    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df['numeric_label'] = merged_df['label'].apply(lambda x: 1 if str(x).strip().lower() == 'fall' else 0)

    fall_count = merged_df['numeric_label'].sum()
    total = len(merged_df)
    print(f"数据加载完成，共 {total} 行 | 跌倒帧: {fall_count} ({fall_count/total*100:.1f}%) | 正常帧: {total-fall_count}")
    return merged_df


def normalize_data(df, scaler_path, fit=True):
    """对传感器数据做标准化，并保存 scaler 供 ESP32 推理时使用"""
    feature_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'acc_mag', 'gyro_mag']
    sensor_data = df[feature_cols].values

    scaler = StandardScaler()
    if fit:
        normalized = scaler.fit_transform(sensor_data)
        joblib.dump(scaler, scaler_path)
        # 同时导出均值和标准差为 C 头文件，方便硬件组移植到 ESP32
        _export_scaler_to_header(scaler, os.path.dirname(scaler_path))
        print(f"Scaler 已保存至: {scaler_path}")
    else:
        scaler = joblib.load(scaler_path)
        normalized = scaler.transform(sensor_data)

    df = df.copy()
    df[feature_cols] = normalized
    return df


def _export_scaler_to_header(scaler, output_dir):
    """将 scaler 参数导出为 C 头文件，硬件组直接复制到 ESP32 工程中使用"""
    means = scaler.mean_
    stds = scaler.scale_
    feature_names = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'acc_mag', 'gyro_mag']

    lines = [
        "// 自动生成，勿手动修改。由 preprocess.py 导出。",
        "// 使用方法: normalized = (raw_value - MEAN[i]) / STD[i]",
        "#ifndef SCALER_PARAMS_H",
        "#define SCALER_PARAMS_H",
        "",
        f"const float FEATURE_MEANS[8] = {{{', '.join(f'{v:.6f}f' for v in means)}}};",
        f"const float FEATURE_STDS[8]  = {{{', '.join(f'{v:.6f}f' for v in stds)}}};",
        "",
        "// 特征顺序: " + ", ".join(feature_names),
        "#endif",
    ]
    header_path = os.path.join(output_dir, "scaler_params.h")
    with open(header_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"ESP32 归一化头文件已导出至: {header_path}")


def create_sliding_windows(df, window_size=100, step_size=50, fall_ratio_threshold=0.5):
    """
    滑动窗口切割。
    窗口内跌倒帧比例超过 fall_ratio_threshold 才标为跌倒，
    避免"过渡窗口"噪声标签。
    """
    print(f"开始滑动窗口切割... (窗口大小: {window_size}, 步长: {step_size}, 跌倒判定阈值: {fall_ratio_threshold*100:.0f}%)")
    feature_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'acc_mag', 'gyro_mag']
    sensor_data = df[feature_cols].values
    label_data = df['numeric_label'].values

    features, labels = [], []
    for i in range(0, len(df) - window_size + 1, step_size):
        window_features = sensor_data[i: i + window_size]
        fall_ratio = np.mean(label_data[i: i + window_size])
        window_label = 1 if fall_ratio >= fall_ratio_threshold else 0
        features.append(window_features)
        labels.append(window_label)

    X = np.array(features)
    y = np.array(labels)
    fall_windows = y.sum()
    print(f"切割完毕！共 {len(y)} 个窗口 | 跌倒窗口: {fall_windows} ({fall_windows/len(y)*100:.1f}%) | 正常窗口: {len(y)-fall_windows}")
    return X, y


# ==========================================
# 2. 模型构建模块 (专为 ESP32 设计的轻量级 1D-CNN)
# ==========================================
def build_tinyml_model(input_shape):
    """
    轻量级 1D-CNN，加入 BatchNormalization 加速收敛、稳定训练。
    参数量约 5K，量化后 TFLite 体积 < 30KB，适合 ESP32。
    """
    model = Sequential([
        Input(shape=input_shape),

        Conv1D(filters=16, kernel_size=5, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
        GlobalAveragePooling1D(),

        Dense(32, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    return model


# ==========================================
# 3. 模型转换与导出 (TensorFlow Lite)
# ==========================================
def convert_to_tflite(model, save_path):
    """将 Keras 模型转换为 TFLite 格式（动态范围量化）"""
    print("\n正在将模型转换为 TensorFlow Lite 格式...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(save_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite 模型已保存至: {save_path} (大小: {len(tflite_model) / 1024:.2f} KB)")


# ==========================================
# 主执行流程
# ==========================================
if __name__ == '__main__':
    os.makedirs('../models', exist_ok=True)
    raw_data_dir = '../data/raw/'
    scaler_path = '../models/scaler.pkl'

    try:
        # 1. 加载数据并归一化
        df = load_and_merge_csvs(raw_data_dir)
        df = normalize_data(df, scaler_path, fit=True)

        # 2. 滑动窗口切割 (采样率 50Hz，窗口 2 秒，50% 重叠)
        X, y = create_sliding_windows(df, window_size=100, step_size=50, fall_ratio_threshold=0.5)

        # 3. 划分训练集 (80%) 和测试集 (20%)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"训练集: {len(X_train)} 样本，测试集: {len(X_test)} 样本")

        # 4. 计算类别权重，解决跌倒样本稀少问题
        class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        print(f"类别权重: 正常={class_weights[0]:.2f}, 跌倒={class_weights[1]:.2f}")

        # 5. 构建模型
        input_shape = (X_train.shape[1], X_train.shape[2])  # (100, 8)
        model = build_tinyml_model(input_shape)
        model.summary()

        # 6. 训练
        print("\n开始训练模型...")
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=8,
                                             restore_best_weights=True, mode='max'),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                 patience=3, min_lr=1e-5),
        ]

        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            class_weight=class_weight_dict,
            callbacks=callbacks
        )

        # 7. 评估：重点看 Recall（跌倒漏报率），误报比漏报代价低
        print("\n模型在测试集上的表现:")
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()

        print("\n混淆矩阵:")
        print(confusion_matrix(y_test, y_pred))
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, target_names=["正常动作", "跌倒"]))

        # 8. 保存模型
        keras_model_path = '../models/lunacane_model.keras'
        tflite_model_path = '../models/lunacane_model.tflite'

        model.save(keras_model_path)
        print(f"\nKeras 模型已保存至: {keras_model_path}")

        convert_to_tflite(model, tflite_model_path)
        print("\n训练完成！")

    except Exception as e:
        import traceback
        print(f"\n运行出错: {e}")
        traceback.print_exc()
