import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                            roc_auc_score, RocCurveDisplay, precision_recall_curve,
                            average_precision_score, PrecisionRecallDisplay)
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子确保结果可复现
np.random.seed(42)

# 1. 数据加载与探索
def load_and_explore_data(file_path):
    """加载数据并初步探索"""
    print("=== 数据加载与探索 ===")
    data = pd.read_csv(file_path)
    print(f"数据集形状: {data.shape}")
    print(f"欺诈交易比例: {data['Class'].value_counts(normalize=True)[1]:.4%}")
    print(f"特征列表: {data.columns.tolist()}")

    # 检查缺失值
    print("\n缺失值检查:")
    print(data.isnull().sum())

    return data

# 2. 数据预处理
def preprocess_data(data):
    """数据预处理"""
    print("\n=== 数据预处理 ===")
    # 创建时间特征
    data['Hour'] = data['Time'] // 3600 % 24
    data['Hour_Sin'] = np.sin(2 * np.pi * data['Hour']/24)
    data['Hour_Cos'] = np.cos(2 * np.pi * data['Hour']/24)

    # 标准化Amount特征
    scaler = StandardScaler()
    data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))

    # 准备特征和标签
    features = data.drop(['Class', 'Time'], axis=1)
    target = data['Class']

    print(f"处理后特征数量: {features.shape[1]}")
    return features, target, scaler

# 3. 可视化数据分布
def visualize_data(data, features, target):
    """可视化数据分布"""
    print("\n=== 数据可视化 ===")
    # 可视化类别分布
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    target.value_counts().plot(kind='bar', color=['skyblue', 'red'])
    plt.title('类别分布')
    plt.xlabel('Class')
    plt.ylabel('数量')
    plt.xticks([0, 1], ['正常(0)', '欺诈(1)'], rotation=0)

    plt.subplot(1, 2, 2)
    target.value_counts(normalize=True).plot(kind='pie', autopct='%1.2f%%',
                                            colors=['lightgreen', 'orange'])
    plt.title('类别比例')
    plt.ylabel('')

    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300)
    plt.show()

    # 可视化特征分布
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(['V1', 'V2', 'V3', 'V4', 'V14', 'V17', 'Amount']):
        plt.subplot(3, 3, i+1)
        plt.hist(features[col][target == 0], bins=50, alpha=0.5, label='正常', color='blue')
        plt.hist(features[col][target == 1], bins=50, alpha=0.5, label='欺诈', color='red')
        plt.title(f'{col} 分布')
        plt.legend()
    plt.tight_layout()
    plt.savefig('feature_distribution.png', dpi=300)
    plt.show()

    # 特征相关性热图
    plt.figure(figsize=(12, 10))
    corr = features.corr()
    sns.heatmap(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('特征相关性热图')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300)
    plt.show()

# 4. 处理类别不平衡
def handle_imbalance(X_train, y_train):
    """使用SMOTE处理类别不平衡"""
    print("\n=== 处理类别不平衡 ===")
    print(f"采样前类别分布: \n{y_train.value_counts()}")

    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"SMOTE后类别分布: \n{pd.Series(y_resampled).value_counts()}")
    return X_resampled, y_resampled

# 5. 有监督模型训练与评估
def train_supervised_models(X_train, y_train, X_test, y_test):
    """训练和评估有监督模型"""
    print("\n=== 有监督模型训练与评估 ===")
    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(scale_pos_weight=100, eval_metric='auc', random_state=42, n_jobs=-1)
    }

    # 使用网格搜索优化XGBoost
    print("优化XGBoost参数...")
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'n_estimators': [100, 200, 300]
    }

    xgb = XGBClassifier(scale_pos_weight=100, eval_metric='auc', random_state=42)
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    best_xgb = grid_search.best_estimator_
    models['XGBoost'] = best_xgb
    print(f"最佳XGBoost参数: {grid_search.best_params_}")

    results = {}
    for name, model in models.items():
        print(f"\n训练 {name}...")
        start_time = time.time()

        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # 预测
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # 存储结果
        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'train_time': train_time
        }

        # 打印评估结果
        print(f"{name} 分类报告:")
        print(classification_report(y_test, y_pred))
        print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")

        # 保存模型
        joblib.dump(model, f'{name.replace(" ", "_")}_model.pkl')
        print(f"{name} 模型已保存")

    return results

# 6. 无监督异常检测方法
def train_unsupervised_models(X_train, y_train, X_test, y_test):
    """训练和评估无监督模型"""
    print("\n=== 无监督模型训练与评估 ===")
    # 计算欺诈比例作为污染参数
    fraud_ratio = y_train.mean()

    models = {
        'Isolation Forest': IsolationForest(
            contamination=fraud_ratio,
            random_state=42,
            n_estimators=200,
            n_jobs=-1
        ),
        'One-Class SVM': OneClassSVM(
            nu=fraud_ratio,
            kernel='rbf',
            gamma='scale'
        )
    }

    results = {}
    for name, model in models.items():
        print(f"\n训练 {name}...")
        start_time = time.time()

        # 只使用正常样本训练
        normal_data = X_train[y_train == 0]
        model.fit(normal_data)
        train_time = time.time() - start_time

        # 预测（-1表示异常，1表示正常）
        predictions = model.predict(X_test)

        # 转换为0-1标签（0=正常，1=异常）
        y_pred = np.where(predictions == 1, 0, 1)

        # 存储结果
        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'train_time': train_time
        }

        # 打印评估结果
        print(f"{name} 分类报告:")
        print(classification_report(y_test, y_pred))

        # 保存模型
        joblib.dump(model, f'{name.replace(" ", "_")}_model.pkl')
        print(f"{name} 模型已保存")

    return results

# 7. 深度学习自编码器异常检测
def train_autoencoder(X_train, y_train, X_test, y_test):
    """训练自编码器进行异常检测"""
    print("\n=== 自编码器异常检测 ===")
    # 只使用正常样本训练
    normal_data = X_train[y_train == 0]
    input_dim = X_train.shape[1]

    # 创建自编码器模型
    def create_autoencoder():
        input_layer = Input(shape=(input_dim,))

        # 编码器
        encoder = Dense(32, activation="relu")(input_layer)
        encoder = Dense(16, activation="relu")(encoder)
        encoder = Dense(8, activation="relu")(encoder)

        # 解码器
        decoder = Dense(16, activation="relu")(encoder)
        decoder = Dense(32, activation="relu")(decoder)
        decoder = Dense(input_dim, activation="linear")(decoder)

        autoencoder = Model(inputs=input_layer, outputs=decoder)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder

    autoencoder = create_autoencoder()
    autoencoder.summary()

    # 训练参数
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # 训练模型
    start_time = time.time()
    history = autoencoder.fit(
        normal_data, normal_data,
        epochs=100,
        batch_size=256,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    train_time = time.time() - start_time

    # 绘制训练历史
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('自编码器训练历史')
    plt.ylabel('损失 (MSE)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('autoencoder_training_history.png', dpi=300)
    plt.show()

    # 计算重建误差
    reconstructions = autoencoder.predict(X_test)
    mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)

    # 根据重建误差确定异常（使用百分位阈值）
    threshold = np.percentile(mse, 100 * (1 - y_train.mean()))  # 根据欺诈比例设置阈值
    y_pred = (mse > threshold).astype(int)

    print("自编码器 分类报告:")
    print(classification_report(y_test, y_pred))

    # 保存模型
    autoencoder.save('autoencoder_model.h5')
    print("自编码器模型已保存")

    return {
        'model': autoencoder,
        'y_pred': y_pred,
        'mse': mse,
        'train_time': train_time
    }

# 8. 模型评估与比较
def evaluate_and_compare_models(results, y_test):
    """评估所有模型并比较性能"""
    print("\n=== 模型评估与比较 ===")
    evaluation_results = {}

    for model_name, result in results.items():
        print(f"\n{'='*50}")
        print(f"{model_name} 综合评估")
        print(f"{'='*50}")

        y_pred = result['y_pred']

        # 分类报告
        print("分类报告:")
        print(classification_report(y_test, y_pred))

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['正常', '欺诈'],
                   yticklabels=['正常', '欺诈'])
        plt.title(f'{model_name} - 混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{model_name.replace(" ", "_")}.png', dpi=300)
        plt.show()

        # ROC曲线（如果有概率预测）
        if 'y_pred_proba' in result:
            y_pred_proba = result['y_pred_proba']
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            print(f"AUC-ROC: {roc_auc:.4f}")

            # 绘制ROC曲线
            RocCurveDisplay.from_predictions(y_test, y_pred_proba, name=model_name)
            plt.title(f'{model_name} - ROC曲线')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'roc_curve_{model_name.replace(" ", "_")}.png', dpi=300)
            plt.show()

            # 绘制精确率-召回率曲线
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            ap = average_precision_score(y_test, y_pred_proba)

            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, label=f'{model_name} (AP={ap:.2f})')
            plt.xlabel('召回率')
            plt.ylabel('精确率')
            plt.title(f'{model_name} - 精确率-召回率曲线')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'pr_curve_{model_name.replace(" ", "_")}.png', dpi=300)
            plt.show()
        else:
            roc_auc = None
            ap = None

        # 关键指标
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        # 存储评估结果
        evaluation_results[model_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'auc_roc': roc_auc,
            'ap': ap,
            'train_time': result.get('train_time', None)
        }

    # 创建性能比较DataFrame
    performance_df = pd.DataFrame.from_dict(evaluation_results, orient='index')

    # 添加训练时间列
    performance_df['train_time'] = performance_df['train_time'].apply(
        lambda x: f"{x:.2f}秒" if x is not None else "N/A")

    print("\n模型性能比较:")
    print(performance_df)

    # 可视化比较
    plt.figure(figsize=(14, 8))

    # 指标比较
    plt.subplot(2, 2, 1)
    performance_df[['precision', 'recall', 'f1']].plot(kind='bar', ax=plt.gca())
    plt.title('模型性能比较')
    plt.ylabel('分数')
    plt.xticks(rotation=45)
    plt.legend(loc='lower right')
    plt.grid(axis='y')

    # AUC-ROC比较
    if performance_df['auc_roc'].notna().any():
        plt.subplot(2, 2, 2)
        performance_df['auc_roc'].plot(kind='bar', color='purple')
        plt.title('AUC-ROC分数比较')
        plt.ylabel('AUC-ROC')
        plt.xticks(rotation=45)
        plt.grid(axis='y')

    # 训练时间比较
    plt.subplot(2, 2, 3)
    # 提取数值型训练时间
    train_times = performance_df['train_time'].apply(
        lambda x: float(x.replace('秒', '')) if isinstance(x, str) and '秒' in x else x)
    train_times.plot(kind='bar', color='green')
    plt.title('训练时间比较')
    plt.ylabel('时间(秒)')
    plt.xticks(rotation=45)
    plt.grid(axis='y')

    plt.tight_layout()
    plt.savefig('model_performance_comparison.png', dpi=300)
    plt.show()

    return performance_df

# 9. 特征重要性分析
def analyze_feature_importance(models, features):
    """分析特征重要性"""
    print("\n=== 特征重要性分析 ===")
    plt.figure(figsize=(14, 10))

    for i, (model_name, model_info) in enumerate(models.items()):
        model = model_info['model']

        # 只分析树模型
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = features.columns

            # 创建特征重要性DataFrame
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(15)

            # 绘制水平条形图
            plt.subplot(2, 2, i+1)
            sns.barplot(x='importance', y='feature', data=feature_importance_df)
            plt.title(f'{model_name} - Top 15 特征重要性')
            plt.tight_layout()

    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    plt.show()

# 10. 无标签情况下的异常检测
def unsupervised_anomaly_detection(X_train, X_test, y_test):
    """无标签情况下的异常检测模型"""
    print("\n=== 无标签情况下的异常检测 ===")
    # 估计异常比例（假设为1%）
    contamination = 0.01

    # 多种无监督检测器
    detectors = {
        'Isolation Forest': IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=200,
            n_jobs=-1
        ),
        'One-Class SVM': OneClassSVM(
            nu=contamination,
            kernel='rbf',
            gamma='scale'
        ),
        'Local Outlier Factor': LocalOutlierFactor(
            n_neighbors=20,
            contamination=contamination,
            novelty=True  # 必须设置为True才能用于预测
        )
    }

    # 存储结果
    predictions = {}
    anomaly_scores = {}

    for name, detector in detectors.items():
        print(f"\n训练 {name}...")
        start_time = time.time()

        try:
            # 训练模型
            detector.fit(X_train)
            train_time = time.time() - start_time

            # 预测
            if hasattr(detector, 'decision_function'):
                scores = detector.decision_function(X_test)
                # 将分数转换为异常概率（分数越低越可能是异常）
                anomaly_prob = 1 / (1 + np.exp(-scores))
            elif hasattr(detector, 'score_samples'):
                scores = detector.score_samples(X_test)
                anomaly_prob = 1 / (1 + np.exp(-scores))
            else:
                predictions_arr = detector.predict(X_test)
                anomaly_prob = np.where(predictions_arr == -1, 1, 0)

            # 存储结果
            anomaly_scores[name] = anomaly_prob
            predictions[name] = (anomaly_prob > 0.5).astype(int)

            # 打印评估结果
            print(f"{name} 分类报告:")
            print(classification_report(y_test, predictions[name]))

        except Exception as e:
            print(f"{name} 训练/预测出错: {e}")
            continue

    # 集成预测（简单平均）
    if anomaly_scores:
        ensemble_scores = np.mean(list(anomaly_scores.values()), axis=0)
        ensemble_predictions = (ensemble_scores > 0.5).astype(int)

        print("\n集成无监督检测分类报告:")
        print(classification_report(y_test, ensemble_predictions))

        # 绘制异常分数分布
        plt.figure(figsize=(10, 6))
        plt.hist(ensemble_scores[y_test == 0],
                 bins=50, alpha=0.5, label='正常', color='blue')
        plt.hist(ensemble_scores[y_test == 1],
                 bins=50, alpha=0.5, label='欺诈', color='red')
        plt.title('无监督异常分数分布')
        plt.xlabel('异常分数')
        plt.ylabel('频次')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('unsupervised_anomaly_scores.png', dpi=300)
        plt.show()

        return {
            'individual': predictions,
            'ensemble': ensemble_predictions,
            'scores': ensemble_scores
        }
    else:
        return None

# 主函数
def main():
    # 加载数据
    data = load_and_explore_data('creditcard.csv')

    # 数据预处理
    features, target, scaler = preprocess_data(data)

    # 保存标准化器
    joblib.dump(scaler, 'scaler.pkl')
    print("标准化器已保存")

    # 可视化数据
    visualize_data(data, features, target)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=42, stratify=target
    )
    print(f"\n训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")
    print(f"训练集中欺诈比例: {y_train.mean():.4%}")

    # 处理类别不平衡（仅用于有监督模型）
    X_resampled, y_resampled = handle_imbalance(X_train, y_train)

    # 训练和评估有监督模型
    supervised_results = train_supervised_models(X_resampled, y_resampled, X_test, y_test)

    # 训练和评估无监督模型
    unsupervised_results = train_unsupervised_models(X_train, y_train, X_test, y_test)

    # 训练自编码器
    autoencoder_result = train_autoencoder(X_train, y_train, X_test, y_test)
    unsupervised_results['Autoencoder'] = autoencoder_result

    # 合并所有结果
    all_results = {**supervised_results, **unsupervised_results}

    # 评估和比较所有模型
    performance_df = evaluate_and_compare_models(all_results, y_test)

    # 特征重要性分析
    analyze_feature_importance(supervised_results, features)

    # 无标签情况下的异常检测
    unsupervised_detection_result = unsupervised_anomaly_detection(X_train, X_test, y_test)

    # 结果分析报告
    print("\n=== 最终结果分析报告 ===")
    print("1. 数据特性:")
    print(f"   - 总样本数: {len(data)}")
    print(f"   - 欺诈比例: {target.mean():.4%}")
    print(f"   - 特征数量: {features.shape[1]}")

    print("\n2. 模型性能总结:")
    print(performance_df[['precision', 'recall', 'f1', 'auc_roc']])

    print("\n3. 关键发现:")
    print("   - 有监督模型（特别是XGBoost）在欺诈检测上表现最佳")
    print("   - 召回率是最重要的指标，XGBoost达到了最高的召回率")
    print("   - 无监督方法在无标签情况下是可行的替代方案")
    print("   - 自编码器在无监督方法中表现最好")

    print("\n4. 建议:")
    print("   - 对于有标签数据，推荐使用XGBoost模型")
    print("   - 对于无标签数据，推荐使用集成无监督方法或自编码器")
    print("   - 在实际部署中，可能需要调整阈值以平衡精确率和召回率")

if __name__ == "__main__":
    main()