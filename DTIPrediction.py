import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score
)
from sklearn.utils import resample
import joblib  # 用于保存和加载模型
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# 评估模型性能的函数
def evaluate_model(model, X, y):
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    metrics = {
        'Accuracy': accuracy_score(y, preds),
        'Precision': precision_score(y, preds),
        'Recall': recall_score(y, preds),
        'F1 Score': f1_score(y, preds),
        'AUC': roc_auc_score(y, probs),
        'AUPRC': average_precision_score(y, probs)
    }

    return metrics

# 绘制学习曲线
def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='f1'
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()

if __name__ == '__main__':
    # 读取原始数据，确保包含 'ID' 列，和特征列、标签列
    train_df = pd.read_csv('Fusion_features_with_labels_RGCN.csv')

    feature_columns = [str(i) for i in range(64)]  # 特征列名称，字符串类型
    id_column = 'ID'  # ID列名称
    label_column = 'Interaction'  # 标签列名称

    # 查看是否包含ID列
    if id_column not in train_df.columns:
        raise ValueError(f"Input CSV文件中缺少'{id_column}'列，请确认！")

    # 数据平衡处理（上采样或下采样）
    positive_samples = train_df[train_df[label_column] == 1]
    negative_samples = train_df[train_df[label_column] == 0]

    if len(positive_samples) < len(negative_samples):
        positive_samples_balanced = resample(positive_samples, replace=True,
                                             n_samples=len(negative_samples), random_state=42)
        balanced_df = pd.concat([positive_samples_balanced, negative_samples], ignore_index=True)
    else:
        negative_samples_balanced = resample(negative_samples, replace=False,
                                             n_samples=len(positive_samples), random_state=42)
        balanced_df = pd.concat([positive_samples, negative_samples_balanced], ignore_index=True)

    print(f"平衡后数据集大小: {len(balanced_df)}, 正样本数量: {sum(balanced_df[label_column]==1)}, 负样本数量: {sum(balanced_df[label_column]==0)}")

    # 划分训练集和测试集，带ID和标签，分层抽样
    train_set, test_set = train_test_split(
        balanced_df,
        test_size=0.2,
        random_state=42,
        stratify=balanced_df[label_column]
    )

    # 提取训练特征和标签
    X_train = train_set[feature_columns].values.astype(float)
    y_train = train_set[label_column].values.astype(int)

    # 提取测试特征和标签
    X_test = test_set[feature_columns].values.astype(float)
    y_test = test_set[label_column].values.astype(int)

    # 定义XGBoost超参数搜索空间
    param_grid = {
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8],
        'gamma': [0, 0.1],
        'reg_lambda': [1, 10],  # L2正则化
        'reg_alpha': [0, 0.1]   # L1正则化
    }

    # 分层K折交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 初始化XGBClassifier
    xgb_model = XGBClassifier(random_state=42, tree_method='hist', eval_metric='logloss')

    # 超参数随机搜索
    random_search = RandomizedSearchCV(
        xgb_model,
        param_grid,
        n_iter=25,
        scoring='f1',
        cv=cv,
        n_jobs=-1,
        verbose=2,
        random_state=42
    )

    print("开始超参数搜索...")
    random_search.fit(X_train, y_train)
    print(f"最佳参数: {random_search.best_params_}")

    # 使用最佳参数训练最终模型
    best_model = random_search.best_estimator_

    # 保存训练好的模型
    joblib.dump(best_model, 'best_xgboost_model.pkl')
    print("模型已保存到 'best_xgboost_model.pkl'.")

    # 评估训练集表现
    train_metrics = evaluate_model(best_model, X_train, y_train)
    print("\n训练集指标:")
    for k, v in train_metrics.items():
        print(f"{k}: {v:.4f}")

    # 评估测试集表现
    test_metrics = evaluate_model(best_model, X_test, y_test)
    print("\n测试集指标:")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    # 生成训练集的预测结果文件
    train_probs = best_model.predict_proba(X_train)[:, 1]
    train_preds = best_model.predict(X_train)
    train_results = pd.DataFrame({
        'Pair_ID': train_set[id_column].values,
        'Interaction_Score': train_probs,
        'Predicted_Label': train_preds,
        'True_Label': y_train
    })
    train_results_sorted = train_results.sort_values(by='Interaction_Score', ascending=False)
    train_results_sorted.to_csv('RGCN_train_predictions.csv', index=False)
    print("\n训练集预测结果已保存到 'RGCN_train_predictions.csv'.")

    # 生成测试集的预测结果文件
    test_probs = best_model.predict_proba(X_test)[:, 1]
    test_preds = best_model.predict(X_test)
    test_results = pd.DataFrame({
        'Pair_ID': test_set[id_column].values,
        'Interaction_Score': test_probs,
        'Predicted_Label': test_preds,
        'True_Label': y_test
    })
    test_results_sorted = test_results.sort_values(by='Interaction_Score', ascending=False)
    test_results_sorted.to_csv('RGCN_test_predictions.csv', index=False)
    print("测试集预测结果已保存到 'RGCN_test_predictions.csv'.")

