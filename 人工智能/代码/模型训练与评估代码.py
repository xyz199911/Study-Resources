# -*- coding: utf-8 -*-
"""
基于逻辑回归与朴素贝叶斯的金融风险预测研究
完整可运行代码
"""

# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data(file_path):
    """
    数据加载与预处理函数
    参数: file_path - 数据集文件路径
    返回: 预处理后的特征矩阵X, 标签y, 特征名称列表
    """
    # 读取CSV文件
    data = pd.read_csv(file_path)
    print("数据集形状:", data.shape)
    print("\n数据集前5行:")
    print(data.head())
    
    # 检查缺失值
    print("\n缺失值统计:")
    print(data.isnull().sum())
    
    # 分离特征和标签
    X = data.drop('是否违约', axis=1)
    y = data['是否违约']
    feature_names = X.columns.tolist()
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, feature_names, data

def split_dataset(X, y, test_size=0.3, random_state=42):
    """
    数据集划分函数
    参数: X-特征矩阵, y-标签, test_size-测试集比例, random_state-随机种子
    返回: 训练集和测试集
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\n训练集样本数: {X_train.shape[0]}")
    print(f"测试集样本数: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, y_train):
    """
    训练逻辑回归模型
    """
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    return lr_model

def train_naive_bayes(X_train, y_train):
    """
    训练朴素贝叶斯模型
    """
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    return nb_model

def evaluate_model(model, X_test, y_test, model_name):
    """
    模型评估函数
    计算准确率、精确率、召回率、F1值、AUC等指标
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 计算各项评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"{model_name}模型评估结果")
    print(f"{'='*50}")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1值: {f1:.4f}")
    print(f"AUC值: {auc:.4f}")
    print(f"\n混淆矩阵:")
    print(cm)
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm
    }

def plot_confusion_matrix(cm, model_name, save_path=None):
    """
    绘制混淆矩阵热力图
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['未违约', '违约'], 
                yticklabels=['未违约', '违约'])
    plt.title(f'{model_name} - 混淆矩阵', fontsize=14)
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_curve(y_test, results_dict, save_path=None):
    """
    绘制ROC曲线对比图
    """
    plt.figure(figsize=(10, 8))
    
    for model_name, results in results_dict.items():
        fpr, tpr, _ = roc_curve(y_test, results['y_pred_proba'])
        plt.plot(fpr, tpr, linewidth=2, 
                 label=f'{model_name} (AUC = {results["auc"]:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率 (FPR)', fontsize=12)
    plt.ylabel('真阳性率 (TPR)', fontsize=12)
    plt.title('ROC曲线对比', fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_importance(model, X_test, y_test, feature_names, model_name, save_path=None):
    """
    绘制特征重要性图
    """
    # 使用排列重要性计算特征重要性
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': result.importances_mean
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance_df, palette='viridis')
    plt.title(f'{model_name} - 特征重要性', fontsize=14)
    plt.xlabel('重要性得分', fontsize=12)
    plt.ylabel('特征名称', fontsize=12)
    plt.grid(True, alpha=0.3, axis='x')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return importance_df

def plot_metrics_comparison(results_dict, save_path=None):
    """
    绘制模型各项指标对比图
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    metric_names = ['准确率', '精确率', '召回率', 'F1值', 'AUC']
    
    comparison_data = []
    for model_name, results in results_dict.items():
        for metric, metric_name in zip(metrics, metric_names):
            comparison_data.append({
                '模型': model_name,
                '指标': metric_name,
                '数值': results[metric]
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(metric_names))
    width = 0.35
    
    model_names = list(results_dict.keys())
    for i, model_name in enumerate(model_names):
        values = comparison_df[comparison_df['模型'] == model_name]['数值'].values
        plt.bar(x + i * width, values, width, label=model_name, alpha=0.8)
    
    plt.xlabel('评估指标', fontsize=12)
    plt.ylabel('数值', fontsize=12)
    plt.title('模型性能指标对比', fontsize=14)
    plt.xticks(x + width / 2, metric_names, fontsize=11)
    plt.legend(fontsize=12)
    plt.ylim([0, 1.1])
    plt.grid(True, alpha=0.3, axis='y')
    
    # 在柱子上添加数值标签
    for i, model_name in enumerate(model_names):
        values = comparison_df[comparison_df['模型'] == model_name]['数值'].values
        for j, v in enumerate(values):
            plt.text(j + i * width, v + 0.02, f'{v:.3f}', 
                     ha='center', va='bottom', fontsize=10)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    主函数：执行完整的机器学习流程
    """
    print("="*60)
    print("基于逻辑回归与朴素贝叶斯的金融风险预测研究")
    print("="*60)
    
    # 1. 数据加载与预处理
    print("\n【步骤1】数据加载与预处理")
    X, y, feature_names, raw_data = load_and_preprocess_data('金融风险数据集.csv')
    
    # 2. 数据集划分
    print("\n【步骤2】数据集划分")
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    
    # 3. 模型训练
    print("\n【步骤3】模型训练")
    print("训练逻辑回归模型...")
    lr_model = train_logistic_regression(X_train, y_train)
    print("训练朴素贝叶斯模型...")
    nb_model = train_naive_bayes(X_train, y_train)
    
    # 4. 模型评估
    print("\n【步骤4】模型评估")
    lr_results = evaluate_model(lr_model, X_test, y_test, "逻辑回归")
    nb_results = evaluate_model(nb_model, X_test, y_test, "朴素贝叶斯")
    
    results_dict = {
        '逻辑回归': lr_results,
        '朴素贝叶斯': nb_results
    }
    
    # 5. 结果可视化
    print("\n【步骤5】结果可视化")
    
    # 混淆矩阵
    plot_confusion_matrix(lr_results['confusion_matrix'], '逻辑回归')
    plot_confusion_matrix(nb_results['confusion_matrix'], '朴素贝叶斯')
    
    # ROC曲线对比
    plot_roc_curve(y_test, results_dict)
    
    # 特征重要性
    print("\n逻辑回归特征重要性:")
    lr_importance = plot_feature_importance(lr_model, X_test, y_test, feature_names, '逻辑回归')
    print(lr_importance)
    
    print("\n朴素贝叶斯特征重要性:")
    nb_importance = plot_feature_importance(nb_model, X_test, y_test, feature_names, '朴素贝叶斯')
    print(nb_importance)
    
    # 指标对比图
    plot_metrics_comparison(results_dict)
    
    # 6. 输出最终对比表格
    print("\n" + "="*60)
    print("模型性能综合对比")
    print("="*60)
    comparison_table = pd.DataFrame({
        '评估指标': ['准确率', '精确率', '召回率', 'F1值', 'AUC'],
        '逻辑回归': [
            lr_results['accuracy'],
            lr_results['precision'],
            lr_results['recall'],
            lr_results['f1'],
            lr_results['auc']
        ],
        '朴素贝叶斯': [
            nb_results['accuracy'],
            nb_results['precision'],
            nb_results['recall'],
            nb_results['f1'],
            nb_results['auc']
        ]
    })
    print(comparison_table.round(4))
    
    print("\n" + "="*60)
    print("实验完成！")
    print("="*60)

if __name__ == "__main__":
    main()
