import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

FEATURES_OUTPUT_DIR = "data/processed/extracted_features/"
RESULTS_DIR = "results/"  # Directory to save plots

def read_features(method):
    features = []
    for filename in os.listdir(FEATURES_OUTPUT_DIR):
        if filename.endswith('.csv') and filename.startswith(method):
            fold_number = int(filename.split('_')[2])
            df = pd.read_csv(os.path.join(FEATURES_OUTPUT_DIR, filename))
            df['fold'] = fold_number
            features.append(df)
    return pd.concat(features, ignore_index=True)

def evaluate_model(y_test, y_pred, y_score=None):
    results = {}
    results['Accuracy'] = accuracy_score(y_test, y_pred)
    results['Precision'] = precision_score(y_test, y_pred, average='weighted')
    results['Recall'] = recall_score(y_test, y_pred, average='weighted')
    results['F1 Score'] = f1_score(y_test, y_pred, average='weighted')

    if y_score is not None:
        results['ROC AUC Score'] = roc_auc_score(y_test, y_score)
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        plt.figure()
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.savefig(f"{RESULTS_DIR}/precision_recall_curve.png")
        plt.close()

    return results

def train_evaluate_knn(features, method, labels_column='class', n_neighbors=5):
    unique_folds = features['fold'].unique()
    for fold in unique_folds:
        print(f"Training on Fold {fold}")
        train_data = features[features['fold'] != fold]
        test_data = features[features['fold'] == fold]

        
        print(test_data, train_data)
        if labels_column not in train_data.columns:
            print(f"Column '{labels_column}' not found in train_data")
            continue

        # X_train = train_data.drop([labels_column, 'fold'], axis=1)
        # y_train = train_data[labels_column]
        # X_test = test_data.drop([labels_column, 'fold'], axis=1)
        # y_test = test_data[labels_column]

        # knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        # knn.fit(X_train, y_train)
        # y_pred = knn.predict(X_test)
        # y_score = knn.predict_proba(X_test)[:, 1]

        # print(classification_report(y_test, y_pred))
        # cm = confusion_matrix(y_test, y_pred)
        # results = evaluate_model(y_test, y_pred, y_score)
        # for metric, value in results.items():
        #     print(f"{metric}: {value}")

        # cm = confusion_matrix(y_test, y_pred)
        # plt.figure(figsize=(10, 7))
        # sns.heatmap(cm, annot=True, fmt="d")
        # plt.title(f"Confusion Matrix for {method.capitalize()} Method - Fold {fold}")
        # plt.xlabel('Predicted')
        # plt.ylabel('True')
        # plt.savefig(f"{RESULTS_DIR}/confusion_matrix_{method}_fold_{fold}.png")
        # plt.close()

if __name__ == "__main__":
    otsu_features = read_features('otsu')
    train_evaluate_knn(otsu_features, 'otsu')
    adaptive_features = read_features('adaptive')
    train_evaluate_knn(adaptive_features, 'adaptive')
