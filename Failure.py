import pandas as pd
import numpy as np

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# import pdb

class Explainer:
    def __init__(self, model, X_test, y_test, shap_values):
        self.version = '0.0'
        self.model = model
        self.X_test = X_test.reset_index(drop=True)
        self.y_test = y_test.reset_index(drop=True)
        self.shap_values = shap_values
        self.features = X_test.columns
        self.y_test_pred = []
        self.false_pred_indices = []
        self.false_positive_indices = []
        self.false_negative_indices = []
        
        self.fp_features = []
        self.fp_mean_feature_values = []
        self.fp_mean_shap_values = []
        self.fp_lengths_of_errors = []
        
        self.fn_features = []
        self.fn_mean_feature_values = []
        self.fn_mean_shap_values = []
        self.fn_lengths_of_errors = []
        
    def __version__(self):
        return self.version
    
    def find_failures(self):
        self.y_test_pred = self.model.predict(self.X_test)
        self.false_pred_indices = self.X_test.index[self.y_test_pred != self.y_test]
        self.false_positive_indices = self.X_test.index[(self.y_test_pred != self.y_test) & (self.y_test_pred == 1)].values
        self.false_negative_indices = self.X_test.index[(self.y_test_pred != self.y_test) & (self.y_test_pred == 0)].values

    def present_false_positive_rows(self):
        print('Number of false positives: %s' % len(self.X_test.loc[self.false_positive_indices]))
        return self.X_test.loc[self.false_positive_indices]
    
    def present_false_negative_rows(self):
        print('Number of false negative: %s' % len(self.X_test.loc[self.false_negative_indices]))
        return self.X_test.loc[self.false_negative_indices]
    
    def present_false_positive_summary(self):
        
        shap_arr = self.shap_values.values[self.false_positive_indices]
        
        positive_shap_arr = shap_arr
        positive_shap_arr[positive_shap_arr < 0] = 0
        
        mean_shap_values = np.array([])
        mean_feature_values = np.array([])
        features = np.array([])
        lengths_of_errors = np.array([])
        
        X_test_filtered = self.X_test.loc[self.false_positive_indices]
        for j, feature in enumerate(self.features):
            feature_shap_values = shap_arr[:, j]
            feature_values = X_test_filtered[feature].values
            
            if len(feature_values[feature_shap_values > 0] > 0):
                mean_shap_values = np.append(mean_shap_values, np.mean(feature_shap_values[feature_shap_values > 0]))
                mean_feature_values = np.append(mean_feature_values, np.mean(feature_values[feature_shap_values > 0]))
                features = np.append(features, feature)
                lengths_of_errors = np.append(lengths_of_errors, len(feature_shap_values[feature_shap_values > 0]))
        
        features = features[np.argsort(-mean_shap_values)]
        mean_feature_values = mean_feature_values[np.argsort(-mean_shap_values)]
        mean_shap_values = mean_shap_values[np.argsort(-mean_shap_values)]
        lengths_of_errors =  lengths_of_errors[np.argsort(-mean_shap_values)]
        
        self.fp_features = features
        self.fp_mean_feature_values = mean_feature_values
        self.fp_mean_shap_values = mean_shap_values
        self.fp_lengths_of_errors = lengths_of_errors
        
        return features, mean_feature_values, mean_shap_values, lengths_of_errors
    

    def present_false_negative_summary(self):
        
        shap_arr = self.shap_values.values[self.false_negative_indices]
        
        negative_shap_arr = shap_arr
        negative_shap_arr[negative_shap_arr > 0] = 0
        
        mean_shap_values = np.array([])
        mean_feature_values = np.array([])
        features = np.array([])
        lengths_of_errors = np.array([])
        
        X_test_filtered = self.X_test.loc[self.false_negative_indices]
        for j, feature in enumerate(self.features):
            feature_shap_values = shap_arr[:, j]
            feature_values = X_test_filtered[feature].values
            
            if len(feature_values[feature_shap_values < 0] > 0):
                mean_shap_values = np.append(mean_shap_values, np.abs(np.mean(feature_shap_values[feature_shap_values < 0])))
                mean_feature_values = np.append(mean_feature_values, np.mean(feature_values[feature_shap_values < 0]))
                features = np.append(features, feature)
                lengths_of_errors = np.append(lengths_of_errors, len(feature_shap_values[feature_shap_values < 0]))
                    
        
        features = features[np.argsort(-mean_shap_values)]
        mean_feature_values = mean_feature_values[np.argsort(-mean_shap_values)]
        mean_shap_values = mean_shap_values[np.argsort(-mean_shap_values)]
        lengths_of_errors =  lengths_of_errors[np.argsort(-mean_shap_values)]
        
        self.fn_features = features
        self.fn_mean_feature_values = mean_feature_values
        self.fn_mean_shap_values = mean_shap_values
        self.fn_lengths_of_errors = lengths_of_errors
        
        return features, mean_feature_values, mean_shap_values, lengths_of_errors
    
    def plot_false_positive_summary(self, top_n=10, features=[]):
        if len(features) == 0:
            y_labels = self.fp_features[:10]
        
        y_labels_post = []

        for i, label in enumerate(y_labels):
            label_string = label + ' (' + str(round(self.fp_mean_feature_values[i], 2)) + ')'
            y_labels_post.append(label_string)

        y_labels = y_labels_post

        sns.set_theme(style="white", context="talk", font_scale=0.7)

        # Set up the matplotlib figure
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

        # Plot for mean SHAP values
        sns.barplot(y=y_labels, x=self.fp_mean_shap_values[:10], palette="rocket", ax=ax1, orient="h")
        ax1.set_xlabel("SHAP values")

        # Plot for error counts
        sns.barplot(y=y_labels, x=self.fp_lengths_of_errors[:10], palette="deep", ax=ax2, orient="h")
        ax2.set_xlabel("Error counts")

        # Finalize the plot
        sns.despine(bottom=True)
        plt.tight_layout(h_pad=2)
        
    def plot_false_negative_summary(self, top_n=10, features=[]):
        if len(features) == 0:
            y_labels = self.fn_features[:10]
        
        y_labels_post = []

        for i, label in enumerate(y_labels):
            label_string = label + ' (' + str(round(self.fn_mean_feature_values[i], 2)) + ')'
            y_labels_post.append(label_string)

        y_labels = y_labels_post

        sns.set_theme(style="white", context="talk", font_scale=0.7)

        # Set up the matplotlib figure
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

        # Plot for mean SHAP values
        sns.barplot(y=y_labels, x=self.fn_mean_shap_values[:10], palette="rocket", ax=ax1, orient="h")
        ax1.set_xlabel("SHAP values")

        # Plot for error counts
        sns.barplot(y=y_labels, x=self.fn_lengths_of_errors[:10], palette="deep", ax=ax2, orient="h")
        ax2.set_xlabel("Error counts")

        # Finalize the plot
        sns.despine(bottom=True)
        plt.tight_layout(h_pad=2)
        
