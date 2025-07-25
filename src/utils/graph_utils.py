import sys
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from src.utils.data_utils import drop_id
import time

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Graphing
def display_correlation(df):
    """
    Display correlation matrix using a heatmap
    
    Args:
        df (pd.DataFrame): Input DataFrame
    """
    # Visualize the correlation matrix using a heatmap
    df_copy = df.copy()
    corr_matrix = df_copy.corr()

    plt.figure(figsize=(14, 9))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()
    