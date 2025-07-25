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


def display_histoplot(df, column_name, title):
    """
    Display histogram of a column
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column_name (str): Column name
        """
    df_copy = df.copy()
    warnings.filterwarnings("ignore", category=FutureWarning)
    plt.style.use('dark_background')

    # Histogram to see the distribution of age column
    sns.histplot(df_copy[column_name], kde=True, bins=30)
    plt.title(title)
    plt.show()

def display_relative_histoplot(df, x_column_name, hue_column_name, title):
    """
    Display relative histogram of a column that shows column colored by the hue of rel_column
    
    Args:
        df (pd.DataFrame): Input DataFrame
        x_column_name (str): Column name for x-axis
        hue_column_name (str): Column name for hue
        title (str): Title of the plot
    """ 
    df_copy = df.copy()
    warnings.filterwarnings("ignore", category=FutureWarning)
    plt.style.use('dark_background')

    # Histogram to see the distribution of age column
    sns.histplot(data=df_copy, x=x_column_name, hue=hue_column_name, multiple='dodge', palette='deep', edgecolor='white')
    plt.title(title)
    plt.show()

def display_multi_histoplot(df, cols):
    """
    Display multiple histograms of a column
    
    Args:
        df (pd.DataFrame): Input DataFrame
        cols (list[str]): List of column names
    """
    df_copy = df.copy()
    warnings.filterwarnings("ignore", category=FutureWarning)
    plt.style.use('dark_background')

    # Histogram to see the distribution of age column
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(cols, 1):
        plt.subplot(2, 3, i)
        sns.histplot(df_copy[col], bins=20, kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

def display_boxplot(df, x_column_name , y_column_name, title):
    """
    Display boxplot of a column
    
    Args:
        df (pd.DataFrame): Input DataFrame
        x_column_name (str): Column name for x-axis
        y_column_name (str): Column name for y-axis
        """
    df_copy = df.copy()
    warnings.filterwarnings("ignore", category=FutureWarning)
    plt.style.use('dark_background')

    # Boxplot to see the distribution of age column
    sns.boxplot(data=df_copy, x=x_column_name, y=y_column_name)
    plt.title(title)
    plt.show()



def display_multi_boxplot(df, cols, target_col='num'):
    """
    Display multiple boxplots of numerical columns against target
    
    Args:
        df (pd.DataFrame): Input DataFrame
        cols (list[str]): List of numerical column names
        target_col (str): Target column for comparison
    """
    df_copy = df.copy()
    warnings.filterwarnings("ignore", category=FutureWarning)
    plt.style.use('dark_background')

    # Create subplots
    n_cols = len(cols)
    n_rows = (n_cols + 2) // 3  # 3 columns per row
    
    plt.figure(figsize=(15, 5 * n_rows))
    
    for i, col in enumerate(cols, 1):
        plt.subplot(n_rows, 3, i)
        
        # For numerical columns, create boxplot with target as hue
        if target_col in df_copy.columns:
            sns.boxplot(data=df_copy, x=target_col, y=col)
            plt.title(f'{col} by {target_col}')
        else:
            # If no target, just show distribution
            sns.boxplot(data=df_copy, y=col)
            plt.title(f'Distribution of {col}')
            
        plt.xlabel(target_col if target_col in df_copy.columns else '')
        plt.ylabel(col)
    
    plt.tight_layout()
    plt.show()
    
def display_categorical_plots(df, cat_cols, target_col='num'):
    """
    Display count plots for categorical columns
    
    Args:
        df (pd.DataFrame): Input DataFrame
        cat_cols (list[str]): List of categorical column names
        target_col (str): Target column for comparison
    """
    df_copy = df.copy()
    warnings.filterwarnings("ignore", category=FutureWarning)
    plt.style.use('dark_background')

    n_cols = len(cat_cols)
    n_rows = (n_cols + 2) // 3  # 3 columns per row
    
    plt.figure(figsize=(15, 5 * n_rows))
    
    for i, col in enumerate(cat_cols, 1):
        plt.subplot(n_rows, 3, i)
        
        # Count plot with target as hue
        if target_col in df_copy.columns:
            sns.countplot(data=df_copy, x=col, hue=target_col)
            plt.title(f'{col} by {target_col}')
        else:
            sns.countplot(data=df_copy, x=col)
            plt.title(f'Count of {col}')
            
        plt.xticks(rotation=45)
        plt.xlabel(col)
        plt.ylabel('Count')
    
    plt.tight_layout()
    plt.show()
def display_scatterplot(df, x_column_name, y_column_name, title):
    """
    Display scatterplot of two columns
    
    Args:
        df (pd.DataFrame): Input DataFrame
        x_column_name (str): Column name for x-axis
        y_column_name (str): Column name for y-axis
        title (str): Title of the plot
    """
    df_copy = df.copy()
    warnings.filterwarnings("ignore", category=FutureWarning)
    plt.style.use('dark_background')

    # Scatterplot to see the relationship between age and target column
    sns.scatterplot(data=df_copy, x=x_column_name, y=y_column_name)
    plt.title(title)
    plt.show()