import numpy as np 
import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import pickle
import matplotlib
import matplotlib.font_manager as fm
from os import path
import seaborn as sns

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def preprocess_data(file_path):
    # Load data
    X = pickle.load(open(file_path, 'rb'))
    
    # Define column order
    column_order = ['Faculty', 'Gender', 'Year', 'PHYSICAL', 'RELATIONSHIP', 'ACADEMIC', 'ENVIRONMENTAL', 'PROBLEM-SOLVING', 'SEEKING-SOCIAL-SUPPORT', 'AVOIDANCE']
    
    # Reindex columns
    X = X.reindex(columns=column_order)
    
    # Calculate means and standard deviations
    PHYSICAL_MEAN = X['PHYSICAL'].mean()
    PHYSICAL_STD = X['PHYSICAL'].std()

    RELATIONSHIP_MEAN = X['RELATIONSHIP'].mean()
    RELATIONSHIP_STD = X['RELATIONSHIP'].std()

    ACADEMIC_MEAN = X['ACADEMIC'].mean()
    ACADEMIC_STD = X['ACADEMIC'].std()

    ENVIRONMENTAL_MEAN = X['ENVIRONMENTAL'].mean()
    ENVIRONMENTAL_STD = X['ENVIRONMENTAL'].std()

    PROBLEM_SOLVING_MEAN = X['PROBLEM-SOLVING'].mean()
    PPROBLEM_SOLVING_STD = X['PROBLEM-SOLVING'].std()

    SEEKING_SOCIAL_SUPPOR_MEAN = X['SEEKING-SOCIAL-SUPPORT'].mean()
    SEEKING_SOCIAL_SUPPOR_STD = X['SEEKING-SOCIAL-SUPPORT'].std()

    AVOIDANCE_MEAN = X['AVOIDANCE'].mean()
    AVOIDANCE_STD = X['AVOIDANCE'].std()
    
    # Normalize columns
    columns_to_normalize = ['PHYSICAL', 'RELATIONSHIP', 'ACADEMIC', 'ENVIRONMENTAL', 'PROBLEM-SOLVING', 'SEEKING-SOCIAL-SUPPORT', 'AVOIDANCE']
    X[columns_to_normalize] = X[columns_to_normalize].apply(zscore)
    
    # Get dummy variables
    X = pd.get_dummies(X, columns=['Faculty', 'Gender', 'Year'])
    
    return X, PHYSICAL_MEAN, PHYSICAL_STD, RELATIONSHIP_MEAN, RELATIONSHIP_STD, ACADEMIC_MEAN, ACADEMIC_STD, ENVIRONMENTAL_MEAN, ENVIRONMENTAL_STD, PROBLEM_SOLVING_MEAN, PPROBLEM_SOLVING_STD, SEEKING_SOCIAL_SUPPOR_MEAN, SEEKING_SOCIAL_SUPPOR_STD, AVOIDANCE_MEAN, AVOIDANCE_STD










def preparation_columns_and_scores(df):
    # Summarize scores
    P1 = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    P2 = ['Q6', 'Q7', 'Q8', 'Q9', 'Q10']
    P3 = ['Q11', 'Q12', 'Q13', 'Q14', 'Q15']
    P4 = ['Q16', 'Q17', 'Q18', 'Q19', 'Q20']
    P5 = ['Q21', 'Q22', 'Q23', 'Q24', 'Q25']
    P6 = ['Q26', 'Q27', 'Q28', 'Q29', 'Q30']
    P7 = ['Q31', 'Q32', 'Q33', 'Q34', 'Q35']

    df['PHYSICAL'] = df[P1].sum(axis=1) 
    df['RELATIONSHIP'] = df[P2].sum(axis=1) 
    df['ACADEMIC'] = df[P3].sum(axis=1) 
    df['ENVIRONMENTAL'] = df[P4].sum(axis=1) 
    df['PROBLEM-SOLVING'] = df[P5].sum(axis=1) 
    df['SEEKING-SOCIAL-SUPPORT'] = df[P6].sum(axis=1)
    df['AVOIDANCE'] = df[P7].sum(axis=1)
    
    # Drop unnecessary columns
    columns_to_drop = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11', 'Q12', 'Q13', 'Q14', 'Q15', 'Q16', 'Q17', 'Q18', 'Q19', 'Q20', 'Q21', 'Q22', 'Q23', 'Q24', 'Q25', 'Q26', 'Q27', 'Q28', 'Q29', 'Q30', 'Q31', 'Q32', 'Q33', 'Q34', 'Q35']
    df = df.drop(columns=columns_to_drop)
    
    # Normalize scores
    df['PROBLEM-SOLVING'] = (df['PROBLEM-SOLVING'] / 15) * 20
    df['SEEKING-SOCIAL-SUPPORT'] = (df['SEEKING-SOCIAL-SUPPORT'] / 15) * 20
    df['AVOIDANCE'] = (df['AVOIDANCE'] / 15) * 20

    return df













def normalization_from_old_data(df, file_path):
#สร้าง function Data Normalization โดยอิงจากคอลัมน์ X
    X, PHYSICAL_MEAN, PHYSICAL_STD, RELATIONSHIP_MEAN, RELATIONSHIP_STD, ACADEMIC_MEAN, ACADEMIC_STD, ENVIRONMENTAL_MEAN, ENVIRONMENTAL_STD, PROBLEM_SOLVING_MEAN, PPROBLEM_SOLVING_STD, SEEKING_SOCIAL_SUPPOR_MEAN, SEEKING_SOCIAL_SUPPOR_STD, AVOIDANCE_MEAN, AVOIDANCE_STD = preprocess_data(file_path)
    df = preparation_columns_and_scores(df)
    df_ready = pd.get_dummies(df, columns=['Faculty', 'Gender', 'Year'])
    df_ready['PHYSICAL'] = (df_ready['PHYSICAL'] - PHYSICAL_MEAN) / PHYSICAL_STD
    df_ready['RELATIONSHIP'] = (df_ready['RELATIONSHIP'] - RELATIONSHIP_MEAN) / RELATIONSHIP_STD
    df_ready['ACADEMIC'] = (df_ready['ACADEMIC'] - ACADEMIC_MEAN) / ACADEMIC_STD
    df_ready['ENVIRONMENTAL'] = (df_ready['ENVIRONMENTAL'] - ENVIRONMENTAL_MEAN) / ENVIRONMENTAL_STD
    df_ready['PROBLEM-SOLVING'] = (df_ready['PROBLEM-SOLVING'] - PROBLEM_SOLVING_MEAN) / PPROBLEM_SOLVING_STD
    df_ready['SEEKING-SOCIAL-SUPPORT'] = (df_ready['SEEKING-SOCIAL-SUPPORT'] - SEEKING_SOCIAL_SUPPOR_MEAN) / SEEKING_SOCIAL_SUPPOR_STD
    df_ready['AVOIDANCE'] = (df_ready['AVOIDANCE'] - AVOIDANCE_MEAN) / AVOIDANCE_STD
    df_ready = df_ready.reindex(columns=X.columns)
    df_ready = df_ready.fillna("0")
    return df ,df_ready, X















def predict_pca_newdata(df, file_path):
    df ,df_ready, X = normalization_from_old_data(df, file_path)

    # สร้าง function ทำงาน Clustering ข้อมูลเก่าใหม่
    pca = PCA(n_components=3)
    x_pca = pca.fit_transform(X)

    kmeans_pca = KMeans(n_clusters=3, init='k-means++', random_state=200)
    kmeans_pca.fit(x_pca)

    # สร้าง function predict ข้อมูลใหม่เพื่อให้ตกลง Cluster
    df_cluster = pca.transform(df_ready)
    df_cluster = kmeans_pca.predict(df_cluster)
    df['Cluster'] = df_cluster
    df['Cluster'] = df['Cluster'].map({0: 'The Life Balancers', 1: 'The Chill Crew', 2: 'The Always Fighters'})
    return df













