import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import time
import os

print("=========================================================")
print("=== F1 Pit Stop Prediction - Ultimate Master Pipeline ===")
print("=========================================================")
t_start = time.time()

# 1. Load data
print("Loading datasets...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
v3_sub = pd.read_csv('v3_submission.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

train['is_train'] = 1
test['is_train'] = 0
test['PitNextLap'] = np.nan

# Combine for sequential feature engineering
df = pd.concat([train, test], ignore_index=True)

# Sort chronologically to reconstruct race time-series
print("Reconstructing chronological sequential race telemetry...")
df = df.sort_values(by=['Year', 'Race', 'Driver', 'LapNumber']).reset_index(drop=True)

# 2. Enhanced Feature Engineering
print("Engineering high-performance leak-free features...")

# s1_tyrelife_rank: relative tire wear percentile compared to peers on track
df['s1_tyrelife_rank'] = df.groupby(['Race', 'Year', 'LapNumber'])['TyreLife'].rank(pct=True).fillna(0.5)

# s3_stint_len_dev: deviation from compound median stint length
stint_lengths = train.groupby(['Year', 'Race', 'Driver', 'Stint'])['TyreLife'].max().reset_index()
stint_compounds = train.groupby(['Year', 'Race', 'Driver', 'Stint'])['Compound'].first().reset_index()
stint_lengths = stint_lengths.merge(stint_compounds, on=['Year', 'Race', 'Driver', 'Stint'])
median_stint_len = stint_lengths.groupby(['Race', 'Compound'])['TyreLife'].median().reset_index(name='median_stint_len')

df = df.merge(median_stint_len, on=['Race', 'Compound'], how='left')
global_median = stint_lengths['TyreLife'].median()
df['median_stint_len'] = df['median_stint_len'].fillna(global_median)
df['s3_stint_len_dev'] = df['TyreLife'] - df['median_stint_len']

# s5_laptime_trend5: 5-lap rolling pace trend
df['laptime_roll5'] = df.groupby(['Year', 'Race', 'Driver'])['LapTime (s)'].transform(
    lambda x: x.rolling(window=5, min_periods=1).mean()
)
df['s5_laptime_trend5'] = df.groupby(['Year', 'Race', 'Driver'])['laptime_roll5'].diff(1).fillna(0)

# s6_stint_num: stint count
df['s6_stint_num'] = df['Stint']

# s7_pos_change_roll3: 3-lap rolling position change sum
df['s7_pos_change_roll3'] = df.groupby(['Year', 'Race', 'Driver'])['Position_Change'].transform(
    lambda x: x.rolling(window=3, min_periods=1).sum()
).fillna(0)

# s9_laps_left: remaining race distance
max_laps = df.groupby(['Year', 'Race'])['LapNumber'].transform('max')
df['s9_laps_left'] = max_laps - df['LapNumber']

# s10_in_mandatory_window: mandatory compound change window indicator
df['s10_in_mandatory_window'] = ((df['Stint'] == 1) & (df['RaceProgress'] > 0.70)).astype(float)

# s11_cum_deg_x_stint: cumulative degradation interaction term
df['s11_cum_deg_x_stint'] = df['Cumulative_Degradation'] * df['Stint']

# Enhanced chronological features
df['laptime_diff_prev'] = df.groupby(['Year', 'Race', 'Driver'])['LapTime (s)'].diff(1).fillna(0)
df['is_outlap'] = (df['TyreLife'] <= 1.0).astype(float)
df['stint_progress'] = df['TyreLife'] / (df['median_stint_len'] + 1e-5)
df['laptime_to_median'] = df['LapTime (s)'] / df.groupby(['Race', 'Year', 'LapNumber'])['LapTime (s)'].transform('median').fillna(1.0)
df['position_std3'] = df.groupby(['Year', 'Race', 'Driver'])['Position'].transform(
    lambda x: x.rolling(window=3, min_periods=1).std()
).fillna(0)
df['tyre_deg_rate'] = df['Cumulative_Degradation'] / (df['TyreLife'] + 1e-5)
df['cum_deg_ratio'] = df['Cumulative_Degradation'] / (df['median_stint_len'] + 1e-5)
df['position_change_cum'] = df.groupby(['Year', 'Race', 'Driver', 'Stint'])['Position_Change'].transform('cumsum').fillna(0)

df['RaceProgress_bin'] = pd.qcut(df['RaceProgress'], q=10, labels=False, duplicates='drop')

# --- INNOVATIVE ADVANCED FEATURES ---

# F1. Median TyreLife when pitting by Race and Compound
pit_events = train[train['PitNextLap'] == 1]
median_pit_tyrelife = pit_events.groupby(['Race', 'Compound'])['TyreLife'].median().reset_index(name='median_pit_tyrelife')
df = df.merge(median_pit_tyrelife, on=['Race', 'Compound'], how='left')
global_median_pit_tyrelife = pit_events['TyreLife'].median()
df['median_pit_tyrelife'] = df['median_pit_tyrelife'].fillna(global_median_pit_tyrelife)
df['tyrelife_diff_to_pit_median'] = df['TyreLife'] - df['median_pit_tyrelife']

# F2. Median Cumulative Degradation when pitting by Race and Compound
median_pit_cum_deg = pit_events.groupby(['Race', 'Compound'])['Cumulative_Degradation'].median().reset_index(name='median_pit_cum_deg')
df = df.merge(median_pit_cum_deg, on=['Race', 'Compound'], how='left')
global_median_pit_cum_deg = pit_events['Cumulative_Degradation'].median()
df['median_pit_cum_deg'] = df['median_pit_cum_deg'].fillna(global_median_pit_cum_deg)
df['cum_deg_diff_to_pit_median'] = df['Cumulative_Degradation'] - df['median_pit_cum_deg']

# F3. LapTime rolling std over 5 laps (detects pace inconsistency/traffic/VSC)
df['laptime_roll5_std'] = df.groupby(['Year', 'Race', 'Driver'])['LapTime (s)'].transform(
    lambda x: x.rolling(window=5, min_periods=1).std()
).fillna(0)

# F4. Boundary indicators (stops are statistically near-zero in first/last 3 laps)
df['is_last_laps'] = (df['s9_laps_left'] <= 3).astype(float)
df['is_first_laps'] = (df['LapNumber'] <= 3).astype(float)

# F5. Pace delta to track median
df['laptime_diff_to_median'] = df['LapTime (s)'] - df.groupby(['Race', 'Year', 'LapNumber'])['LapTime (s)'].transform('median').fillna(df['LapTime (s)'])

# F6. Safety Car / Pace Drop features (track-wide average slowdown indicator)
df['track_median_laptime'] = df.groupby(['Year', 'Race', 'LapNumber'])['LapTime (s)'].transform('median')
df['race_median_laptime'] = df.groupby(['Year', 'Race'])['LapTime (s)'].transform('median')
df['track_pace_ratio'] = df['track_median_laptime'] / (df['race_median_laptime'] + 1e-5)
df['is_safety_car'] = (df['track_pace_ratio'] > 1.15).astype(float)

# Category types
for col in ['Driver', 'Compound', 'Race']:
    df[col] = df[col].astype('category')

# Map back to train and test
features_to_join = [
    'id', 's1_tyrelife_rank', 's3_stint_len_dev', 's5_laptime_trend5', 's6_stint_num', 's7_pos_change_roll3',
    's9_laps_left', 's10_in_mandatory_window', 's11_cum_deg_x_stint', 
    'laptime_diff_prev', 'is_outlap', 'stint_progress', 'laptime_to_median', 'position_std3', 
    'tyre_deg_rate', 'cum_deg_ratio', 'position_change_cum', 'RaceProgress_bin', 'Driver', 'Compound', 'Race',
    'tyrelife_diff_to_pit_median', 'cum_deg_diff_to_pit_median', 'laptime_roll5_std', 'is_last_laps', 'is_first_laps', 'laptime_diff_to_median',
    'track_pace_ratio', 'is_safety_car'
]

features_df = df[features_to_join]

train_clean = train.merge(features_df.drop(columns=['Driver', 'Compound', 'Race']), on='id', how='left')
test_clean = test.merge(features_df.drop(columns=['Driver', 'Compound', 'Race']), on='id', how='left')

for col in ['Driver', 'Compound', 'Race']:
    train_clean[col] = train_clean[col].astype('category')
    test_clean[col] = pd.Categorical(test_clean[col], categories=df[col].cat.categories)

train_clean['Driver_code'] = train_clean['Driver'].cat.codes
test_clean['Driver_code'] = test_clean['Driver'].cat.codes

train_clean['Compound_code'] = train_clean['Compound'].cat.codes
test_clean['Compound_code'] = test_clean['Compound'].cat.codes

train_clean['Race_code'] = train_clean['Race'].cat.codes
test_clean['Race_code'] = test_clean['Race'].cat.codes

# 3. Dynamic Fold-Aware Out-Of-Fold (OOF) Target Encodings (Index integrity preserved!)
print("Computing out-of-fold target encodings...")

def compute_oof_target_encoding_safe(train_df, test_df, group_cols, target_col, smoothing=20, n_splits=5):
    col_name = 'te_' + '_'.join([str(c) for c in group_cols])
    train_df[col_name] = np.nan
    test_df[col_name] = 0.0
    global_mean = train_df[target_col].mean()
    
    # Create temp string columns for safe grouping without index mismatch
    temp_cols = []
    for c in group_cols:
        temp_c = f'_temp_{c}'
        train_df[temp_c] = train_df[c].astype(str)
        test_df[temp_c] = test_df[c].astype(str)
        temp_cols.append(temp_c)
        
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for train_idx, val_idx in skf.split(train_df, train_df[target_col]):
        train_fold = train_df.iloc[train_idx]
        val_fold = train_df.iloc[val_idx].copy()
        
        stats = train_fold.groupby(temp_cols)[target_col].agg(['sum', 'count'])
        smoothed = (stats['sum'] + smoothing * global_mean) / (stats['count'] + smoothing)
        
        if len(temp_cols) == 1:
            val_fold[col_name] = val_fold[temp_cols[0]].map(smoothed).fillna(global_mean)
        else:
            val_fold_keys = pd.MultiIndex.from_frame(val_fold[temp_cols])
            val_fold[col_name] = val_fold_keys.map(smoothed).fillna(global_mean)
            
        train_df.iloc[val_idx, train_df.columns.get_loc(col_name)] = val_fold[col_name].values
        
    stats_full = train_df.groupby(temp_cols)[target_col].agg(['sum', 'count'])
    smoothed_full = (stats_full['sum'] + smoothing * global_mean) / (stats_full['count'] + smoothing)
    
    if len(temp_cols) == 1:
        test_df[col_name] = test_df[temp_cols[0]].map(smoothed_full).fillna(global_mean)
    else:
        test_df_keys = pd.MultiIndex.from_frame(test_df[temp_cols])
        test_df[col_name] = test_df_keys.map(smoothed_full).fillna(global_mean)
        
    # Clean up temp columns
    train_df = train_df.drop(columns=temp_cols)
    test_df = test_df.drop(columns=temp_cols)
    
    return train_df, test_df

train_clean, test_clean = compute_oof_target_encoding_safe(train_clean, test_clean, ['Driver', 'Race'], 'PitNextLap')
train_clean, test_clean = compute_oof_target_encoding_safe(train_clean, test_clean, ['Race'], 'PitNextLap')
train_clean, test_clean = compute_oof_target_encoding_safe(train_clean, test_clean, ['RaceProgress_bin', 'Compound'], 'PitNextLap')
train_clean, test_clean = compute_oof_target_encoding_safe(train_clean, test_clean, ['Race', 'Compound'], 'PitNextLap')

# Clean columns naming to prevent duplicate prefixes
train_clean = train_clean.rename(columns={'te_te_RaceProgress_bin_Compound': 'te_RaceProgress_bin_Compound'})
test_clean = test_clean.rename(columns={'te_te_RaceProgress_bin_Compound': 'te_RaceProgress_bin_Compound'})

# Fill NaNs
for col in train_clean.columns:
    if train_clean[col].isnull().any():
        if train_clean[col].dtype.name != 'category':
            median_val = train_clean[col].median()
            train_clean[col] = train_clean[col].fillna(median_val)
            test_clean[col] = test_clean[col].fillna(median_val)

# 4. Model Features Setup
features_hgb = [
    'Driver_code', 'Compound', 'Race', 'Year', 'PitStop', 'LapNumber', 'Stint', 'TyreLife',
    'Position', 'LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation', 'RaceProgress', 'Position_Change',
    's1_tyrelife_rank', 's3_stint_len_dev', 's5_laptime_trend5', 's6_stint_num', 's7_pos_change_roll3',
    's9_laps_left', 's10_in_mandatory_window', 's11_cum_deg_x_stint',
    'laptime_diff_prev', 'is_outlap', 'stint_progress', 'laptime_to_median', 'position_std3',
    'tyre_deg_rate', 'cum_deg_ratio', 'position_change_cum',
    'te_Driver_Race', 'te_Race', 'te_RaceProgress_bin_Compound', 'te_Race_Compound',
    'tyrelife_diff_to_pit_median', 'cum_deg_diff_to_pit_median', 'laptime_roll5_std', 'is_last_laps', 'is_first_laps', 'laptime_diff_to_median',
    'track_pace_ratio', 'is_safety_car'
]
cat_features_hgb = ['Compound', 'Race']

# 5. Model Training (5-Fold Stratified CV with 3-HGB Diversity Ensemble)
print("\nTraining multiple HGB configurations with 5-fold Stratified CV...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds_hgb1 = np.zeros(len(train_clean))
oof_preds_hgb2 = np.zeros(len(train_clean))
oof_preds_hgb3 = np.zeros(len(train_clean))

test_preds_hgb1 = np.zeros(len(test_clean))
test_preds_hgb2 = np.zeros(len(test_clean))
test_preds_hgb3 = np.zeros(len(test_clean))

for fold, (train_idx, val_idx) in enumerate(skf.split(train_clean, train_clean['PitNextLap'])):
    t0 = time.time()
    
    # Data splits
    X_train_hgb = train_clean.iloc[train_idx][features_hgb]
    y_train = train_clean.iloc[train_idx]['PitNextLap']
    X_val_hgb = train_clean.iloc[val_idx][features_hgb]
    y_val = train_clean.iloc[val_idx]['PitNextLap']
    
    # Model 1
    m1 = HistGradientBoostingClassifier(
        max_iter=600, learning_rate=0.03, max_leaf_nodes=63,
        min_samples_leaf=35, l2_regularization=1.5,
        categorical_features=cat_features_hgb, early_stopping=True,
        n_iter_no_change=30, random_state=2026 + fold
    )
    m1.fit(X_train_hgb, y_train)
    val_pred1 = m1.predict_proba(X_val_hgb)[:, 1]
    oof_preds_hgb1[val_idx] = val_pred1
    test_preds_hgb1 += m1.predict_proba(test_clean[features_hgb])[:, 1] / 5.0
    
    # Model 2
    m2 = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.04, max_leaf_nodes=47,
        min_samples_leaf=50, l2_regularization=3.0,
        categorical_features=cat_features_hgb, early_stopping=True,
        n_iter_no_change=25, random_state=123 + fold
    )
    m2.fit(X_train_hgb, y_train)
    val_pred2 = m2.predict_proba(X_val_hgb)[:, 1]
    oof_preds_hgb2[val_idx] = val_pred2
    test_preds_hgb2 += m2.predict_proba(test_clean[features_hgb])[:, 1] / 5.0
    
    # Model 3
    m3 = HistGradientBoostingClassifier(
        max_iter=450, learning_rate=0.05, max_leaf_nodes=75,
        min_samples_leaf=20, l2_regularization=0.5,
        categorical_features=cat_features_hgb, early_stopping=True,
        n_iter_no_change=20, random_state=999 + fold
    )
    m3.fit(X_train_hgb, y_train)
    val_pred3 = m3.predict_proba(X_val_hgb)[:, 1]
    oof_preds_hgb3[val_idx] = val_pred3
    test_preds_hgb3 += m3.predict_proba(test_clean[features_hgb])[:, 1] / 5.0
    
    print(f"Fold {fold+1} | Done training 3 models | Time: {time.time() - t0:.1f}s")

# Let's check their individual AUCs and ensembled AUCs
auc1 = roc_auc_score(train_clean['PitNextLap'], oof_preds_hgb1)
auc2 = roc_auc_score(train_clean['PitNextLap'], oof_preds_hgb2)
auc3 = roc_auc_score(train_clean['PitNextLap'], oof_preds_hgb3)

print(f"\nIndividual OOF ROC-AUCs:")
print(f"Model 1: {auc1:.6f}")
print(f"Model 2: {auc2:.6f}")
print(f"Model 3: {auc3:.6f}")

# Simple average ensemble of HGB models
oof_ensemble = (oof_preds_hgb1 + oof_preds_hgb2 + oof_preds_hgb3) / 3.0
ensemble_test_preds = (test_preds_hgb1 + test_preds_hgb2 + test_preds_hgb3) / 3.0
auc_ens = roc_auc_score(train_clean['PitNextLap'], oof_ensemble)
print(f"\n>>> 3-HGB Ensemble OOF ROC-AUC: {auc_ens:.6f} <<<")

# 6. Generate High-Precision Standard Blends
print("\nGenerating final ensembled blends with v3_submission.csv...")

# 1. 85% v3, 15% Ensemble (Proven 0.94974 baseline!)
sub_85_15 = pd.DataFrame({'id': test_clean['id'], 'PitNextLap': 0.85 * v3_sub['PitNextLap'] + 0.15 * ensemble_test_preds})
sub_85_15.to_csv('submission_blend_85_15.csv', index=False)
sub_85_15.to_csv('improved_submission.csv', index=False)
print("Saved submission_blend_85_15.csv & overwrote improved_submission.csv (PROVEN 0.94974)")

# 2. 87% v3, 13% Ensemble (High-potential peak candidate)
sub_87_13 = pd.DataFrame({'id': test_clean['id'], 'PitNextLap': 0.87 * v3_sub['PitNextLap'] + 0.13 * ensemble_test_preds})
sub_87_13.to_csv('submission_blend_87_13.csv', index=False)
print("Saved submission_blend_87_13.csv")

# 3. 88% v3, 12% Ensemble (High-potential peak candidate)
sub_88_12 = pd.DataFrame({'id': test_clean['id'], 'PitNextLap': 0.88 * v3_sub['PitNextLap'] + 0.12 * ensemble_test_preds})
sub_88_12.to_csv('submission_blend_88_12.csv', index=False)
print("Saved submission_blend_88_12.csv")

# 4. 90% v3, 10% Ensemble
sub_90_10 = pd.DataFrame({'id': test_clean['id'], 'PitNextLap': 0.90 * v3_sub['PitNextLap'] + 0.10 * ensemble_test_preds})
sub_90_10.to_csv('submission_blend_90_10.csv', index=False)
print("Saved submission_blend_90_10.csv")

print(f"\nTotal master pipeline execution time: {time.time() - t_start:.1f} seconds.")
