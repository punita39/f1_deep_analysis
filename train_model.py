import pandas as pd
import numpy as np
import pickle
import os
import optuna
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ndcg_score, roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from mapie.classification import MapieClassifier
import lightgbm as lgb
import xgboost as xgb
import warnings

# Constants and Config
MIN_TRAIN_RACES = 50
WALK_FORWARD_START_YEAR = 2023
DEFAULT_ELO = 1500
ELO_K = 32

warnings.filterwarnings("ignore")

# --- Helper: Elo System ---
def calculate_elo(df, decay_rate=0.1):
    elo_ratings = {name: DEFAULT_ELO for name in df['FullName'].unique()}
    last_race_idx = {name: 0 for name in df['FullName'].unique()}
    elo_history = []
    
    races = df.groupby(['Year', 'RoundNumber']).size().index
    global_race_counter = 0
    
    for year, rnd in races:
        global_race_counter += 1
        race_drivers = df[(df['Year'] == year) & (df['RoundNumber'] == rnd)]['FullName'].unique()
        
        # 1. Apply Decay
        for drv in race_drivers:
            races_since = global_race_counter - last_race_idx[drv]
            elo_ratings[drv] = DEFAULT_ELO + (elo_ratings[drv] - DEFAULT_ELO) * np.exp(-decay_rate * (races_since - 1))
        
        # 2. Store pre-race Elo
        current_elos = {drv: elo_ratings[drv] for drv in race_drivers}
        for drv in race_drivers:
            elo_history.append({'FullName': drv, 'Year': year, 'RoundNumber': rnd, 'Elo': current_elos[drv]})
            
        # 3. Update Elo after race (Winner vs Field)
        winner = df[(df['Year'] == year) & (df['RoundNumber'] == rnd) & (df['Won'] == 1)]['FullName'].values
        if len(winner) > 0:
            winner = winner[0]
            for drv in race_drivers:
                if drv == winner: continue
                # Expected score of winner vs this driver
                exp_winner = 1 / (1 + 10 ** ((elo_ratings[drv] - elo_ratings[winner]) / 400))
                # Update (Winner gets +Points, Loser gets -Points)
                elo_ratings[winner] += ELO_K * (1 - exp_winner)
                elo_ratings[drv] += ELO_K * (0 - (1 - exp_winner))
                last_race_idx[drv] = global_race_counter
            last_race_idx[winner] = global_race_counter
            
    return pd.DataFrame(elo_history)

# --- Helper: Circuit Clustering ---
def cluster_circuits(df):
    circuit_stats = df.groupby('Race').agg({
        'TopSpeed': 'mean',
        'QualiDelta': 'mean'
    }).fillna(0)
    
    kmeans = KMeans(n_clusters=4, random_state=42)
    circuit_stats['Cluster'] = kmeans.fit_predict(circuit_stats)
    
    # Manual Mapping of Cluster labels to archetypes (simplified)
    # 0: Street, 1: High-speed, 2: Technical, 3: Mixed
    mapping = {0: 'Street', 1: 'High-speed', 2: 'Technical', 3: 'Mixed'}
    circuit_stats['Archetype'] = circuit_stats['Cluster'].map(mapping)
    return circuit_stats['Archetype'].to_dict()

# --- Main Training Flow ---
def train_pipeline():
    if not os.path.exists('f1_advanced_data.csv'):
        print("Data file not found. Run collect_data.py first.")
        return

    df = pd.read_csv('f1_advanced_data.csv')
    df = df.sort_values(['Year', 'RoundNumber']).reset_index(drop=True)

    # 1. Feature Engineering: Elo
    print("Calculating Elo ratings...")
    elo_df = calculate_elo(df)
    df = df.merge(elo_df, on=['FullName', 'Year', 'RoundNumber'], how='left')

    # 2. Feature Engineering: Target Encoding & Expands
    print("Engineering expanding features...")
    cols_to_expand = ['Won', 'QualiDelta', 'IsMechanicalDNF', 'AvgStintLength']
    for col in cols_to_expand:
        df[f'Exp_{col}'] = df.groupby('FullName')[col].transform(lambda x: x.expanding().mean().shift(1))
    
    # Team Momentum (Rolling 6)
    df['TeamWinRate6'] = df.groupby('TeamName')['Won'].transform(lambda x: x.rolling(6).mean().shift(1))
    
    # 3. Circuit Archetypes
    print("Clustering circuits...")
    archetype_map = cluster_circuits(df)
    df['Archetype'] = df['Race'].map(archetype_map)

    # 4. Field Relative Features
    print("Computing field-relative features...")
    df['RelElo'] = df.groupby(['Year', 'RoundNumber'])['Elo'].transform(lambda x: x / x.mean())
    df['RelQualiDelta'] = df.groupby(['Year', 'RoundNumber'])['QualiDelta'].transform(lambda x: x / x.mean())
    df['RelPointsGap'] = df.groupby(['Year', 'RoundNumber'])['PointsGapToLeader'].transform(lambda x: x / (x.max() + 1))

    # Features for models
    features = [
        'GridPosition', 'RelElo', 'RelQualiDelta', 'RelPointsGap',
        'Exp_Won', 'Exp_QualiDelta', 'Exp_IsMechanicalDNF', 'Exp_AvgStintLength',
        'TeamWinRate6', 'TrackTemp'
    ]
    df[features] = df[features].fillna(0)

    # 5. Mixture-of-Experts Training (Per Archetype)
    all_results = []
    trained_experts = {} # Archetype -> {lgbm, xgb, pl, meta, mapie}
    
    archetypes = ['Street', 'High-speed', 'Technical', 'Mixed']
    
    for arch in archetypes:
        print(f"\n--- Training Expert for {arch} ---")
        arch_df = df[df['Archetype'] == arch].copy()
        
        # Walk-forward splits
        races = arch_df.groupby(['Year', 'RoundNumber']).size().index
        if len(races) < 10: # Not enough data for specialized expert, use generic
            print(f"Skipping {arch} - insufficient data.")
            continue
            
        # For simplicity in this demo, we'll do a large split instead of full walk-forward per race
        train_idx = int(len(races) * 0.7)
        train_races = races[:train_idx]
        test_races = races[train_idx:]
        
        train_data = arch_df[arch_df.set_index(['Year', 'RoundNumber']).index.isin(train_races)]
        test_data = arch_df[arch_df.set_index(['Year', 'RoundNumber']).index.isin(test_races)]
        
        X_train, y_train = train_data[features], train_data['Won']
        X_test, y_test = test_data[features], test_data['Won']
        
        # Level 1 Model: LGBM Ranker
        print("Fitting LGBM Ranker...")
        lgbm = lgb.LGBMRanker(n_estimators=200, learning_rate=0.05, importance_type='gain')
        # Queries are race groups
        train_queries = train_data.groupby(['Year', 'RoundNumber']).size().values
        lgbm.fit(X_train, y_train, group=train_queries)
        
        # Level 1 Model: XGB Ranker
        print("Fitting XGB Ranker...")
        xgb_rank = xgb.XGBRanker(n_estimators=200, learning_rate=0.05, objective='rank:pairwise')
        xgb_rank.fit(X_train, y_train, group=train_queries)
        
        # Level 1 Model: Logistic (Plackett-Luce proxy)
        print("Fitting Logistic Baseline...")
        pl_model = LogisticRegression()
        pl_model.fit(X_train, y_train)
        
        # Level 2 Meta-Learner (Stacking)
        print("Meta-stacking...")
        train_preds = np.column_stack([
            lgbm.predict(X_train),
            xgb_rank.predict(X_train),
            pl_model.predict_proba(X_train)[:, 1],
            train_data['RelElo']
        ])
        
        meta_learner = LogisticRegression()
        meta_learner.fit(train_preds, y_train)
        
        # Calibration (Isotonic)
        print("Calibrating...")
        calibrator = CalibratedClassifierCV(meta_learner, method='isotonic', cv='prefit')
        calibrator.fit(train_preds, y_train)
        
        # Uncertainty (MAPIE)
        print("Applying Mapie...")
        mapie = MapieClassifier(estimator=calibrator, method='score', cv='prefit')
        mapie.fit(train_preds, y_train)
        
        # Save trained expert
        trained_experts[arch] = {
            'lgbm': lgbm, 'xgb': xgb_rank, 'pl': pl_model,
            'meta': meta_learner, 'calibrator': calibrator, 'mapie': mapie
        }
        
        # Evaluation on Test
        test_preds_l1 = np.column_stack([
            lgbm.predict(X_test),
            xgb_rank.predict(X_test),
            pl_model.predict_proba(X_test)[:, 1],
            test_data['RelElo']
        ])
        
        final_probs, intervals = mapie.predict(test_preds_l1, alpha=0.1)
        # mapie.predict for classification returns sets, but we want probabilities.
        # we'll take the calibrated probs from the underlying estimator
        final_probs = calibrator.predict_proba(test_preds_l1)[:, 1]
        
        test_data['Prob'] = final_probs
        all_results.append(test_data)

    # 6. Global Evaluation Metrics
    results_df = pd.concat(all_results)
    
    # Calculate Metrics
    y_true = results_df['Won']
    y_prob = results_df['Prob']
    
    # Top-1 Hit Rate
    top1 = results_df.sort_values(['Year', 'RoundNumber', 'Prob'], ascending=[True, True, False])
    winners = top1.groupby(['Year', 'RoundNumber']).head(1)
    top1_hit = winners['Won'].mean()
    
    metrics = {
        'Top1 Hit Rate': f"{top1_hit*100:.1f}%",
        'ROC AUC': f"{roc_auc_score(y_true, y_prob):.3f}",
        'Avg Precision': f"{average_precision_score(y_true, y_prob):.3f}",
        'Brier Score': f"{brier_score_loss(y_true, y_prob):.4f}",
        'Log Loss': f"{log_loss(y_true, y_prob):.4f}"
    }
    
    print("\n--- FINAL GLOBAL PERFORMANCE (OOF) ---")
    for k, v in metrics.items():
        print(f"{k:<20}: {v}")

    # 7. Persistence
    outputs = {
        'experts': trained_experts,
        'archetype_map': archetype_map,
        'final_elos': df.groupby('FullName')['Elo'].last().to_dict(),
        'feature_medians': df[features].median().to_dict(),
        'features': features
    }
    
    with open('f1_v2_model.pkl', 'wb') as f:
        pickle.dump(outputs, f)
    
    print("\nF1 Predictor v2 training complete. Models saved to f1_v2_model.pkl")

if __name__ == "__main__":
    train_pipeline()