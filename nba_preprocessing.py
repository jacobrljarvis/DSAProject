"""
NBA Player Position Classification - Data Preprocessing
CPSC 322 Fall 2025
"""

import pandas as pd
import numpy as np

def load_nba_data(filepath):
    """
    Load NBA player statistics from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with NBA player stats
    """
    df = pd.read_csv(filepath, sep=';', encoding='latin-1')
    return df

def handle_traded_players(df):
    """
    Handle players who were traded (appear multiple times with different teams).
    Keep only the 'TOT' (total) row for traded players, or the single row if not traded.
    
    Args:
        df: DataFrame with NBA stats
        
    Returns:
        DataFrame with one row per player
    """
    # Find players with multiple entries
    player_counts = df['Player'].value_counts()
    traded_players = player_counts[player_counts > 1].index
    
    # For traded players, keep only TOT rows; for others, keep as is
    tot_rows = df[df['Tm'] == 'TOT'].copy()
    non_traded = df[~df['Player'].isin(traded_players)].copy()
    
    # Combine
    result = pd.concat([tot_rows, non_traded], ignore_index=True)
    
    print(f"Original rows: {len(df)}")
    print(f"After handling trades: {len(result)}")
    print(f"Traded players (with TOT stats): {len(tot_rows)}")
    
    return result

def simplify_positions(df):
    """
    Simplify hybrid positions to primary position.
    E.g., 'PG-SG' -> 'PG', 'C-PF' -> 'C'
    
    Args:
        df: DataFrame with Pos column
        
    Returns:
        DataFrame with simplified positions
    """
    df = df.copy()
    
    # Extract first position from hybrid positions (split on '-')
    df['Pos'] = df['Pos'].apply(lambda x: x.split('-')[0] if '-' in x else x)
    
    print("\nPosition distribution after simplification:")
    print(df['Pos'].value_counts())
    
    return df

def filter_low_playtime_players(df, min_games=10, min_minutes=10.0):
    """
    Filter out players with insufficient playing time for reliable stats.
    
    Args:
        df: DataFrame with NBA stats
        min_games: Minimum games played
        min_minutes: Minimum minutes per game
        
    Returns:
        Filtered DataFrame
    """
    df = df.copy()
    
    initial_count = len(df)
    df = df[(df['G'] >= min_games) & (df['MP'] >= min_minutes)]
    
    print(f"\nFiltered players with < {min_games} games or < {min_minutes} MPG")
    print(f"Removed: {initial_count - len(df)} players")
    print(f"Remaining: {len(df)} players")
    
    return df

def create_derived_features(df):
    """
    Create additional derived features that may be useful for classification.
    
    Args:
        df: DataFrame with NBA stats
        
    Returns:
        DataFrame with additional features
    """
    df = df.copy()
    
    # 3-point attempt rate (3PA per FGA)
    df['3P_rate'] = df['3PA'] / df['FGA'].replace(0, 1)  # Avoid division by zero
    
    # Rebound rate (TRB per game already exists, but let's add ORB%)
    df['ORB_rate'] = df['ORB'] / df['TRB'].replace(0, 1)
    
    # Assist-to-turnover ratio
    df['AST_TO_ratio'] = df['AST'] / df['TOV'].replace(0, 1)
    
    # Usage proxy (simple version based on FGA, FTA, TOV)
    df['Usage_proxy'] = (df['FGA'] + 0.44 * df['FTA'] + df['TOV']) / df['MP'].replace(0, 1)
    
    return df

def discretize_features(df, feature_cols, n_bins=10):
    """
    Discretize continuous features into bins for decision tree classifiers.
    
    Args:
        df: DataFrame with features
        feature_cols: List of column names to discretize
        n_bins: Number of bins for each feature
        
    Returns:
        DataFrame with discretized features
    """
    df_discretized = df.copy()
    
    for col in feature_cols:
        # Use quantile-based binning (equal frequency)
        df_discretized[col] = pd.qcut(df[col], q=n_bins, labels=False, duplicates='drop')
    
    return df_discretized

def select_features_for_classification(df, discretize=True, n_bins=10):
    """
    Select relevant features for position classification.
    Exclude player identifiers and team information.
    
    Args:
        df: DataFrame with all features
        discretize: Whether to discretize continuous features (for decision trees)
        n_bins: Number of bins for discretization
        
    Returns:
        X (features), y (labels), feature_names
    """
    # Features to use for classification
    feature_cols = [
        'PTS',      # Points per game
        'TRB',      # Total rebounds
        'AST',      # Assists
        'STL',      # Steals
        'BLK',      # Blocks
        'FG%',      # Field goal percentage
        '3P',       # 3-pointers made
        '3P%',      # 3-point percentage
        '3P_rate',  # 3-point attempt rate
        'FT%',      # Free throw percentage
        'TOV',      # Turnovers
        'MP',       # Minutes per game
    ]
    
    # Discretize features if requested
    if discretize:
        print(f"\nDiscretizing features into {n_bins} bins...")
        df_processed = discretize_features(df, feature_cols, n_bins)
    else:
        df_processed = df
    
    X = df_processed[feature_cols].values
    y = df['Pos'].values
    
    print(f"\nFeatures selected: {len(feature_cols)}")
    print(f"Feature names: {feature_cols}")
    print(f"Discretized: {discretize}")
    print(f"Class distribution:")
    print(df['Pos'].value_counts())
    
    return X, y, feature_cols

def preprocess_nba_data(filepath, min_games=10, min_minutes=10.0, discretize=True, n_bins=10):
    """
    Complete preprocessing pipeline for NBA data.
    
    Args:
        filepath: Path to CSV file
        min_games: Minimum games played
        min_minutes: Minimum minutes per game
        discretize: Whether to discretize continuous features (for decision trees)
        n_bins: Number of bins for discretization
        
    Returns:
        X (features), y (labels), feature_names, original_df
    """
    print("="*80)
    print("NBA DATA PREPROCESSING")
    print("="*80)
    
    # Load data
    print("\n1. Loading data...")
    df = load_nba_data(filepath)
    print(f"Loaded {len(df)} rows")
    
    # Handle traded players
    print("\n2. Handling traded players...")
    df = handle_traded_players(df)
    
    # Simplify positions
    print("\n3. Simplifying positions...")
    df = simplify_positions(df)
    
    # Filter low playtime
    print("\n4. Filtering low playtime players...")
    df = filter_low_playtime_players(df, min_games, min_minutes)
    
    # Create derived features
    print("\n5. Creating derived features...")
    df = create_derived_features(df)
    
    # Select features
    print("\n6. Selecting features for classification...")
    X, y, feature_names = select_features_for_classification(df, discretize=discretize, n_bins=n_bins)
    
    print("\n" + "="*80)
    print(f"PREPROCESSING COMPLETE")
    print(f"Final dataset: {X.shape[0]} instances, {X.shape[1]} features")
    print("="*80)
    
    return X, y, feature_names, df

if __name__ == "__main__":
    # Test the preprocessing pipeline
    filepath = '/mnt/user-data/uploads/2023-2024_NBA_Player_Stats_-_Regular.csv'
    X, y, feature_names, df = preprocess_nba_data(filepath)
    
    print("\nSample of preprocessed data:")
    print(df[['Player', 'Pos', 'PTS', 'TRB', 'AST', 'BLK']].head(10))
