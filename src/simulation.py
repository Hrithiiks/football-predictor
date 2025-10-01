import pandas as pd
import joblib
import numpy as np
from tqdm import tqdm
from collections import deque


def get_dynamic_features(sim_table, team_form, home_team, away_team):
    """
    Calculates features for a match based on the CURRENT state of the simulation.
    """
    # Create a temporary league table from the current simulation to get ranks
    temp_table = pd.DataFrame.from_dict(sim_table, orient='index')
    temp_table['goal_difference'] = temp_table['goals_for'] - temp_table['goals_against']
    temp_table = temp_table.sort_values(['points', 'goal_difference', 'goals_for'], ascending=False).reset_index()
    temp_table['rank'] = temp_table.index + 1
    temp_table.set_index('index', inplace=True)

    home_rank = temp_table.loc[home_team, 'rank']
    away_rank = temp_table.loc[away_team, 'rank']

    # Get form from the deque (a list that automatically keeps the last 5 results)
    home_form_points = np.mean([p for p, gf, ga in team_form.get(home_team, [])]) if team_form.get(home_team) else 0
    away_form_points = np.mean([p for p, gf, ga in team_form.get(away_team, [])]) if team_form.get(away_team) else 0
    home_form_gd = np.mean([gf - ga for p, gf, ga in team_form.get(home_team, [])]) if team_form.get(home_team) else 0
    away_form_gd = np.mean([gf - ga for p, gf, ga in team_form.get(away_team, [])]) if team_form.get(away_team) else 0

    return {
        'home_form_points': home_form_points, 'home_form_goal_difference': home_form_gd,
        'away_form_points': away_form_points, 'away_form_goal_difference': away_form_gd,
        'home_rank': home_rank, 'home_points': sim_table[home_team]['points'],
        'away_rank': away_rank, 'away_points': sim_table[away_team]['points'],
        'h2h_home_avg_pts': 0, 'h2h_away_avg_pts': 0,  # H2H is too complex for this dynamic sim
        'odds_home_win': 0, 'odds_draw': 0, 'odds_away_win': 0  # No odds for future games
    }


def run_simulation(num_simulations=1000):
    """
    Runs a DYNAMIC Monte Carlo simulation for the rest of the season.
    """
    model = joblib.load('models/model.joblib')
    df = pd.read_csv('data/processed/matches_with_features.csv')
    df['date'] = pd.to_datetime(df['date'])

    current_season = df['season'].max()
    season_df = df[df['season'] == current_season]

    # --- Establish the starting point (current reality) ---
    teams = pd.unique(season_df[['home_team', 'away_team']].values.ravel('K'))
    initial_table = {team: {'points': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'goals_for': 0, 'goals_against': 0} for
                     team in teams}
    initial_form = {team: deque(maxlen=5) for team in teams}

    for _, row in season_df.iterrows():
        ht, at = row['home_team'], row['away_team']
        initial_table[ht]['goals_for'] += row['home_goals'];
        initial_table[ht]['goals_against'] += row['away_goals']
        initial_table[at]['goals_for'] += row['away_goals'];
        initial_table[at]['goals_against'] += row['home_goals']
        if row['result'] == 'HOME_TEAM':
            initial_table[ht]['points'] += 3;
            initial_table[ht]['wins'] += 1;
            initial_table[at]['losses'] += 1
            initial_form[ht].append((3, row['home_goals'], row['away_goals']));
            initial_form[at].append((0, row['away_goals'], row['home_goals']))
        elif row['result'] == 'AWAY_TEAM':
            initial_table[at]['points'] += 3;
            initial_table[at]['wins'] += 1;
            initial_table[ht]['losses'] += 1
            initial_form[ht].append((0, row['home_goals'], row['away_goals']));
            initial_form[at].append((3, row['away_goals'], row['home_goals']))
        else:
            initial_table[ht]['points'] += 1;
            initial_table[at]['points'] += 1;
            initial_table[ht]['draws'] += 1;
            initial_table[at]['draws'] += 1
            initial_form[ht].append((1, row['home_goals'], row['away_goals']));
            initial_form[at].append((1, row['away_goals'], row['home_goals']))

    # --- Get remaining fixtures, sorted by date ---
    # --- Get remaining fixtures, sorted by date ---
    all_fixtures_df = pd.read_csv('data/raw/all_avail_games.csv', encoding='latin1')

    # FIX 1: Filter using the correct column 'Div' and code 'E0'
    season_fixtures = all_fixtures_df[all_fixtures_df['Div'] == 'E0'].copy()

    # FIX 2: Rename columns to the lowercase format used by the rest of the script
    season_fixtures.rename(columns={'Date': 'date', 'HomeTeam': 'home_team', 'AwayTeam': 'away_team'}, inplace=True)

    # FIX 3: Convert date column correctly (dd/mm/yy format) and filter by year
    season_fixtures['date'] = pd.to_datetime(season_fixtures['date'], dayfirst=True)
    season_fixtures = season_fixtures[season_fixtures['date'].dt.year == current_season]

    # Now, finding the remaining fixtures will work correctly with consistent column names
    played_matches_str = season_df.apply(lambda r: f"{r['home_team']}-{r['away_team']}", axis=1)
    remaining_fixtures = season_fixtures[
        ~season_fixtures.apply(lambda r: f"{r['home_team']}-{r['away_team']}", axis=1).isin(played_matches_str)].copy()
    remaining_fixtures.sort_values('date', inplace=True)

    # --- Run Dynamic Simulations ---
    final_standings = []
    for _ in tqdm(range(num_simulations), desc="Running Dynamic Simulations"):
        sim_table = {team: stats.copy() for team, stats in initial_table.items()}
        sim_form = {team: form.copy() for team, form in initial_form.items()}

        for _, fixture in remaining_fixtures.iterrows():
            ht, at = fixture['home_team'], fixture['away_team']
            if ht not in sim_table or at not in sim_table: continue  # Skip teams not in the league this season

            features = get_dynamic_features(sim_table, sim_form, ht, at)
            input_df = pd.DataFrame([features])
            probabilities = model.predict_proba(input_df)[0]
            outcome = np.random.choice([0, 1, 2], p=probabilities)

            home_goals, away_goals = (1, 0) if outcome == 1 else (0, 1) if outcome == 2 else (1, 1)

            sim_table[ht]['goals_for'] += home_goals;
            sim_table[ht]['goals_against'] += away_goals
            sim_table[at]['goals_for'] += away_goals;
            sim_table[at]['goals_against'] += home_goals
            if outcome == 1:
                sim_table[ht]['points'] += 3;
                sim_table[ht]['wins'] += 1;
                sim_table[at]['losses'] += 1
                sim_form[ht].append((3, home_goals, away_goals));
                sim_form[at].append((0, away_goals, home_goals))
            elif outcome == 2:
                sim_table[at]['points'] += 3;
                sim_table[at]['wins'] += 1;
                sim_table[ht]['losses'] += 1
                sim_form[ht].append((0, home_goals, away_goals));
                sim_form[at].append((3, away_goals, home_goals))
            else:
                sim_table[ht]['points'] += 1;
                sim_table[at]['points'] += 1;
                sim_table[ht]['draws'] += 1;
                sim_table[at]['draws'] += 1
                sim_form[ht].append((1, home_goals, away_goals));
                sim_form[at].append((1, away_goals, home_goals))

        final_sim_df = pd.DataFrame.from_dict(sim_table, orient='index')
        final_sim_df['team'] = final_sim_df.index
        final_sim_df['goal_difference'] = final_sim_df['goals_for'] - final_sim_df['goals_against']
        final_sim_df = final_sim_df.sort_values(['points', 'goal_difference', 'goals_for'],
                                                ascending=False).reset_index(drop=True)
        final_sim_df['rank'] = final_sim_df.index + 1
        final_standings.append(final_sim_df)

    # --- Aggregate Results ---
    all_sims_df = pd.concat(final_standings)
    summary = all_sims_df.groupby('team').agg(
        avg_rank=('rank', 'mean'), avg_points=('points', 'mean'),
        avg_wins=('wins', 'mean'), avg_draws=('draws', 'mean'), avg_losses=('losses', 'mean')
    ).round(1)
    summary['chance_to_win_league'] = all_sims_df[all_sims_df['rank'] == 1].groupby('team').size().div(
        num_simulations).mul(100).fillna(0)
    summary['chance_for_top_4'] = all_sims_df[all_sims_df['rank'] <= 4].groupby('team').size().div(num_simulations).mul(
        100).fillna(0)
    summary['chance_of_relegation'] = all_sims_df[all_sims_df['rank'] >= 18].groupby('team').size().div(
        num_simulations).mul(100).fillna(0)
    return summary.fillna(0).sort_values('avg_rank').reset_index()


if __name__ == '__main__':
    # You may need to install tqdm: conda install -c conda-forge tqdm -y
    final_table = run_simulation(num_simulations=100)  # Keep it low for testing
    print(final_table)