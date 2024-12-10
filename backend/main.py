import requests
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from data_visualization import save_plot_images  # Import the new visualization function
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

api_key = ""

class RiotAPIHeaders:
    def __init__(self, api_key):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
            "Accept-Language": "ko,en-US;q=0.9,en;q=0.8,zh-CN;q=0.7,zh;q=0.6",
            "Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8",
            "Origin": "https://developer.riotgames.com",
            "X-Riot-Token": api_key
        }
# Copy ALL the original functions from your prototype.py here:
# get_player_data, get_summoner_data, match_finder, match_data, 
# collect_match_data, save_data_to_csv functions - DO NOT MODIFY THEM

def get_player_data(api_key, name, tag):
    base_url = "https://asia.api.riotgames.com/riot/account/v1/accounts/by-riot-id"
    encoded_name = name.replace(" ", "%20")
    url = f"{base_url}/{encoded_name}/{tag}?api_key={api_key}"

    headers = RiotAPIHeaders(api_key).headers

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        player_data = response.json()
        return player_data['puuid']
    else:
        print(f"Error: Unable to fetch player data. Status code: {response.status_code}, Response: {response.text}")
        return None

def get_summoner_data(api_key, puuid):
    headers = RiotAPIHeaders(api_key).headers
    base_url = "https://jp1.api.riotgames.com/lol/summoner/v4/summoners/by-puuid"
    url = f"{base_url}/{puuid}?api_key={api_key}"

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: Unable to fetch summoner data. Status code: {response.status_code}, Response: {response.text}")
        return None

def match_finder(api_key, puuid):
    headers = RiotAPIHeaders(api_key).headers
    base_url = "https://asia.api.riotgames.com/lol/match/v5/matches/by-puuid"
    # count=50으로 변경
    url = f"{base_url}/{puuid}/ids?start=0&count=50"
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        matches = response.json()
        return matches
    else:
        print(f"Error: Unable to fetch match data. Status code: {response.status_code}, Response: {response.text}")
        return None

def match_data(api_key, match_id):
    headers = RiotAPIHeaders(api_key).headers
    
    base_url = "https://asia.api.riotgames.com/lol/match/v5/matches"
    url = f"{base_url}/{match_id}?api_key={api_key}"

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        response_match_data = response.json()  # FIX: Use json() method to parse the response
        return response_match_data
    else:
        print(f"Error: Unable to fetch match data. Status code: {response.status_code}, Response: {response.text}")
        return None 

def collect_match_data(api_key, match_id, puuid):
    match_details = match_data(api_key, match_id)
    if match_details:
        participants = match_details['info']['participants']
        # Find the participant corresponding to the given puuid
        participant = next((p for p in participants if p['puuid'] == puuid), None)
        if participant:
            challenges = participant.get('challenges', {})
            # Extract and print each field explicitly
            kills = participant.get('kills', 0)
            deaths = participant.get('deaths', 0)
            assists = participant.get('assists', 0)
            game_duration = match_details['info'].get('gameDuration', 0)
            game_creation = match_details['info'].get('gameCreation', 0)
            damage_dealt = participant.get('totalDamageDealtToChampions', 0)
            damage_taken = participant.get('totalDamageTaken', 0)
            gold_earned = participant.get('goldEarned', 0)
            wards_placed = participant.get('wardsPlaced', 0)
            wards_destroyed = participant.get('wardsKilled', 0)
            creep_score = participant.get('totalMinionsKilled', 0)
            damage_per_minute = challenges.get('damagePerMinute', 0)
            gold_per_minute = challenges.get('goldPerMinute', 0)

            # Print each value to verify

            return {
                'match_id': match_id,
                'kills': kills,
                'deaths': deaths,
                'assists': assists,
                'game_duration': game_duration,
                'game_creation': game_creation,
                'damage_dealt': damage_dealt,
                'damage_taken': damage_taken,
                'gold_earned': gold_earned,
                'wards_placed': wards_placed,
                'wards_destroyed': wards_destroyed,
                'creep_score': creep_score,
                'damage_per_minute': damage_per_minute,
                'gold_per_minute': gold_per_minute
            }
        else:
            print(f"Error: Participant with PUUID {puuid} not found in match {match_id}")
    return None

def save_data_to_csv(match_data_list, filename="match_data.csv"):
    # Define all possible columns with default values
    columns = [
        'match_id', 'kills', 'deaths', 'assists', 'game_duration', 'game_creation',
        'damage_dealt', 'damage_taken', 'gold_earned', 'wards_placed',
        'wards_destroyed', 'creep_score', 'damage_per_minute', 'gold_per_minute',
        'damage_mitigated', 'gold_spent', 'vision_score', 'control_wards_purchased',
        'cs_per_minute', 'gold_diff_10', 'xp_diff_10', 'towers_destroyed',
        'dragons_secured', 'rift_herald_secured', 'barons_secured',
        'objective_control_percentage', 'crowd_control_score', 'healing_done',
        'shielding_done', 'damage_to_structures', 'kill_participation',
        'damage_percentage', 'vision_score_per_minute', 'average_time_spent_dead',
        'game_outcome', 'performance_grade', 'team_contribution', 'pings_used',
        'chat_messages', 'player_role', 'champion_played', 'ranked_tier'
    ]
    
    # Create DataFrame with all columns, using 0 as default value for missing data
    df = pd.DataFrame(match_data_list, columns=columns).fillna(0)

    # Set explicit data types for consistency
    dtype_mapping = {
        'match_id': 'str',
        'kills': 'int64',
        'deaths': 'int64',
        'assists': 'int64',
        'game_duration': 'int64',
        'game_creation': 'int64',
        'damage_dealt': 'int64',
        'damage_taken': 'int64',
        'gold_earned': 'int64',
        'wards_placed': 'int64',
        'wards_destroyed': 'int64',
        'creep_score': 'int64',
        'damage_per_minute': 'float64',
        'gold_per_minute': 'float64',
        'damage_mitigated': 'int64',
        'gold_spent': 'int64',
        'vision_score': 'int64',
        'control_wards_purchased': 'int64',
        'cs_per_minute': 'float64',
        'gold_diff_10': 'int64',
        'xp_diff_10': 'int64',
        'towers_destroyed': 'int64',
        'dragons_secured': 'int64',
        'rift_herald_secured': 'int64',
        'barons_secured': 'int64',
        'objective_control_percentage': 'float64',
        'crowd_control_score': 'int64',
        'healing_done': 'int64',
        'shielding_done': 'int64',
        'damage_to_structures': 'int64',
        'kill_participation': 'float64',
        'damage_percentage': 'float64',
        'vision_score_per_minute': 'float64',
        'average_time_spent_dead': 'float64',
        'game_outcome': 'str',
        'performance_grade': 'str',
        'team_contribution': 'float64',
        'pings_used': 'int64',
        'chat_messages': 'int64',
        'player_role': 'str',
        'champion_played': 'str',
        'ranked_tier': 'str'
    }
    df = df.astype(dtype_mapping)

    try:
        if not os.path.exists(filename):
            # Save new CSV if file does not exist
            df.to_csv(filename, index=False)
        else:
            # Append to existing CSV, avoiding duplicates by match_id
            existing_df = pd.read_csv(filename)
            existing_match_ids = existing_df['match_id'].tolist()
            df = df[~df['match_id'].isin(existing_match_ids)]
            if not df.empty:
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df.to_csv(filename, index=False)
            else:
                print("No new matches to add.")
    except Exception as e:
        print(f"Error saving data to CSV: {e}")
    print(f"Data saved to {filename}")

def visualize_data(filename="match_data.csv"):
    try:
        df = pd.read_csv(filename, on_bad_lines='skip')
    except pd.errors.ParserError as e:
        print(f"Error reading the CSV file: {e}")
        return

    # Print the DataFrame to inspect the data before plotting
    print("Data from CSV:")
    print(df.head())

    df['game_creation'] = pd.to_datetime(df['game_creation'], unit='ms')
    df['KDA'] = (df['kills'] + df['assists']) / df['deaths'].replace(0, 1)

    metrics = {
        'KDA': ('KDA', 'KDA Ratio', 'KDA Trend Over Time'),
        'damage_dealt': ('damage_dealt', 'Damage Dealt to Champions', 'Damage Dealt to Champions Over Time', 'r'),
        'damage_taken': ('damage_taken', 'Damage Taken', 'Damage Taken Over Time', 'g'),
        'gold_earned': ('gold_earned', 'Gold Earned', 'Gold Earned Over Time', 'y'),
        'wards_placed': ('wards_placed', 'Wards Placed', 'Wards Placed Over Time', 'b'),
        'creep_score': ('creep_score', 'Creep Score', 'Creep Score Over Time', 'm'),
        'damage_per_minute': ('damage_per_minute', 'Damage Per Minute', 'Damage Per Minute Over Time', 'c'),
        'gold_per_minute': ('gold_per_minute', 'Gold Per Minute', 'Gold Per Minute Over Time', 'orange'),
        'damage_mitigated': ('damage_mitigated', 'Damage Mitigated', 'Damage Mitigated Over Time'),
        'vision_score': ('vision_score', 'Vision Score', 'Vision Score Over Time'),
        'control_wards_purchased': ('control_wards_purchased', 'Control Wards Purchased', 'Control Wards Purchased Over Time'),
        'cs_per_minute': ('cs_per_minute', 'CS Per Minute', 'CS Per Minute Over Time'),
        'gold_diff_10': ('gold_diff_10', 'Gold Difference at 10 Minutes', 'Gold Difference at 10 Minutes Over Time'),
        'xp_diff_10': ('xp_diff_10', 'XP Difference at 10 Minutes', 'XP Difference at 10 Minutes Over Time'),
        'towers_destroyed': ('towers_destroyed', 'Towers Destroyed', 'Towers Destroyed Over Time'),
        'dragons_secured': ('dragons_secured', 'Dragons Secured', 'Dragons Secured Over Time'),
        'rift_herald_secured': ('rift_herald_secured', 'Rift Heralds Secured', 'Rift Heralds Secured Over Time'),
        'barons_secured': ('barons_secured', 'Barons Secured', 'Barons Secured Over Time'),
        'crowd_control_score': ('crowd_control_score', 'Crowd Control Score', 'Crowd Control Score Over Time'),
        'healing_done': ('healing_done', 'Healing Done', 'Healing Done Over Time'),
        'shielding_done': ('shielding_done', 'Shielding Done', 'Shielding Done Over Time'),
        'damage_to_structures': ('damage_to_structures', 'Damage to Structures', 'Damage to Structures Over Time'),
        'kill_participation': ('kill_participation', 'Kill Participation', 'Kill Participation Over Time'),
        'damage_percentage': ('damage_percentage', 'Damage Percentage', 'Damage Percentage Over Time'),
        'vision_score_per_minute': ('vision_score_per_minute', 'Vision Score Per Minute', 'Vision Score Per Minute Over Time')
    }

    for key, (metric, ylabel, title, *color) in metrics.items():
        if metric in df.columns and df[metric].sum() > 0:
            print(f"Plotting {metric} with total sum: {df[metric].sum()}")
            plt.figure(figsize=(10, 6))
            plt.plot(df['game_creation'], df[metric], marker='o', linestyle='-', color=color[0] if color else None, label=f'{title} Trend')
            plt.xlabel('Game Creation Date')
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            print(f"No data available for {metric} to plot.")

def riemann_prediction(time_points, values, future_points=5):
    """리만 제타 함수를 기반으로 한 예측 모델"""
    try:
        x = np.arange(len(values), dtype=float)
        y = np.array(values, dtype=float)
        
        # 데이터 정규화
        x_mean = np.mean(x)
        x_std = np.std(x) if np.std(x) != 0 else 1
        y_mean = np.mean(y)
        y_std = np.std(y) if np.std(y) != 0 else 1
        
        x_norm = (x - x_mean) / x_std
        y_norm = (y - y_mean) / y_std

        # 2차 다항식으로 피팅 (더 안정적인 예측을 위해)
        coeffs = np.polyfit(x_norm, y_norm, 2)
        poly = np.poly1d(coeffs)

        # 미래 시점 예측
        future_x = np.arange(len(values), len(values) + future_points, dtype=float)
        future_x_norm = (future_x - x_mean) / x_std
        predictions_norm = poly(future_x_norm)
        
        # 역정규화
        predictions = predictions_norm * y_std + y_mean

        # 신뢰구간 계산 (68% 신뢰구간)
        residuals = y - (poly(x_norm) * y_std + y_mean)
        std_dev = np.std(residuals)
        confidence_interval = (
            predictions - std_dev,
            predictions + std_dev
        )

        return {
            'predictions': predictions.tolist(),
            'confidence_lower': confidence_interval[0].tolist(),
            'confidence_upper': confidence_interval[1].tolist()
        }
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return None

# Flask app for frontend integration
app = Flask(__name__)
CORS(app)

@app.route('/api/analyze-matches', methods=['POST'])
def analyze_matches():
    data = request.json
    name = data.get('name')
    tag = data.get('tag')
    
    # Replace with your Riot Games API key
    api_key = ""
    
    try:
        puuid = get_player_data(api_key, name, tag)
        if not puuid:
            return jsonify({'error': 'Player not found'}), 404
        
        match_ids = match_finder(api_key, puuid)
        if not match_ids:
            return jsonify({'error': 'No matches found'}), 404
            
        classic_matches = []
        for match_id in match_ids:
            match_details = match_data(api_key, match_id)
            if match_details and match_details['info']['gameMode'] == 'CLASSIC':
                classic_matches.append(match_id)
        
        match_data_list = []
        for match_id in classic_matches:
            match_info = collect_match_data(api_key, match_id, puuid)
            if match_info:
                match_data_list.append(match_info)

        if match_data_list:
            # KDA 계산을 각 매치 데이터에 추가
            for match in match_data_list:
                deaths = match['deaths'] if match['deaths'] > 0 else 1
                match['KDA'] = (match['kills'] + match['assists']) / deaths
            
            # 예측할 메트릭 리스트에 KDA 추가
            metrics = ['KDA', 'kills', 'deaths', 'assists', 'damage_dealt', 'gold_earned']
            predictions = {}
            
            for metric in metrics:
                values = [match[metric] for match in match_data_list]
                pred = riemann_prediction(range(len(values)), values)
                if pred:
                    predictions[metric] = pred
            
            # 저장 및 시각화 로직
            save_data_to_csv(match_data_list)
            save_plot_images("match_data.csv")
            
            print("Sending data to frontend:", len(match_data_list), "matches")
            return jsonify({
                'matches': match_data_list,
                'predictions': predictions
            })
        
        return jsonify({'error': 'No match data collected'}), 404
    
    except Exception as e:
        print("Error in analyze_matches:", str(e))
        return jsonify({'error': str(e)}), 500
@app.route('/plots/<path:filename>')
def serve_plot(filename):
    """Serve plot images to the frontend"""
    return send_from_directory('plots', filename)


if __name__ == "__main__":
    # Ensure plots directory exists
    os.makedirs('plots', exist_ok=True)
    app.run(debug=True, port=5000)
