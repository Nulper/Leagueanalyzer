import requests
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
import json
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from data_visualization import save_plot_images
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import openai
import sys
import asyncio  # Add this import
from functools import partial

api_key = ""
openai_api_key = ""

# 로깅 설정
def setup_logging():
    os.makedirs('logs', exist_ok=True)
    log_filename = f'logs/app_{datetime.datetime.now().strftime("%Y%m%d")}.log'
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

if getattr(sys, 'frozen', False):
    application_path = sys._MEIPASS
else:
    application_path = os.path.dirname(os.path.abspath(__file__))


class RiotAPIHeaders:
    def __init__(self, api_key):
        logger.info("Initializing RiotAPIHeaders")
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
            "Accept-Language": "ko,en-US;q=0.9,en;q=0.8,zh-CN;q=0.7,zh;q=0.6",
            "Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8",
            "Origin": "https://developer.riotgames.com",
            "X-Riot-Token": api_key
        }

async def get_openai_response(prompt):
    logger.info("Requesting OpenAI response")
    openai.api_key = openai_api_key
    try:
        logger.info("Starting OpenAI request")
        
        # Create a ClientSession for async HTTP requests
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: openai.chat.completions.create(  # Using create instead of acreate
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes League of Legends match data. Provide detailed analysis and champion recommendations. Be thorough but concise."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.7,
                presence_penalty=0.1,
                frequency_penalty=0.1
            )
        )
        
        logger.info("Successfully received OpenAI response")
        content = response.choices[0].message.content
        logger.info(f"Response received: {len(content)} characters")
        return content

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return f"Unexpected error: {str(e)}"

def get_player_data(api_key, name, tag):
    logger.info(f"Fetching player data for {name}#{tag}")
    base_url = "https://asia.api.riotgames.com/riot/account/v1/accounts/by-riot-id"
    encoded_name = name.replace(" ", "%20")
    url = f"{base_url}/{encoded_name}/{tag}?api_key={api_key}"

    headers = RiotAPIHeaders(api_key).headers

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        player_data = response.json()
        logger.info(f"Successfully retrieved player data for {name}#{tag}")
        return player_data['puuid']
    else:
        logger.error(f"Error: Unable to fetch player data. Status code: {response.status_code}, Response: {response.text}")
        return None

def get_summoner_data(api_key, puuid):
    logger.info(f"Fetching summoner data for PUUID: {puuid}")
    headers = RiotAPIHeaders(api_key).headers
    base_url = "https://jp1.api.riotgames.com/lol/summoner/v4/summoners/by-puuid"
    url = f"{base_url}/{puuid}?api_key={api_key}"

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        logger.info("Successfully retrieved summoner data")
        return response.json()
    else:
        logger.error(f"Error: Unable to fetch summoner data. Status code: {response.status_code}, Response: {response.text}")
        return None

def match_finder(api_key, puuid):
    logger.info(f"Finding matches for PUUID: {puuid}")
    headers = RiotAPIHeaders(api_key).headers
    base_url = "https://asia.api.riotgames.com/lol/match/v5/matches/by-puuid"
    url = f"{base_url}/{puuid}/ids?start=0&count=50"
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        matches = response.json()
        logger.info(f"Successfully found {len(matches)} matches")
        return matches
    else:
        logger.error(f"Error: Unable to fetch match data. Status code: {response.status_code}, Response: {response.text}")
        return None

def match_data(api_key, match_id):
    logger.info(f"Fetching data for match ID: {match_id}")
    headers = RiotAPIHeaders(api_key).headers
    
    base_url = "https://asia.api.riotgames.com/lol/match/v5/matches"
    url = f"{base_url}/{match_id}?api_key={api_key}"

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        response_match_data = response.json()
        logger.info(f"Successfully retrieved data for match {match_id}")
        return response_match_data
    else:
        logger.error(f"Error: Unable to fetch match data. Status code: {response.status_code}, Response: {response.text}")
        return None

def collect_match_data(api_key, match_id, puuid):
    logger.info(f"Collecting match data for match ID: {match_id}, PUUID: {puuid}")
    match_details = match_data(api_key, match_id)
    if match_details:
        participants = match_details['info']['participants']
        participant = next((p for p in participants if p['puuid'] == puuid), None)
        if participant:
            challenges = participant.get('challenges', {})
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

            logger.info(f"Successfully collected match data for match {match_id}")
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
            logger.error(f"Error: Participant with PUUID {puuid} not found in match {match_id}")
    return None

def save_data_to_csv(match_data_list, filename="match_data.csv"):
    logger.info(f"Saving match data to CSV: {filename}")
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
    
    df = pd.DataFrame(match_data_list, columns=columns).fillna(0)

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
            df.to_csv(filename, index=False)
            logger.info(f"Created new CSV file: {filename}")
        else:
            existing_df = pd.read_csv(filename)
            existing_match_ids = existing_df['match_id'].tolist()
            df = df[~df['match_id'].isin(existing_match_ids)]
            if not df.empty:
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df.to_csv(filename, index=False)
                logger.info(f"Updated existing CSV with {len(df)} new matches")
            else:
                logger.info("No new matches to add")
    except Exception as e:
        logger.error(f"Error saving data to CSV: {e}")
    print(f"Data saved to {filename}")

def visualize_data(filename="match_data.csv"):
    global logger
    
    logger.info(f"Visualizing data from {filename}")
    try:
        df = pd.read_csv(filename, on_bad_lines='skip')
    except pd.errors.ParserError as e:
        logger.error(f"Error reading CSV file: {e}")
        return

    logger.info("Data loaded successfully, starting visualization")
    
    # 매치 아이디 숫자로 변환 (예: KR_12345 -> 12345)
    df['match_number'] = df['match_id'].str.extract('(\d+)').astype(int)
    df = df.sort_values('match_number')  # 매치 번호로 정렬
    df['KDA'] = (df['kills'] + df['assists']) / df['deaths'].replace(0, 1)

    metrics = {
        'KDA': ('KDA', 'KDA Ratio', 'KDA Trend Over Matches'),
        'damage_dealt': ('damage_dealt', 'Damage Dealt to Champions', 'Damage Dealt Over Matches', 'r'),
        'damage_taken': ('damage_taken', 'Damage Taken', 'Damage Taken Over Matches', 'g'),
        'gold_earned': ('gold_earned', 'Gold Earned', 'Gold Earned Over Matches', 'y'),
        'wards_placed': ('wards_placed', 'Wards Placed', 'Wards Placed Over Matches', 'b'),
        'creep_score': ('creep_score', 'Creep Score', 'Creep Score Over Matches', 'm'),
        'damage_per_minute': ('damage_per_minute', 'Damage Per Minute', 'Damage Per Minute Over Matches', 'c'),
        'gold_per_minute': ('gold_per_minute', 'Gold Per Minute', 'Gold Per Minute Over Matches', 'orange')
    }

    for key, (metric, ylabel, title, *color) in metrics.items():
        logger.info(f"Plotting {metric}")
        if metric in df.columns and df[metric].sum() > 0:
            logger.info(f"Plotting {metric} with total sum: {df[metric].sum()}")
            plt.figure(figsize=(12, 6))
            
            # 플롯 생성
            plt.plot(range(len(df)), df[metric], marker='o', linestyle='-', 
                    color=color[0] if color else None, label=f'{title}')
            
            # x축 설정
            plt.xticks(range(len(df)), [f'Match {i+1}' for i in range(len(df))], 
                      rotation=45, ha='right')
            
            # 그리드, 레이블 등 설정
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlabel('Match Number')
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend()
            
            # 여백 조정
            plt.tight_layout()
            plt.show()
        else:
            logger.warning(f"No data available for {metric} to plot")

def riemann_prediction(time_points, values, future_points=5):
    logger.info("Starting Riemann prediction calculation")
    try:
        x = np.arange(len(values), dtype=float)
        y = np.array(values, dtype=float)
        
        x_mean = np.mean(x)
        x_std = np.std(x) if np.std(x) != 0 else 1
        y_mean = np.mean(y)
        y_std = np.std(y) if np.std(y) != 0 else 1
        
        x_norm = (x - x_mean) / x_std
        y_norm = (y - y_mean) / y_std

        coeffs = np.polyfit(x_norm, y_norm, 2)
        poly = np.poly1d(coeffs)

        future_x = np.arange(len(values), len(values) + future_points, dtype=float)
        future_x_norm = (future_x - x_mean) / x_std
        predictions_norm = poly(future_x_norm)
        
        predictions = predictions_norm * y_std + y_mean

        residuals = y - (poly(x_norm) * y_std + y_mean)
        std_dev = np.std(residuals)
        confidence_interval = (
            predictions - std_dev,
            predictions + std_dev
        )

        logger.info("Successfully completed Riemann prediction")
        return {
            'predictions': predictions.tolist(),
            'confidence_lower': confidence_interval[0].tolist(),
            'confidence_upper': confidence_interval[1].tolist()
        }
    except Exception as e:
        logger.error(f"Error in Riemann prediction: {str(e)}")
        return None

def calculate_performance_score(match_data):
    total_score = 0
    total_weight = 0

    criteria = {
        'gold_per_minute': (0.3, 150),
        'kda': (0.25, 125),
        'damage_dealt': (0.2, 100),
        'damage_taken': (0.15, 75),
        'vision_score': (0.1, 50)
    }

    for key, (weight, max_score) in criteria.items():
        if key in match_data:
            score = (match_data[key] / max_score) * weight * 100
            total_score += score
            total_weight += weight

    if total_weight > 0:
        performance_score = (total_score / total_weight) * 5
    else:
        performance_score = 0

    return performance_score

app = Flask(__name__, 
    static_folder=os.path.join(application_path, 'frontend', 'build'),
    static_url_path='/')
CORS(app)

@app.route('/api/analyze-matches', methods=['POST'])
def analyze_matches():
    logger.info("Received analyze-matches request")
    data = request.json
    name = data.get('name')
    tag = data.get('tag')
    
    logger.info(f"Analyzing matches for player: {name}#{tag}")
    
    try:
        puuid = get_player_data(api_key, name, tag)
        if not puuid:
            logger.error("Player not found")
            return jsonify({'error': 'Player not found'}), 404
        
        match_ids = match_finder(api_key, puuid)
        if not match_ids:
            logger.error("No matches found")
            return jsonify({'error': 'No matches found'}), 404
            
        classic_matches = []
        for match_id in match_ids:
            match_details = match_data(api_key, match_id)
            if match_details and match_details['info']['gameMode'] == 'CLASSIC':
                classic_matches.append(match_id)
        
        logger.info(f"Found {len(classic_matches)} classic matches")
        
        match_data_list = []
        for match_id in classic_matches:
            match_info = collect_match_data(api_key, match_id, puuid)
            if match_info:
                match_data_list.append(match_info)

        if match_data_list:
            for match in match_data_list:
                deaths = match['deaths'] if match['deaths'] > 0 else 1
                match['KDA'] = (match['kills'] + match['assists']) / deaths
            
            metrics = ['KDA', 'kills', 'deaths', 'assists', 'damage_dealt', 'gold_earned']
            predictions = {}
            
            logger.info("Calculating predictions for metrics")
            for metric in metrics:
                values = [match[metric] for match in match_data_list]
                pred = riemann_prediction(range(len(values)), values)
                if pred:
                    predictions[metric] = pred

            prompt = f"Analyze the following match data and recommend champions: {json.dumps(match_data_list)}"
            logger.info("Requesting AI analysis")
            
            # 비동기 함수를 동기적으로 실행
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            ai_response = loop.run_until_complete(get_openai_response(prompt))
            loop.close()
            
            save_data_to_csv(match_data_list)
            save_plot_images("match_data.csv")
            
            performance_scores = [calculate_performance_score(match) for match in match_data_list]
            average_performance_score = sum(performance_scores) / len(performance_scores) if performance_scores else 0
            
            logger.info(f"Analysis completed successfully for {name}#{tag}")
            return jsonify({
                'matches': match_data_list,
                'predictions': predictions,
                'ai_response': ai_response,
                'performance_score': average_performance_score
            })
        
        logger.error("No match data collected")
        return jsonify({'error': 'No match data collected'}), 404
    
    except Exception as e:
        logger.error(f"Error in analyze_matches: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/plots/<path:filename>')
def serve_plot(filename):
    logger.info(f"Serving plot: {filename}")
    return send_from_directory('plots', filename)

@app.route('/download-executable', methods=['GET'])
def download_executable():
    logger.info("Serving executable file")
    return send_from_directory('dist', 'main.exe')

@app.route('/')
def serve_frontend():
    logger.info("Serving frontend")
    logger.info(f"Application path: {application_path}")
    logger.info(f"Static folder path: {app.static_folder}")
    static_folder_path = os.path.join(app.static_folder, 'index.html')
    logger.info(f"Looking for index.html at: {static_folder_path}")
    logger.info(f"Static folder exists: {os.path.exists(app.static_folder)}")
    logger.info(f"Files in static folder: {os.listdir(app.static_folder) if os.path.exists(app.static_folder) else 'folder not found'}")
    logger.info(f"Index.html exists: {os.path.exists(static_folder_path)}")
    
    if os.path.exists(os.path.join(app.static_folder, 'index.html')):
        return send_from_directory(app.static_folder, 'index.html')
    else:
        logger.error(f"Frontend build not found at {static_folder_path}")
        return "Frontend build not found. Please run 'npm run build' in the 'frontend' directory.", 404

@app.route('/frontend/<path:filename>')
def serve_frontend_files(filename):
    return send_from_directory(os.path.join(application_path, 'frontend', 'build'), filename)

if __name__ == "__main__":
    logger.info("Starting application")
    os.makedirs('plots', exist_ok=True)
    logger.info("Application started successfully")
    # debug=False로 변경
    app.run(debug=False, port=5000, host='0.0.0.0')
