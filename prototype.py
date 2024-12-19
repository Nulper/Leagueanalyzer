import requests
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os

class RiotAPIHeaders:
    def __init__(self, api_key):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
            "Accept-Language": "ko,en-US;q=0.9,en;q=0.8,zh-CN;q=0.7,zh;q=0.6",
            "Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8",
            "Origin": "https://developer.riotgames.com",
            "X-Riot-Token": api_key
        }

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
    url = f"{base_url}/{puuid}/ids?start=0&count=100&api_key={api_key}"

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
    global logger    # Use the global logger instance
    
    logger.info(f"Visualizing data from {filename}")
    try:
        df = pd.read_csv(filename, on_bad_lines='skip')
    except pd.errors.ParserError as e:
        logger.error(f"Error reading CSV file: {e}")
        return

    logger.info("Data loaded successfully, starting visualization")
    
    # 매치 번호순으로 정렬
    df = df.sort_values('game_creation')
    # 매치 인덱스 생성 (1부터 시작)
    df['match_number'] = range(1, len(df) + 1)
    df['KDA'] = (df['kills'] + df['assists']) / df['deaths'].replace(0, 1)

    metrics = {
        'KDA': ('KDA', 'KDA Ratio', 'KDA Trend Over Matches'),
        'damage_dealt': ('damage_dealt', 'Damage Dealt to Champions', 'Damage Dealt Over Matches', 'r'),
        'damage_taken': ('damage_taken', 'Damage Taken', 'Damage Taken Over Matches', 'g'),
        'gold_earned': ('gold_earned', 'Gold Earned', 'Gold Earned Over Matches', 'y'),
        'wards_placed': ('wards_placed', 'Wards Placed', 'Wards Placed Over Matches', 'b'),
        'creep_score': ('creep_score', 'Creep Score', 'Creep Score Over Matches', 'm'),
        'damage_per_minute': ('damage_per_minute', 'Damage Per Minute', 'Damage Per Minute Over Matches', 'c'),
        'gold_per_minute': ('gold_per_minute', 'Gold Per Minute', 'Gold Per Minute Over Matches', 'orange'),
        'damage_mitigated': ('damage_mitigated', 'Damage Mitigated', 'Damage Mitigated Over Matches'),
        'vision_score': ('vision_score', 'Vision Score', 'Vision Score Over Matches'),
        'control_wards_purchased': ('control_wards_purchased', 'Control Wards Purchased', 'Control Wards Purchased Over Matches'),
        'cs_per_minute': ('cs_per_minute', 'CS Per Minute', 'CS Per Minute Over Matches'),
        'gold_diff_10': ('gold_diff_10', 'Gold Difference at 10 Minutes', 'Gold Difference Over Matches'),
        'xp_diff_10': ('xp_diff_10', 'XP Difference at 10 Minutes', 'XP Difference Over Matches'),
        'towers_destroyed': ('towers_destroyed', 'Towers Destroyed', 'Towers Destroyed Over Matches'),
        'dragons_secured': ('dragons_secured', 'Dragons Secured', 'Dragons Secured Over Matches'),
        'rift_herald_secured': ('rift_herald_secured', 'Rift Heralds Secured', 'Rift Heralds Over Matches'),
        'barons_secured': ('barons_secured', 'Barons Secured', 'Barons Secured Over Matches'),
        'crowd_control_score': ('crowd_control_score', 'Crowd Control Score', 'Crowd Control Score Over Matches'),
        'healing_done': ('healing_done', 'Healing Done', 'Healing Done Over Matches'),
        'shielding_done': ('shielding_done', 'Shielding Done', 'Shielding Done Over Matches'),
        'damage_to_structures': ('damage_to_structures', 'Damage to Structures', 'Damage to Structures Over Matches'),
        'kill_participation': ('kill_participation', 'Kill Participation', 'Kill Participation Over Matches'),
        'damage_percentage': ('damage_percentage', 'Damage Percentage', 'Damage Percentage Over Matches'),
        'vision_score_per_minute': ('vision_score_per_minute', 'Vision Score Per Minute', 'Vision Score Per Minute Over Matches')
    }

    for key, (metric, ylabel, title, *color) in metrics.items():
        logger.info(f"Plotting {metric}")
        if metric in df.columns and df[metric].sum() > 0:
            logger.info(f"Plotting {metric} with total sum: {df[metric].sum()}")
            plt.figure(figsize=(10, 6))
            plt.plot(df['match_number'], df[metric], marker='o', linestyle='-', 
                    color=color[0] if color else None, label=f'{title} Trend')
            plt.xlabel('Match Number')
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend()
            plt.xticks(df['match_number'], rotation=0)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()
        else:
            logger.warning(f"No data available for {metric} to plot")

# Main function to run the script
if __name__ == "__main__":
    api_key = ""
    
    # Get player name and tag input from user
    name = input("Enter the player's name: ")
    tag = input("Enter the player's tag: ")
    
    # Fetch player's PUUID
    puuid = get_player_data(api_key, name, tag)
    
    if puuid:
        summoner_data = get_summoner_data(api_key, puuid)
        if summoner_data:
            match_ids = match_finder(api_key, puuid)
            if match_ids:
                classic_matches = []
                for match_id in match_ids:
                    match_details = match_data(api_key, match_id)
                    if match_details and match_details['info']['gameMode'] == 'CLASSIC':
                        classic_matches.append(match_id)

                print(f"Analyzing the most recent {len(classic_matches)} CLASSIC matches...")

                match_data_list = []
                for match_id in classic_matches:
                    match_info = collect_match_data(api_key, match_id, puuid)
                    if match_info:
                        match_data_list.append(match_info)

                if match_data_list:
                    save_data_to_csv(match_data_list, filename="match_data.csv")
                    visualize_data(filename="match_data.csv")
            else:
                print("No matches found.")
    else:
        print("Error fetching player information.")
