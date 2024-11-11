import requests

global data
data = []

def get_player_data(api_key, name, tag):
    base_url = "https://asia.api.riotgames.com/riot/account/v1/accounts/by-riot-id"
    encoded_name = name.replace(" ", "%20")
    url = f"{base_url}/{encoded_name}/{tag}?api_key={api_key}"

    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
    "Accept-Language": "ko,en-US;q=0.9,en;q=0.8,zh-CN;q=0.7,zh;q=0.6",
    "Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8",
    "Origin": "https://developer.riotgames.com",
    "X-Riot-Token": api_key
}

    response = requests.get(url, headers=headers)
    
    
    if response.status_code == 200:
        player_data = response.json()
        player_puuid = player_data['puuid']
        return player_puuid
        pass
    else:
        print(f"Error: Unable to fetch player data. Status code: {response.status_code}, Response: {response.text}")
        return None
def summoner():

    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Accept-Language": "ko,en-US;q=0.9,en;q=0.8,zh-CN;q=0.7,zh;q=0.6",
    "Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8",
    "Origin": "https://developer.riotgames.com"
}
    base_url = "https://jp1.api.riotgames.com/lol/summoner/v4/summoners/by-puuid"
    url = f"{base_url}/{puuid}?api_key={api_key}"

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        summoner_data = response.json()
        summoner_id = summoner_data['id']
        summoner_accountid = summoner_data['accountId']
        summoner_profileiconid = summoner_data['profileIconId']
        summoner_revisiondate = summoner_data['revisionDate']
        summoner_level = summoner_data['summonerLevel']
        pass
    else:
        print(f"Error: Unable to fetch player data. Status code: {response.status_code}, Response: {response.text}")
        return None

def match_finder():
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Accept-Language": "ko,en-US;q=0.9,en;q=0.8,zh-CN;q=0.7,zh;q=0.6",
    "Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8",
    "Origin": "https://developer.riotgames.com"
}
    base_url = "https://asia.api.riotgames.com/lol/match/v5/matches/by-puuid"
    url = f"{base_url}/{puuid}/ids?start=0&count=100&api_key={api_key}"

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        match_found_data = response.json()
        global data
        data = match_found_data
        pass
    else:
        print(f"Error: Unable to fetch player data. Status code: {response.status_code}, Response: {response.text}")
        return None

def match_data(match):
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Accept-Language": "ko,en-US;q=0.9,en;q=0.8,zh-CN;q=0.7,zh;q=0.6",
    "Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8",
    "Origin": "https://developer.riotgames.com"
}
    base_url = "https://asia.api.riotgames.com/lol/match/v5/matches"
    url = f"{base_url}/{match}?api_key={api_key}"

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        match_data = response.json()
        print(match_data)
        pass
    else:
        print(f"Error: Unable to fetch player data. Status code: {response.status_code}, Response: {response.text}")
        return None

api_key = ""


name = input("Enter the player's name: ")
tag = input("Enter the player's tag: ")

player_data = get_player_data(api_key, name, tag)

if player_data:
    puuid = get_player_data(api_key, name, tag)
    if puuid:
        summoner()
        if summoner:
            match_finder()
            for i in data:
                match_data(i)