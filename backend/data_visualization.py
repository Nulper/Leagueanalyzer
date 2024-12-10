import pandas as pd
import matplotlib.pyplot as plt

def visualize_data(filename="match_data.csv"):
    try:
        df = pd.read_csv(filename, on_bad_lines='skip')
    except pd.errors.ParserError as e:
        print(f"Error reading the CSV file: {e}")
        return

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
    }

    for key, (metric, ylabel, title, *color) in metrics.items():
        if metric in df.columns and df[metric].sum() > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(df['game_creation'], df[metric], marker='o', linestyle='-', color=color[0] if color else None, label=f'{title} Trend')
            plt.xlabel('Game Creation Date')
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

def save_plot_images(filename="match_data.csv"):
    """Save plot images to a directory instead of showing them"""
    import os

    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)

    try:
        df = pd.read_csv(filename, on_bad_lines='skip')
    except pd.errors.ParserError as e:
        print(f"Error reading the CSV file: {e}")
        return

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
    }

    for key, (metric, ylabel, title, *color) in metrics.items():
        if metric in df.columns and df[metric].sum() > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(df['game_creation'], df[metric], marker='o', linestyle='-', color=color[0] if color else None, label=f'{title} Trend')
            plt.xlabel('Game Creation Date')
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'plots/{metric}_trend.png')
            plt.close()  # Close the plot to free up memory

    print("Plots have been saved to the 'plots' directory.")
