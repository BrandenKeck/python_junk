import json
import requests
from datetime import date

# Set Variables
today = date.today()
today_yr = today.year
today_mo = '{:02d}'.format(today.month)
today_dy = '{:02d}'.format(today.day)
prev_yr = "2022"
prev_mo = "10"
prev_dy = "07"
player_id = 8477404
team_id = 5
season = "20222023"

# Example Request URLs
url = f'https://statsapi.web.nhl.com/api/v1/schedule?startDate={prev_yr}-{prev_mo}-{prev_dy}&endDate={today_yr}-{today_mo}-{today_dy}'
url = f'https://statsapi.web.nhl.com/api/v1/schedule?startDate={today_yr}-{today_mo}-{today_dy}&endDate={today_yr}-{prev_mo}-{today_dy}'
url = f'https://statsapi.web.nhl.com/api/v1/game/{game_id}/boxscore'
url = f'https://statsapi.web.nhl.com/api/v1/people/{player_id}'
url = f'https://statsapi.web.nhl.com/api/v1/people/{player_id}/stats?stats=gameLog&season={season}'
url = f"https://statsapi.web.nhl.com/api/v1/schedule?teamId={team_id}"
url = f'https://statsapi.web.nhl.com/api/v1/teams'
url = f'https://statsapi.web.nhl.com/api/v1/teams/{team_id}/?expand=team.roster'

###
# APPROACH
###

# (1) Get games since last check point
url = f'https://statsapi.web.nhl.com/api/v1/schedule?startDate={prev_yr}-{prev_mo}-{prev_dy}&endDate={today_yr}-{prev_mo}-{today_dy}'
resp = requests.get(url).json()

# (2) Get boxscore for each game:
game_id = 2022020001
url = f'https://statsapi.web.nhl.com/api/v1/game/{game_id}/boxscore'
resp = requests.get(url).json()

# (3) Populate Skater Stats - Last One, Last Three, Last Five, Season

# (4) Populate Goalie Stats - Last One, Last Two, Season

# (5) Populate Team Stats - Forwards, Defensemen, Skaters, Goalies

# (6) Populate Skater Opponent Stats - Hits, Blocks, ...

# (7) Populate Goalie Opponent Stats
