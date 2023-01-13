import requests

sport="icehockey_nhl"
api_key = "INSERT_KEY"
regions = "us"
markets = "h2h,spreads,totals"
bookmaker = "fanduel"
format = "american"
url = f'https://api.the-odds-api.com/v4/sports/{sport}/odds/?apiKey={api_key}&regions={regions}&markets={markets}&bookmakers={bookmaker}&oddsFormat={format}'
resp = requests.get(url).json()
