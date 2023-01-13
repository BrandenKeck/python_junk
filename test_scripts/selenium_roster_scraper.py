# Import Selenium and select function for dropdowns
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def get_starting_eighteen(team):

    # Define positions / team id
    s18 = []
    team = team.replace(" ", "-").lower()
    paths = [
        [1, "div/", 2, 1], [1, "div/", 2, 2], [1, "div/", 2, 3],
        [1, "div/", 3, 1], [1, "div/", 3, 2], [1, "div/", 3, 3],
        [1, "div/", 4, 1], [1, "div/", 4, 2], [1, "div/", 4, 3],
        [1, "div/", 5, 1], [1, "div/", 5, 2], [1, "div/", 5, 3],
        [2, "", 2, 1], [2, "", 2, 2],
        [2, "", 3, 1], [2, "", 3, 2],
        [2, "", 4, 1], [2, "", 4, 2],
    ]

    # Scrape starters
    options = Options()
    options.headless = True
    options.add_argument("--log-level=3")
    driver = webdriver.Chrome(options=options)
    driver.get(f"https://www.dailyfaceoff.com/teams/{team}/line-combinations/")
    driver.implicitly_wait(10)
    for path in paths:
        s18.append(driver.find_element("xpath", f"//section[@id='line_combos']/div[{path[0]}]/{path[1]}div[{path[2]}]/div[{path[3]}]/div/div[2]/a/span").text)
    return s18


get_starting_eighteen("Ottawa Senators")
