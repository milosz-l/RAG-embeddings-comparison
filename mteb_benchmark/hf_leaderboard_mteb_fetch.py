import requests
from bs4 import BeautifulSoup

# URL of the leaderboard
url = "https://huggingface.co/spaces/mteb/leaderboard"

# Send a GET request to the URL
response = requests.get(url)

# Parse the HTML content
soup = BeautifulSoup(response.content, "html.parser")

# Print the soup object to inspect the HTML
print(soup.prettify())

# Find the leaderboard table
table = soup.find("table", {"class": "svelte-1bvc1p0"})

# Extract information from each row
leaderboard_data = []
for row in table.find_all("tr"):
    cols = row.find_all("td")
    if cols:
        data = {
            "rank": cols[0].text.strip(),
            "submission": cols[1].text.strip(),
            "score": cols[2].text.strip(),
            "date": cols[3].text.strip(),
        }
        leaderboard_data.append(data)

# Now leaderboard_data contains the extracted data
