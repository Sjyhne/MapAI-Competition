import yaml
import requests

url = "https://raw.githubusercontent.com/Sjyhne/MapAI-Competition/master/version.yaml"

newest_version = yaml.load(requests.get(url, allow_redirects=True).content, yaml.Loader)["version"]
current_version = yaml.load(open("version.yaml", "r"), yaml.Loader)["version"]

print("You're currently at version:", current_version)
print("Newest version available:", newest_version)

print()

if current_version == newest_version:
    print("Great, you're up to date!")
else:
    print("!--------------------------------------------------------------------------------------------------------!")
    print("!Please update to newest version. See readme in GitHub for instructions on updating to the latest version!")
    print("!--------------------------------------------------------------------------------------------------------!")
