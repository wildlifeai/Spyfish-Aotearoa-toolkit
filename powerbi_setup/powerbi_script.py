# Specify the path to your .env file
env_path = "C:/path/to/your/.env"  # Change this to the correct path

import requests
import os

# Define the URL of the script
script_url = "https://raw.githubusercontent.com/wildlifeai/Spyfish-Aotearoa-toolkit/refs/heads/main/powerbi_setup/main/get_kso_data_from_s3.py"

# Define the local path to save the script
script_path = os.path.join(os.getcwd(), "get_kso_data_from_s3.py")

# Download the script
response = requests.get(script_url)
with open(script_path, "w", encoding="utf-8") as file:
    file.write(response.text)

# Load the script
exec(open(script_path).read())

# Run the main function
df = main(env_path)

# Ensure Power BI can read the DataFrame
df
