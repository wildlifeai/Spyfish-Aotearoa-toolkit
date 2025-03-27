import requests

# Define the URL of the script
script_url = "https://raw.githubusercontent.com/wildlifeai/Spyfish-Aotearoa-toolkit/refs/heads/powerbiscript/powerbi_setup/get_kso_data_from_s3.py"

# Fetch the script content
response = requests.get(script_url)

# Execute the script in the current namespace
exec(response.text, globals())

# Specify the path to your .env file
env_path = (
    r"C:\Users\USER\anaconda3\envs\powerbi_env\.env"  # Change this to the correct path
)

# Run the main function from the script
df = main(env_path)
