import importlib.util
import sys
import requests

# Specify the path to your .env file
env_path = r"C:\Users\USER\anaconda3\envs\powerbi_env\.env"

# Define the URL of the script
script_url = "https://raw.githubusercontent.com/wildlifeai/Spyfish-Aotearoa-toolkit/refs/heads/main/powerbi_setup/get_kso_data_from_s3.py"

# Download the script content
response = requests.get(script_url)
response.raise_for_status()  # Stop if the download fails
script_content = response.text

# Create a module in memory
spec = importlib.util.spec_from_loader("kso_to_pbi_module", loader=None, origin=script_url)
kso_to_pbi_module = importlib.util.module_from_spec(spec)

# Execute the script content within the new module's namespace
exec(script_content, kso_to_pbi_module.__dict__)
sys.modules["kso_to_pbi_module"] = kso_to_pbi_module

# Run the main function from the dynamically loaded module
processed_annotations_df, movies_df, sites_df, surveys_df, species_df = (
    kso_to_pbi_module.main(env_path)
)
