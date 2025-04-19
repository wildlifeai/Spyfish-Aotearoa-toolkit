import importlib.util
import tempfile

import requests

# Specify the path to your .env file - change this to the correct path
env_path = r"C:\Users\USER\anaconda3\envs\powerbi_env\.env"

# Define the URL of the script
script_url = "https://raw.githubusercontent.com/wildlifeai/Spyfish-Aotearoa-toolkit/refs/heads/main/powerbi_setup/get_kso_data_from_s3.py"

# Download the script
response = requests.get(script_url)
response.raise_for_status()  # Raises error if download fails

# Save to a temp .py file
with tempfile.NamedTemporaryFile(suffix=".py") as tmp_file:
    tmp_file.write(response.content)
    tmp_file.flush()  # Make sure it's written before loading

    # Import the module from the temp file
    spec = importlib.util.spec_from_file_location("kso_to_pbi", tmp_file.name)
    kso_to_pbi_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kso_to_pbi_module)

# Run the main function from the imported module
processed_annotations_df, movies_df, sites_df, surveys_df, species_df = (
    kso_to_pbi_module.main(env_path)
)
