# Spyfish Aotearoa PowerBI Dashboard

This documentation guides you to set up the spyfish Aotearoa Dashboard in powerBI.

## Requirements
1. Access to relevant Amazon AWS bucket,
2. [Anaconda](https://docs.anaconda.com/anaconda/install/index.html)
3. [PowerBI Desktop](https://www.microsoft.com/en-us/power-platform/products/power-bi/downloads)

## Set up
### Connecting PowerBI to csv files in AWS
#### Install Power BI Desktop
#### Install Anaconda
#### Create an Anaconda environment for the project
Create and set up an environment for Power BI using Anaconda.
Create a new environment:
Open Anaconda Prompt and run:
```Bash
conda create --name powerbi_env python=3.9
```

Activate the new environment:
```Bash
conda activate powerbi_env
```

Install required packages:
```Bash
pip install requests pandas boto3 python-dotenv matplotlib
```

#### Safely save the AWS credentials in the Anaconda environment
Copy the `.env_sample_powerbi` file, rename it to `.env` and replace the values there.

Or create an `.env` file with your AWS credentials with the following info:
```
AWS_ACCESS_KEY_ID=<your_access_key>
AWS_SECRET_ACCESS_KEY=<your_secret_key>
S3_BUCKET=<bucket-name>
```
Change the whole string after the equal sign, including the brackets.

Save the file as `.env` in the same location as the anaconda environment (e.g. `C:\Users\<YourUsername>\Anaconda3\envs\powerbi_env\.env`).

#### Open the latest PowerBI dashboard
Download the most up-to-date "DOC Spyfish report vXXX.pbix" and run it in your computer.

#### Link PowerBI to the Python's version of the Anaconda environment
Configure PowerBI to use the right Python environment. In Power BI Desktop:

Go to File > Options and Settings> Options > Python scripting

<img src="img/screenshot_python_script_options_powerbi.png?raw=true" width="500" alt="python_scripting_options"/>

Set the Python home directory to your anaconda environment's path (e.g. `C:\Users\<YourUsername>\Anaconda3\envs\powerbi_env`)

### Read the csv files from AWS as datasets in PowerBI
In PowerBI Desktop, use the "Get Data" option and Select "More".

Select "Python script" and connect to it.

Copy and paste the script in the "powerbi_script.py" file within this directory.

IMPORTANT: Before running the script, change the first line of code to specify the path to your .env file


Click OK.


You should see the available datasets. Like in the screenshot below


<img src="img/navigator_display_datasets_lodaded.png?raw=true" width="500" alt="loaded_datasets"/>

Check the format is correct by previewing the data and "Load" them into PowerBI.

<img src="img/navigator_display_datasets_preview.png?raw=true" width="500" alt="loaded_datasets"/>



## Citation

If you use this code, please cite:
Anton V., Bouzaid C., Fonda K., Beran H., Marinovich J., Ladds M. (2025). Spyfish Aotearoa Toolkit. https://github.com/wildlifeai/Spyfish-Aotearoa-toolkit/powerbi_setup
