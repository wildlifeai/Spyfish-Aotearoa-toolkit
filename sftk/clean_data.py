import requests
import logging
from dataclasses import dataclass
from typing import Optional

@dataclass
class ScientificNameEntry:
    aphia_id: int = -1
    common_name: Optional[str] = None
    scientific_name: Optional[str] = None
    scientific_name_to_check: Optional[str] = None
    scientific_names_match: bool = False
    taxon_rank: Optional[str] = None
        

class ScientificNameProcessing:
    """Class species information from the WoRMS API."""


    def __init__(self,
            scientific_name_to_check: Optional[str] = None, 
            common_name: Optional[str] = None):
        self.common_name = common_name
        self.scientific_name_to_check = scientific_name_to_check
        self.higher_taxon = ""


    def clean_name(self):
        # Genus values are defined as Sp in the DOC data, so dealing with this here.
        if not isinstance(self.scientific_name_to_check, str):
            return None
        
        cleaned_name = self.scientific_name_to_check
        if self.scientific_name_to_check.endswith("sp"):
            cleaned_name = cleaned_name.split()[0]
            self.higher_taxon = " sp"
        return cleaned_name
        


    def query_api(self) -> ScientificNameEntry:
        """Queries the WoRMS API and returns a ScientificNameEntry populated from the response.""" 

        cleaned_name = self.clean_name()
        if not cleaned_name:
            return self.failed_to_process(f"{self.scientific_name_to_check} was not processed, not a valid string.")

        # TODO: what if there are multiple species? what does offset 1 do?
        api_url = f"https://www.marinespecies.org/rest/AphiaRecordsByName/{cleaned_name}"

        params = { 
            "like": "false",
            "marine_only": "true",
            "offset": 1
        }
        try:
            response = requests.get(api_url, params=params)
        except requests.exceptions.RequestException as e:
            return self.failed_to_process(f"{self.scientific_name_to_check} was not processed, error connecting to API: {e}")
 
        if response.status_code != 200:
            return self.failed_to_process(f"No results found for scientific name {self.scientific_name_to_check}, API request failed with status code {response.status_code}.")
        
        
        try: 
            response_json = response.json()
            if len(response_json) > 1: 
                logging.warning(f"Multiple results found for {self.scientific_name_to_check}, if that's not expected, check it here: {response_json}")
            response_json = response_json[0]
        except: 
            return self.failed_to_process(f"No results found for scientific name {self.scientific_name_to_check}, empty API response.")

        accepted_scientific_name = response_json.get('valid_name')
        if not accepted_scientific_name:
            return self.failed_to_process(f"{self.scientific_name_to_check} was not processed, no valid_name found in API response.")

        return ScientificNameEntry(
            aphia_id=response_json.get("AphiaID"),
            common_name=self.common_name,
            scientific_name=accepted_scientific_name + self.higher_taxon,
            scientific_name_to_check=cleaned_name,
            scientific_names_match=accepted_scientific_name == cleaned_name,
            taxon_rank=response_json.get("rank"),
        )
    
    def failed_to_process(self, message):
        logging.warning(message)
        return ScientificNameEntry(common_name=self.common_name, scientific_name_to_check=self.scientific_name_to_check)




