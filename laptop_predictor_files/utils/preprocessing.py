# Use a dedicated function to break the CPU data into brand, product, clock speed, and number of cores

import re
import math
from typing import Dict

# Dedicated function to parse CPU specification string into CPU brand, product, clock speed, and core count
def FeaturizeCPU(cpu_string:str) -> Dict:

    # regex search for CPU brand, product, clock speed
    brand_clock_pattern = r'(?P<CPUBrand>Intel|AMD|Apple|Samsung)\s+(?P<CPUProduct>.*?)(?P<CPUGhz>\d+\.?\d*)\s*GHz'

    # regex pattern looking for core count using traditional Greek numerical prefixes
    core_pattern = r'\b(?P<CoreLabel>Dual|Quad|Hexa|Octa|Deca|Dodeca|Hexadeca|\d{1,2})\s+Core\b'

    brand_clock_match = re.search(brand_clock_pattern,cpu_string,re.IGNORECASE)

    if brand_clock_match:

        # store parsed text data as a dictionary
        cpu_data = brand_clock_match.groupdict()

        # convert clock speed to a number in GHz (assumes all processors are GHz)
        cpu_data['CPUGhz'] = float(cpu_data["CPUGhz"]) if cpu_data["CPUGhz"] else None

        # remove trailing spaces
        cpu_data["CPUProduct"] = cpu_data["CPUProduct"].strip()

        # separate product string from other data
        core_match = re.search(core_pattern,cpu_data["CPUProduct"],re.IGNORECASE)

        # if CPU data contains a "number of cores" expression, map that expression to an integer
        if core_match:
            core_text = core_match.group("CoreLabel")
            core_map = { # overly complete to search for all possibilities, even though only 'Dual' and 'Quad' are likely
                "Dual":2,
                "Quad":4,
                "Hexa":6,
                "Octa":8,
                "Deca":10,
                "Dodeca":12,
                "Hexadeca":16
            }
        
            cpu_data["CPU Core Count"] = core_map.get(core_text.title(),None) if core_text else None

            return cpu_data
        
        # Note: most CPUs in the training dataset do not specify core counts; in these situations, leave as NaN 
        else:
            cpu_data["CPU Core Count"] = None
            return cpu_data
        
    else: 
        return {"CPUBrand":None,"CPUProduct":None,"CPUGhz":None,"CPU Core Count":None}

# ---------------------------------------------------------------------------------------------------------------    
# Dedicated function to parse storage data into primary and secondary storage (separated by "+")
# ...then identify format and total GB for each storage drive

def FeaturizeStorage(storage_spec:str) -> Dict:

    parts = storage_spec.split("+") # split storage spec into primary and secondary (secondary may be null)
    disk_spec_list = []

    # loop over primary and secondary drive data
    for part in parts:
        part = part.strip()
        pattern_size_unit = r'(?P<size>\d+\.?\d*)\s*(?P<unit>GB|TB)' # identify quantity and units of storage
        pattern_format = r'(HDD|SSD|Flash Storage|Hybrid)' # identify drive format

        size_unit_match = re.search(pattern_size_unit,part,re.IGNORECASE)

        # convert all units to GB
        if size_unit_match:
            disk_spec = size_unit_match.groupdict()
            storage_unit_dict = {"GB":1,"TB":1000} 
            multiplier = storage_unit_dict.get(disk_spec["unit"].upper(),1)
            size_gb = float(disk_spec["size"])*multiplier
        
        else:
            size_gb = 0

        # identify a format segment for all quantities of storage
        format_match = re.search(pattern_format,part,re.IGNORECASE)
        
        # if the format is not seen in the training data (all formats in training are accounted for in this function),
        # ... assign an "Other" format
        if format_match:
            format = format_match.group(1).upper()
        else:
            format = "OTHER"

        disk_spec_list.append({"size":size_gb,"format":format})

    storage_parsed = {}

    # parse the total quantity of GB into each format type, as well as total GB
    total = sum(disk["size"] for disk in disk_spec_list)
    ssd = sum(disk["size"] for disk in disk_spec_list if disk["format"] == "SSD")
    hdd = sum(disk["size"] for disk in disk_spec_list if disk["format"] == "HDD")
    flash = sum(disk["size"] for disk in disk_spec_list if disk["format"] == "FLASH STORAGE")
    hybrid = sum(disk["size"] for disk in disk_spec_list if disk["format"] == "HYBRID")
    other = sum(disk["size"] for disk in disk_spec_list if disk["format"] == "OTHER")

    num_disks = len(disk_spec_list) if total > 0 else 0 # store number of drives (primary = 1 or primary + secondary = 2)

    # return a dictionary of total GB, GB segmented by drive format, and total disks
    storage_parsed = {
        "Total GB":total,
        "SSD GB":ssd,
        "HDD GB":hdd,
        "Flash GB":flash,
        "Hybrid GB":hybrid,
        "Other GB":other,
        "Disk Count":num_disks
        }
    
    return storage_parsed

# ---------------------------------------------------------------------------------------------------------------
# Dedicated function to convert "Screen" data to boolean columns of marketing terms, as well as total pixels
def FeaturizeScreen(screen_spec:str) -> Dict:

    screen_keywords = r'\b(IPS|4K|Touchscreen|Retina)\b' # regex containing common marketing terms

    # regex search function
    found_keywords = re.findall(screen_keywords,screen_spec,re.IGNORECASE)
    found_keywords = [x.upper() for x in found_keywords]  

    # store presence or absence of each marketing term as a local variable
    has_ips = int("IPS" in found_keywords)
    has_4k = int("4K" in found_keywords)
    has_touchscreen = int("TOUCHSCREEN" in found_keywords)
    has_retina = int("RETINA" in found_keywords)

    pixel_dim_pattern = r'(?P<width>\d{3,4})\s*[xX×]\s*(?P<height>\d{3,4})' # regex searching for pixel dimensions

    # store pixel dimensions as the quantity of total diagonal pixels
    match = re.search(pixel_dim_pattern,screen_spec)
    if match:
        width = int(match.group("width"))
        height = int(match.group("height"))
        # pixel_count = width*height # replaced with size-normalized resolution
        diag_pixels = math.sqrt(width**2 + height**2)
    else:
        diag_pixels = None

    # return one-hot encoded columns for marketing terms and diagonal resolution
    return {"IPS":has_ips,"4K":has_4k,"Touchscreen":has_touchscreen,"Retina":has_retina,"Diag Resolution":diag_pixels}
