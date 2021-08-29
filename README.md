# Automated Microscale Audit of Pedestrian Streetscapes (Auto-MAPS)

Auto-MAPS provides a scalable and reliable method for AUTOMATICALLY conducting a validated walkability audit called [MAPS-mini](https://drjimsallis.org/measure_maps.html#MAPSMINI) (i.e., a short version of Microscale Audit of Pedestrian Streetscapes). Auto-MAPS is based on the combination of computer vision technique called [Mask R-CNN](https://github.com/matterport/Mask_RCNN) and Google Street View images. For the full description of the method, please refer to the following publication.

> Koo, B. W., Guhathakurta, S., & Botchwey, N. (2021). How are Neighborhood and Street-Level Walkability Factors Associated with Walking Behaviors? A Big Data Approach Using Street View Images. *Environment and Behavior*, https://doi.org/10.1177/00139165211014609

<br>
Microscale characteristics of streetscapes
The original MAPS-mini has 15 items, and Auto-MAPS on this repo measures 10 of them as the remaining five is measured using the conventional geographic information systems and Pyramid Scene Parsing Network ([PSPNET](https://github.com/Vladkryvoruchko/PSPNet-Keras-tensorflow)). 

| Section | Question                             | Method    |
| :------ | :----------------------------------- | :-------: |
| Crossing| Is a pedestrian walk signal present? | Auto-MAPS |
|         | Is there a ramp at the curb(s)?      | Auto-MAPS |
|         | Is there a marked crosswalk?         | Auto-MAPS |
| Segment | Type of land use?                    | GIS       |
|         | How many public parks are present?   | GIS       |
|         | How many public transit stops are present? | GIS |
|         | Is there a designated bike path      | GIS       |
|         | Are there any benches or places to sit? | Auto-MAPS |
|         | Are streetlights installed?          | Auto-MAPS |
|         | Are the buildings well maintained?   | Auto-MAPS |
|         | Is graffiti/tagging present?         | Auto-MAPS |
|         | Is a sidewalk present?               | Auto-MAPS |
|         | Are there poorly maintained sections of the sidewalk that constitute major trip hazards? | Auto-MAPS |
|         | Is a buffer present?                 | Auto-MAPS |
|         | What percentage of the length of the sidewalk/walkway is covered by trees, awnings, or other overhead coverage? | PSPNet |

The output will be the presence or the count of the items for each street segment.

# Workflow
The project uses R and Python in the workflow:

## Step 1: Generating Sampling Point on Street Network (R)
Step 1 uses the script in **generate_sample_point** folder and is based on R language. Using a street network shapefile or a state-county name as input, a CSV file containing the coordinate, heading, and unique identifying ID for Google Street View images will be generated. 

Once you've cloned this repo, import the functions in the script into your R session and execute prepare_download_points() function. As arguments to the function, you can either provide your own shapefile or state-county name pair. If using your own street network shapefile, Topologically Integrated Geographic Encoding and Referencing (TIGER) is recommended. If you provide state-county name as input, the R script will automatically download the TIGER shapefile. *In the next update, Step 1 will have an option for using Python instead of R.*

See below for an example code:
```
source("path-to-your-cloned-file")
point_gsv <- prepare_download_points(input_shape = your_TIGER_shapefile) # or state = "GA", county = "Fulton"
point_gsv['audit_point'] %>% st_set_geometry(NULL) %>% write.csv(., "path-to-your-output-file")
```

**Dependency**
* tidyverse - version 1.3.0
* sf - version version 0.9-5
* tigris - version 1.0

## Step 2: Downloading Street View Images (Python)
Step 2 uses the script in **auto_audit** folder and is based on Python language. After Step 1, you will have saved a CSV file. This CSV file is used as an input to Step 2 and 3. 

See below for an example code:
```
AUDIT_POINT_PATH = "path-to-your-output-from-Step1"
DOWNLOAD_PATH = "path-to-your-image-folder"

audit_point = pd.read_csv(AUDIT_POINT_PATH)
auto_audit = auto_audit_df()
auto_audit.add_image_info(audit_point)
test_df.download_gsv(download_path = DOWNLOAD_PATH)
```

## Step 3: Applying Computer Vision and Calculate Statistics (Python)
