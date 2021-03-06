# Automated Microscale Audit of Pedestrian Streetscapes (Auto-MAPS)

Auto-MAPS provides a scalable and reliable method for AUTOMATICALLY conducting a validated walkability audit called [MAPS-mini](https://drjimsallis.org/measure_maps.html#MAPSMINI) (i.e., a short version of Microscale Audit of Pedestrian Streetscapes). Auto-MAPS is based on the combination of computer vision technique called [Mask R-CNN](https://github.com/matterport/Mask_RCNN) and Google Street View images. For the full description of the method, please refer to the following publication.

> Koo, B. W., Guhathakurta, S., & Botchwey, N. (2022). Development and validation of automated microscale walkability audit method. *Health & place*, 73, 102733. https://doi.org/10.1016/j.healthplace.2021.102733

Microscale characteristics of streetscapes
The original MAPS-mini has 15 items, and Auto-MAPS on this repo measures 10 of them as the remaining five is measured using the conventional geographic information systems and Pyramid Scene Parsing Network ([PSPNet](https://github.com/Vladkryvoruchko/PSPNet-Keras-tensorflow)). 

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

<br />

# Workflow
The project uses R and Python in the workflow:

## Step 1: Generating Sampling Point on Street Network (R)
Step 1 uses **prepareDownload.R** in Step1 folder and is based on R language. Using a street network shapefile or a state-county name as input, a CSV file containing the coordinate, heading, and unique identifying ID for Google Street View images will be generated. 

*In a future update, Step 1 will have an option to use Python instead of R.*

Once you've cloned this repo, import the functions in the script into your R session and execute prepare_download_points() function. As arguments to the function, you can either (1) provide your own shapefile or (2) state-county name pair. If using your own street network shapefile, Topologically Integrated Geographic Encoding and Referencing (TIGER) is recommended. If you provide state-county name pair as input, the R script will automatically download the TIGER shapefile. Additionally, you will need to provide your own Google Maps API key as an argument. In Step 1, the API key is used for collecting metadata of GSV images, which is free of charge as of 8/29/2021. 

See below for an example code:
```
# R 4.0.2.
source("path-to-prepareDownload.R-file")
point_gsv <- prepare_download_points(input_shape = your_TIGER_shapefile, key = "your-google-api-key") # or state = "GA", county = "Fulton"
point_gsv$audit_point %>% 
  st_set_geometry(NULL) %>% 
  write.csv(., "path-to-your-output-file")
```

**Dependency**
* tidyverse 1.3.0
* sf 0.9-5
* tigris 1.0

<br />

## Step 2: Downloading Street View Images & Applying Computer Vision (Python)

> ***NOTE: Step 2 and 3 assumes that you are using Google Colab for demonstration purposes.***

After Step 1, you will have saved a CSV file. You will need to upload the CSV file to Google Drive so that Google Colab can access the file. After the CSV is uploaded, you can finish the remainder of the process by following steps in [Auto_MAPS_demo.ipynb](https://github.com/BonwooKoo/automated_audit/blob/main/sample/demo.ipynb).

**Dependency**
* Packages needed in [Mask R-CNN](https://github.com/matterport/Mask_RCNN)
* pandas
* urllib
* Numpy
* Pillow
* matplotlib
* h5py
* scipy
* scikit-image

For questions, please contact Bon Woo Koo at bkoo34@gatech.edu.
