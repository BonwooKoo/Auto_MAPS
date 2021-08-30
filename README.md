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
Step 1 uses **prepareDownload.R** in Step1 folder and is based on R language. Using a street network shapefile or a state-county name as input, a CSV file containing the coordinate, heading, and unique identifying ID for Google Street View images will be generated. *In a future update, Step 1 will have an option to use Python instead of R.*

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
* tidyverse - version 1.3.0
* sf - version version 0.9-5
* tigris - version 1.0

<br>

## Step 2: Downloading Street View Images (Python)

> ***NOTE: If you do not have required packages (e.g., Tensorflow) installed, using Google Colab or other similar services is recommended. See [this](https://colab.research.google.com/drive/1_yiTDSLqwJfdvHRXvVWHrsv-_LLRlXFh?usp=sharing) Google Colab notebook for an example.***

Step 2 and 3 uses **auto_audit.py** in Step2_3 folder and is based on Python language. After Step 1, you will have saved a CSV file. This CSV file is used as an input to Step 2 and 3. If you are using Google Colab or other similar services, upload the CSV file to your storage. 

After creating an instance of auto_audit_df class, use .add_image_info method() to add image information and .download_gsv() method to download GSV images. You will need to provide your API key again, and getting the images are NOT FREE OF CHARGE.

See below for an example code:
```
# Python 3.6.10.
AUDIT_POINT_PATH = "path-to-your-CSVfile-from-Step1"
DOWNLOAD_PATH = "path-to-folder-to-which-you-will-download-images"

audit_point = pd.read_csv(AUDIT_POINT_PATH)
auto_audit = auto_audit_df()
auto_audit.add_image_info(audit_point)
test_df.download_gsv(download_path = DOWNLOAD_PATH, key = "your-google-api-key")
```

<br>

## Step 3: Applying Computer Vision and Calculate Statistics (Python)
Once you have downloaded the images, use .predict() method to apply the computer vision technique to the downloaded images. Next, use .prediction_summary() method to summarise the prediction results for each street segment. 

See below for an example code:
```
# Python 3.6.10.
test_df.predict(DOWNLOAD_PATH, model_dense, model_intersection, model_top) # I need to change this part so that model_dense etc. are part of the class object.
output = test_df.prediction_summary()
output.to_csv(DOWNLOAD_PATH + "/output.csv")

# Additionally
test_df.show_prediction(DOWNLOAD_PATH, SegID = your-SegID)
```

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
