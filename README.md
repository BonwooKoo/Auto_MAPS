# Automated Microscale Audit of Pedestrian Streetscapes (Auto-MAPS)

Auto-MAPS provides a scalable and reliable method for AUTOMATICALLY conducting a validated walkability audit called [MAPS-mini](https://drjimsallis.org/measure_maps.html#MAPSMINI) (i.e., a short version of Microscale Audit of Pedestrian Streetscapes). Auto-MAPS is based on the combination of computer vision techniques and Google Street View images. For the full description of the method, please refer to the following publication.

> Koo, B. W., Guhathakurta, S., & Botchwey, N. (2021). How are Neighborhood and Street-Level Walkability Factors Associated with Walking Behaviors? A Big Data Approach Using Street View Images. *Environment and Behavior*, https://doi.org/10.1177/00139165211014609

<br>
Microscale characteristics of streetscapes
The original MAPS-mini has 15 items, and Auto-MAPS on this repo measures 10 of them as the remaining five is measured using the conventional geographic information systems. The 10 items include the following question:

* Is a pedestrian walk signal present?
* Is there a ramp at the curb(s)?
* Is there a marked crosswalk?
* Are there any benches or places to sit?
* Are streetlights installed?
* Are the buildings well maintained?
* Is graffiti/tagging present?
* Is a sidewalk present?
* Are there poorly maintained sections of the sidewalk that constitute major trip hazards?
* Is a buffer present?

The output will be the presence (or count for streetlights) of the items for each street segment.

# Workflow
The project uses R and Python in the workflow:

## Step 1: Generating Sampling Point on Street Network
Using a street network shapefile or a state-county name as input, a CSV file containing the coordinate, heading, and unique identifying ID for Google Street View images will be generated.
