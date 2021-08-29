# Automated Microscale Audit of Pedestrian Streetscapes (Auto-MAPS)

Auto-MAPS provides a scalable and reliable method for AUTOMATICALLY conducting a validated walkability audit called MAPS-mini (i.e., a short version of Microscale Audit of Pedestrian Streetscapes). Auto-MAPS is based on the combination of computer vision techniques and Google Street View images. For the full description of the method, please refer to the following publication.

Koo, B. W., Guhathakurta, S., & Botchwey, N. (2021). How are Neighborhood and Street-Level Walkability Factors Associated with Walking Behaviors? A Big Data Approach Using Street View Images. *Environment and Behavior*, https://doi.org/10.1177/00139165211014609

The original MAPS-mini has 15 items, and Auto-MAPS on this repo measures 11 of them. 


# Workflow
The project uses R and Python in the workflow:

## Step 1: Generating Sampling Point on Street Network
Using a street network shapefile or a state-county name as input, a CSV file containing the coordinate, heading, and unique identifying ID for Google Street View images will be generated.
