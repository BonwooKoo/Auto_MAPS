import os
from  urllib import request
import pandas as pd
import numpy as np
import sys
import joblib
import time
import matplotlib.pyplot as plt
import random
import math
import pickle
import skimage

sys.path.append("/content/Mask_RCNN")
from mrcnn import visualize

class audit_dataset:
    # Initiate
    def __init__(self):
        self.image_ids = [] # a sequential id given just for this class
        self.image_info = {} # SegID, type, LR, sequence_id, pano_id, azimuth, prediction, etc.

    # Method: Add image_info
    def add_image_info(self, pandas_df):
        """
        Purpose : Add image_info dictionary prepared in 'prepare_image_info' to 'self.image_info'
        Argument: Pandas_df, which is 'audit_point'
        Return  : Does not return anything. performs append instead.
        """
        # unique_SegID as the first iterator
        unique_SegID = pandas_df.SegID.unique()
        # unique id which will be linked with each images for easy navigation
        n = 0

        # SegID
        for segid in unique_SegID:
            # Extract pandas_df for segid
            df_by_SegID = pandas_df[pandas_df.SegID.eq(segid)]

            # Append
            self.image_info[segid] = {"image_info": {}, "pred_summary": {}}

            # Dense vs. Intersection
            for ptype in ["dense", "intersection"]:
                # Further extract pandas_df for ptype
                df_by_type = df_by_SegID[df_by_SegID.type.eq(ptype)]

                # Append
                self.image_info[segid]['image_info'][ptype] = {}


                # IF INTERSECTION
                if ptype == "intersection":
                    df_by_LR = df_by_type[df_by_type.LR.eq('I')]
                    self.image_info[segid]['image_info'][ptype]["I"] = {}

                    # Extract unique sequence_id
                    sequence_ids = df_by_LR.sequence_id.unique()

                    # sequence_id
                    for seq_id in sequence_ids:
                        df_by_seq_id = df_by_LR[df_by_LR.sequence_id.eq(seq_id)] # this df_by_seq_id his a two-row data.frame
                        row1 = df_by_seq_id.iloc[0,:]
                        row2 = df_by_seq_id.iloc[1,:]

                        # Append (+45 -45 heading is done here)
                        self.image_info[segid]['image_info'][ptype]["I"][seq_id] = {
                            'filename': [
                                # row 1
                                "micro" + "_SegID-" + str(segid) + "_type-"+ ptype + "_side-I_seqid-" + str(row1.sequence_id) + "_panoid-" + row1.pano_id + "_azi-" + str(round(row1.azimuth, 2)) +".jpg",
                                # row 2
                                "micro" + "_SegID-" + str(segid) + "_type-"+ ptype + "_side-I_seqid-" + str(row2.sequence_id) + "_panoid-" + row2.pano_id + "_azi-" + str(round(row2.azimuth, 2)) +".jpg"],
                            'pano_id': row1.pano_id,
                            'location': str(row1.y_coord) + "," + str(row1.x_coord),
                            'azimuth': row1.azimuth + 45,
                            'length': row1.length,
                            'short_cut_id': n,
                            'prediction': None,
                            'crop_percent': None,
                            'rotate':None
                        }

                        # Append short-cuts
                        short_cut = [segid, ptype, "I", seq_id]
                        self.image_ids.append(short_cut)
                        n += 1

                # IF DENSE
                elif ptype == 'dense':
                    # Left vs. Right
                    for side in ['L', 'R', 'T']:
                        # Further extract pandas_df for side
                        df_by_LR = df_by_type[df_by_type.LR.eq(side)]

                        # Append
                        self.image_info[segid]['image_info'][ptype][side] = {}

                        # Extract unique sequence_id
                        sequence_ids = df_by_LR.sequence_id.unique()
                        # Flip if on the right-side
                        if side == "R":
                            sequence_ids[::-1].sort()

                        # sequence_id
                        for seq_id in sequence_ids:
                            df_by_seq_id = df_by_LR[df_by_LR.sequence_id.eq(seq_id)]

                            # Append
                            self.image_info[segid]['image_info'][ptype][side][seq_id] = {
                                'filename': "micro" + "_SegID-" + str(segid) + "_type-"+ ptype + "_side-" + df_by_seq_id.LR.iat[0] + "_seqid-" + str(df_by_seq_id.sequence_id.iat[0]) + "_panoid-" + df_by_seq_id.pano_id.iat[0] + "_azi-" + str(round(df_by_seq_id.azimuth.iat[0],2)) +".jpg",
                                'pano_id': df_by_seq_id.pano_id.iat[0],
                                'location': str(df_by_seq_id.y_coord.iat[0]) + "," + str(df_by_seq_id.x_coord.iat[0]),
                                'azimuth': df_by_seq_id.azimuth.iat[0],
                                'length': df_by_seq_id.length.iat[0],
                                'short_cut_id': n,
                                'prediction': None,
                                'crop_percent': None,
                                'stitched': None
                            }

                            # Append short-cuts
                            short_cut = [segid, ptype, side, seq_id]
                            self.image_ids.append(short_cut)
                            n += 1

    # METHOD: download GSV images
    def download_gsv(row, img_path):
        # Downloads gsv images
        # argument: row - a pandas series for one GSV image
        # return:   img - a jpg image

        # Parameters
        base_url = "https://maps.googleapis.com/maps/api/streetview?size=640x640&"
        location = "location={},{}&".format(row.y_coord, row.x_coord)
        pitch = "pitch=0&"
        key = "key=" + os.getenv('google_api')

        # Different heading for dense vs. intersection
        if row.type == "dense":
            heading = "heading={}&".format(row.azimuth)
            full_url = base_url + location + heading + pitch + key

            # File name
            # file_name = "micro_" + row.type + "_panoid" + row.pano_id + "_SegID" + str(row.SegID) + "_seq" + str(row.sequence_id) + "_LR-" + row.LR + "_azi" + str(row.azimuth) +".jpg"
            file_name = "micro" + "_SegID-" + str(row.SegID) + "_type-"+ row.type + "_side-" + row.LR + "_seqid-" + str(row.sequence_id) + "_panoid-" + row.pano_id + "_azi-" + str(round(row.azimuth,2)) +".jpg"
            full_path = os.path.join(img_path, file_name)

            # Download if not done already
            if os.path.exists(full_path) is False:
                request.urlretrieve(full_url, os.path.join(img_path, file_name))

        else: # when the point is intersection
            heading_1 = "heading={}&".format(row.azimuth - 45)
            heading_2 = "heading={}&".format(row.azimuth + 45)
            full_url_1 = base_url + location + heading_1 + pitch + key
            full_url_2 = base_url + location + heading_2 + pitch + key

            # File name
            file_name_1 = "micro" + "_SegID-" + str(row.SegID) + "_type-"+ row.type + "_side-" + row.LR + "_seqid-" + str(row.sequence_id) + "_panoid-" + row.pano_id + "_azi-" + str(round(row.azimuth - 45,2)) +".jpg"
            file_name_2 = "micro" + "_SegID-" + str(row.SegID) + "_type-"+ row.type + "_side-" + row.LR + "_seqid-" + str(row.sequence_id) + "_panoid-" + row.pano_id + "_azi-" + str(round(row.azimuth + 45,2)) +".jpg"
            full_path_1 = os.path.join(img_path, file_name_1)
            full_path_2 = os.path.join(img_path, file_name_2)

            # Download if not done already
            if os.path.exists(full_path_1) is False:
                request.urlretrieve(full_url_1, os.path.join(img_path, file_name_1))

            if os.path.exists(full_path_2) is False:
                request.urlretrieve(full_url_2, os.path.join(img_path, file_name_2))

            pass


    # METHOD: predict
    def predict(self, image_path, model_dense, model_intersection, model_top, imshow = True):
        """
        Purpose : Predict, get overlap, and crop using Mask R-CNN for all images in the given dataset object.
                  Also download additional images if gaps exist.
        Argument: Dataset object of class 'dataset'.
        Return  : Append the prediction.
        """

        # Sequentially extract each image
        for segid in self.image_info.keys():
            for ptype in self.image_info[segid]['image_info'].keys():
                for side in self.image_info[segid]['image_info'][ptype].keys():
                    for seqid in self.image_info[segid]['image_info'][ptype][side].keys():

                        # INTERSECTIONS
                        if side is 'I':
                            file_name_1 = self.image_info[segid]['image_info'][ptype][side][seqid]['filename'][0]
                            file_name_2 = self.image_info[segid]['image_info'][ptype][side][seqid]['filename'][1]

                            full_path_1 = os.path.join(image_path, file_name_1)
                            full_path_2 = os.path.join(image_path, file_name_2)

                            # print("[INFO] Now processing -> {}..".format(file_name_1))

                            # Make prediction
                            image_1 = skimage.io.imread(full_path_1)
                            image_2 = skimage.io.imread(full_path_2)

                            yhat_1 = model_intersection.detect([image_1], verbose = 0)
                            yhat_2 = model_intersection.detect([image_2], verbose = 0)

                            # Append
                            self.image_info[segid]['image_info'][ptype][side][seqid]['prediction'] = {"intersection":[[file_name_1, yhat_1],[file_name_2, yhat_2]]}

                        # DENSE - Left or Right
                        elif side in ["L", "R"]:
                            file_name = self.image_info[segid]['image_info'][ptype][side][seqid]['filename']
                            full_path = os.path.join(image_path, file_name)

                            # print("[INFO] Now processing -> {}..".format(file_name))

                            # Make prediction
                            image = skimage.io.imread(full_path)
                            yhat = model_dense.detect([image], verbose = 0)

                            # Append
                            self.image_info[segid]['image_info'][ptype][side][seqid]['prediction'] = {"dense":[file_name, yhat, None]}

                        # DENSE - TOP
                        elif side is "T":
                            file_name = self.image_info[segid]['image_info'][ptype][side][seqid]['filename']
                            full_path = os.path.join(image_path, file_name)

                            # print("[INFO] Now processing -> {}..".format(file_name))

                            # Make prediction
                            image = skimage.io.imread(full_path)
                            yhat = model_top.detect([image], verbose = 0)

                            # Append
                            self.image_info[segid]['image_info'][ptype][side][seqid]['prediction'] = {"top":[file_name, yhat, None]}

    # METHOD:
    def save_dataset(self, path):
        with open(path, 'wb') as f:
            joblib.dump(self, f)

    # METHOD: Parse image_info with integer
    def get_image_info_by_id(self, id):
        """
        Purpose : Returns image_info & prediction based on the image_ids
        Argument: id - an integer representing short_cut_id
        Return  : Full image_info for the given ids
        """
        short_cut = self.image_ids[id]
        parsed_info = self.image_info[short_cut[0]]['image_info'][short_cut[1]][short_cut[2]][short_cut[3]]
        return parsed_info
    # METHOD:
    def get_image_info_by_SegID(self, SegID):
        """
        Purpose : Returns image_info & prediction based on the SegID
        Argument: SegID - an integer representing SegID
        Return  : Full image_info for the given ids
        """
        image_info = self.image_info[SegID]
        return image_info

    # METHOD: Visualize the image prediction for a given integer
    def show_prediction(self, image_path, SegID = None, image_id = None, show_overlap = True):
        """
        Purpose : Visualize the original image as well as the prediction masks.
        Argument: id - an integer representing short_cut_id.
        Return  : Visualizd images
        """
        def get_ax(rows=1, cols=1, size=16):
            """Return a Matplotlib Axes array to be used in
            all visualizations in the notebook. Provide a
            central point to control graph sizes.

            Adjust the size attribute to control how big to render images

            I NEED TO CHECK WHERE I GOT THIS CODE FROM

            """
            _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
            return ax

        def show_one_image(prediction):
            """
            Purpose : an insider function that visualizes different number of image(s) depending whether it is dense or intersection
            Argument: prediction - {"dense":[file_name, yhat]} or {"intersection":[[file_name_1, yhat_1],[file_name_2, yhat_2]]}
            Return  : plt.imshow(image) + mrcnn.visualize.
            """
            # If Intersection
            if "intersection" in prediction.keys():
                print("Dealing with intersection points")

                for fname_yhat in prediction['intersection']:
                    full_path = os.path.join(image_path, fname_yhat[0])
                    image = skimage.io.imread(full_path)
                    # Get prediction result
                    yhat = fname_yhat[1][0]

                    # Plot
#                     plt.figure(figsize = (16,16))
#                     plt.imshow(image)

#                     pyplot.figure(figsize=(16, 16))
#                     pyplot.imshow(image)

                    ax = plt.gca()
                    r = yhat
                    ax = get_ax(1)
                    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], list(self.class_code_intersection.values()), r['scores'], ax = ax, title = fname_yhat[0], show_bbox = False)

            # If Dense
            elif "dense" in prediction.keys():
                file_name = prediction['dense'][0]
                full_path = os.path.join(image_path, file_name)
                print("Dealing with dense points")
                image = skimage.io.imread(full_path)
                # Get prediction result
                yhat = prediction['dense'][1][0]

                # Plot
                ax = plt.gca()
                plt.figure(figsize = (16,16))
                plt.imshow(image)

                ax = get_ax(1)
                r = yhat
                visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], list(self.class_code_dense.values()), r['scores'], ax = ax, title = prediction['dense'][0], show_bbox = False)

            elif "top" in prediction.keys():
                file_name = prediction['top'][0]
                full_path = os.path.join(image_path, file_name)
                print("Dealing with top points")
                image = skimage.io.imread(full_path)
                # Get prediction result
                yhat = prediction['top'][1][0]

                # Plot
                ax = plt.gca()
                plt.figure(figsize = (16,16))
                plt.imshow(image)

                ax = get_ax(1)
                r = yhat
                visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], list(self.class_code_top.values()), r['scores'], ax = ax, title = prediction['top'][0], show_bbox = False)
            pass

        if SegID is None and image_id is None:
            print("Argument(s) for 'SegID' or 'image_id' is needed.")

        elif SegID is None and image_id is None:
            print("Only either one of 'SegID' or 'image_id' should be provided.")

        # When SegID is provided
        elif SegID is not None and image_id is None:
            ids_for_SegID = [x for x in self.image_ids if SegID in x]

            for image_id in ids_for_SegID:
                prediction_for_id = self.image_info[image_id[0]]['image_info'][image_id[1]][image_id[2]][image_id[3]]['prediction']
                show_one_image(prediction_for_id)

        # When iamge_id is provided
        elif SegID is None and image_id is not None:
            prediction_for_id = self.get_image_info_by_id(image_id)
            show_one_image(prediction_for_id)
    pass


    # Method: Calculate overlapping region
    def handle_overlap_gap(self, image_path, imshow = True):
        """
        Purpose : Calculate overlapping region in % of the left-most image in a sequence
        Argument:
        return  : Append the % overlap, cut the image, copy the cut image to another folder.
        """

        # REQUIRED FUNCTIONS ----------------------------------------

        # ***********************************************************
        # CHECK WHETHER THERE ARE ROAD MASKS OVERLAPPED WITH SIDEWALK
        # BC SUCH ROAD MASKS NEEDS TO BE DELETED FROM CONSIDERATION
        # ***********************************************************

        def find_n_given_yhat(yhat):
            # Get the index of objects representing road & sidewalk
            road_index = np.where(yhat['class_ids'] == 2)
            sidewalk_index = np.where(yhat['class_ids'] == 1)
            driveway_index = np.where(yhat['class_ids'] == 20)

            # Extract driveway masks
            if len(driveway_index[0]) > 0 :
                driveway_mask = yhat['masks'][:,:,driveway_index[0]]
#                 print("Driveways(s) detected.")
                for i in range(driveway_mask.shape[2]):
                    if i == 0:
                        continue
                    else:
                        driveway_mask[:,:,0] = driveway_mask[:,:,0] + driveway_mask[:,:,i]

                driveway_mask = driveway_mask[:,:,0]

            # Extract sidewalk masks
            if len(sidewalk_index[0]) > 0:
                sidewalk_mask = yhat['masks'][:,:,sidewalk_index[0]]
#                 print("Sidewalk(s) detected.")

                # collapse all sidewalk masks into one mask
                for i in range(sidewalk_mask.shape[2]):
                    if i == 0:
                        continue
                    else:
                        sidewalk_mask[:,:,0] = sidewalk_mask[:,:,0] + sidewalk_mask[:,:,i]

                sidewalk_mask = sidewalk_mask[:,:,0]

            # Merge driveway_index and sidewalk_index
            if len(driveway_index[0]) > 0 and len(sidewalk_index[0]) > 0:
                sidewalk_mask = sidewalk_mask + driveway_mask
            if len(driveway_index[0]) > 0 and len(sidewalk_index[0]) == 0:
                sidewalk_mask = driveway_mask

            # extract road masks
            if len(road_index[0]) == 0:
#                 print("There is no road in the prediction.")
                return None

            else:
                road_mask = yhat['masks'][:,:,road_index[0]]

                # extract sidewalk masks
                if len(sidewalk_index[0]) > 0 or len(driveway_index[0] > 0):

                    # determine road masks that are overlapped with sidewalk and drop it
                    overlapping_road_mask_index = []
                    for i in range(len(road_index[0])):
                        # total area of ith road mask
                        road_area = np.sum(road_mask[:,:,i])
                        # multiply ith road mask with sidewalk mask ==> only overlapping region will have 1
                        overlap_area = np.sum(road_mask[:,:,i]*sidewalk_mask) / np.sum(road_mask[:,:,i])
                        # note the index if overlap_area is greater than 0.3
                        if overlap_area >= 0.5:
#                             print("Road and sidewalk are overlapped.")
                            overlapping_road_mask_index.append(i)

                    # # delete the road_mask that are overlapped with sidewalk_mask
                    # # EDIT: Turning this function off because when sidewalk is masked on what is actually road,
                    #  #      that can significantly throw off the result.

                    # road_mask = np.delete(road_mask[:][:], overlapping_road_mask_index, axis = 2)

                # Now that road masks overlapped with sidewalks are deleted,
                # Merge multiple road masks into one large mask
                if road_mask.shape[2] > 0:

                    for i in range(road_mask.shape[2]):
                        # skip the first masks because all other masks are added to the first one
                        if i == 0:
                            continue
                        # Add all masks except the first one to the first one, and extract it
                        else:
                            road_mask[:,:,0] = road_mask[:,:,0] + road_mask[:,:,i]

                    road_mask = road_mask[:,:,0]

#                 # find road masks that overlaps with sidewalk for 50% of its area
#                 plt.imshow(road_mask.astype(np.uint8))
#                 plt.show()

                if np.sum(road_mask) == 0:
#                     print("All road masks are deleted due to the overlap.")
                    return None

                else:
                    # *********************************************
                    # CONVOLVE WITH [-1, 1] FILTER TO IDENTIFY n,
                    # THE PIXEL AT WHICH ROAD MASK ENDS (AND BEGINS BUFFER, SIDEWALK, ETC.)
                    # *********************************************

                    # Randomly select 10 columns from the middle-columns to see, on average,
                    # at which pixel from bottom up does the road mask ends.

                    # Min and max column pixel of road
                    max_columns = road_mask.shape[1]
                    select_few_columns = False

                    if select_few_columns:
                        # Randomly select 10 columns
                        col_num = 10
                        column_middle = max_columns/2
                        random_columns = road_mask[:,[random.randint(column_middle-20,column_middle+20) for x in range(col_num)]] # or [random.randint(0,640) for x in range(col_num)] in the column position
                    else:
                        # Instead of randomly selecting 10 columns, I now look at all columns and find the maximum value from that.
                        col_num = max_columns
                        random_columns = road_mask # or [random.randint(0,640) for x in range(col_num)] in the column position

                    # Pad one 0 on top row of the entire 640 x 640 mask space
                    random_columns_pad = np.zeros([641, col_num])
                    for col in range(col_num):
                        random_columns_pad[:,col] = np.insert(random_columns[:,col],0,0)

                    # Convolve using shape [2,1] filter
                    kernel = np.array([-1,1])
                    convolved = np.zeros([640,col_num])
                    for col in range(col_num):
                        for row in range(640):
                            convolved[row,col] = np.sum(np.multiply(random_columns_pad[row:row+2,col], kernel))

#                     # find road masks that overlaps with sidewalk for 50% of its area
#                     plt.imshow(convolved.astype(np.uint8))
#                     plt.show()

                    road_location_per_col = []
                    for col in range(col_num):
                        if len(np.where(convolved[:,col] == 1)[0]) > 0:
                            road_location_per_col.append(max(640 - np.where(convolved[:,col] == 1)[0]))
                        else:
                            continue

                    if select_few_columns:
                        n = np.average(road_location_per_col)
                    else:
                        n = np.max(road_location_per_col)
                    # print("Road mask found. Returning the estimated {} pixels for n".format(n))
                    return n

        def find_d_given_n(n, camera_height = 2.499):
            if n is not None:
                d = (320/(320 - n))*camera_height
#                 print("With n of {}, the estimated d is {}".format(n, d))
                return d
            else:
                return 2 # if n is None, there was no road mask in prediction. So returning 2 meters, the first lane default value.

        def find_I_given_coordinates(c1, c2):
            R = 6373.0

            lat1 = float(c1.split(',')[0])
            lon1 = float(c1.split(',')[1])
            lat2 = float(c2.split(',')[0])
            lon2 = float(c2.split(',')[1])

            lat1 = radians(lat1)
            lon1 = radians(lon1)
            lat2 = radians(lat2)
            lon2 = radians(lon2)

            dlon = lon2 - lon1
            dlat = lat2 - lat1

            a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))

            I = R * c*1000

            return I

        def is_looking_inside(azimuth1, azimuth2):
            """
            Purpose : Given the two azimuth, detemines whether the two azimuth are looking
            inside/outside of the curvature of the road.
            Argument: azimuth1 (left image), azimuth2 - heading of street view images raning bewteen 0 ~ 360
            Return  : Boolean - True if looking inside; False else
            """
            if azimuth1 < 180:
                if azimuth2 >= azimuth1 and azimuth2 <= azimuth1 + 180:
                    return(False)
                else:
                    return(True)

            elif azimuth1 >= 180:
                if (azimuth2 > azimuth1 and azimuth2 <= 360) or (azimuth2 >= 0 and azimuth2 < azimuth1 - 180):
                    return(False)
                else:
                    return(True)

        def find_overlap_gap_given_d(c1, c2, azimuth1, azimuth2, d, d_margin = 0):
            """
            Argument:
            c1 and c2 = The coordinates of camera 1 and 2, in the form of "33.776965,-84.395177".
            azimuth1 and azimuth2 = The azimuth of camera 1 and 2, with the north = 0 = 360.
            d: The estimated distance in meters between the camera and the end of the road in GSV image.
            d_margin: The distance I'd like to add to d. This is to take into consideration the tendency that
                      street lights, seating, etc. are often not immediately adjacent to the edge of the road but
                 are found a few meters from the edge of the road.

            Return:
            x: The estimated distance in meters that are overlapped portion in c1 image.
            W: The estimated width of the image screen on which images of objects on the streets are projected.
            """

            # adjust d
            d = d + d_margin

            # get I and theta
            I = find_I_given_coordinates(c1, c2)
            theta = np.abs(azimuth1 - azimuth2)*(pi/180)
            if theta > pi:
                 theta = 2*pi - theta

            # calculate overlapping x
            degree45 = pi / 4
            x1 = ( d*sqrt(2)*cos(theta) - I*sin(degree45 + (theta/2)) ) / ( sin(degree45 + theta) )
            x2 = (( d*sqrt(2)*cos(theta)*cos(theta) ) / cos(degree45 + theta) ) - I*sqrt(2)*sin(degree45 - (theta/2))
            W = d*2
#             print("d: {} meters with d_margin: {} meters, Theta: {} degrees, I: {} meters, W: {} meters".format(d - d_margin, d_margin, theta, I, W))
#             print("azimuth1: {}, azimuth2: {}".format(azimuth1, azimuth2))

            # Looking inside?
            # If images are on a straight road
            if round(azimuth1, 1) == round(azimuth2, 1):
                x2 = W - I
#                 print("Images are parallel: ", round((x2/W)*100, 3), "% needs to be cropped")
                return x2, W

            # If images are on a curvy road
            else:
                look_inside = is_looking_inside(azimuth1, azimuth2)

                if look_inside:
#                     print("Using inner formula:", round((x1/W)*100, 3), "% needs to be cropped")
                    return x1, W
                else:
#                     print("Using outer formula:", round((x2/W)*100, 3), "% needs to be cropped")
                    return x2, W

        # Find rotate degree
        def find_rotate_angle(x, W):
            """
            Purpose : Calculate the degree to which the heading needs to be rotated to cover the gap.
            Argument:
              percent_gap: The gap representated in x/W format.
            Return: The amount of rotation needed to fill the gap, in degrees.
            """
            theta = math.degrees(math.atan(abs(x)/(W + abs(x))))
            return(theta)

        def url_to_image(url):
            # download the image, convert it to a NumPy array, and then read
            # it into OpenCV format
            resp = urlopen(url)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            # return the image
            return image

        # *****************************************
        # Apply the required functions sequentially
        # *****************************************

        error_images = {}
        avg_gap_overlap = {}
        existing_imgs = os.listdir(image_path)

        # Sequentially extract each image
        for segid in self.image_info.keys():
            avg_gap_overlap[segid] = []

            for ptype in self.image_info[segid]['image_info'].keys():
                for side in self.image_info[segid]['image_info'][ptype].keys():
                    for index, seqid in enumerate(self.image_info[segid]['image_info'][ptype][side].keys()):
#                         print(" ------------------ ")
#                         print("SegID: {}, ptype: {}, side: {}, seqid: {}".format(segid, ptype, side, seqid))
#                         print("short_cut_id: {}".format(self.image_info[segid]['image_info'][ptype][side][seqid]['short_cut_id']))

                        # IF INTERSECTIONS
                        if side in ['I', 'T']:
                            continue

                        # IF DENSE
                        else:
                            # Continue if the given image is the last one in the sequence.
                            if index + 1 == len(self.image_info[segid]['image_info'][ptype][side].keys()):
                                continue

                            else:
                                # Get the prediction result
                                prediction = self.image_info[segid]['image_info'][ptype][side][seqid]['prediction'] # this line returns: {"dense":[file_name, yhat]}
                                yhat = prediction['dense'][1][0]

                                # ********************************
                                # APPLYING THE SERIES OF FUNCTIONS
                                # ********************************

                                # Get the next sequence_id
                                next_seqid = list(self.image_info[segid]['image_info'][ptype][side].keys())[index + 1]

                                c1 = self.image_info[segid]['image_info'][ptype][side][seqid]['location']
                                c2 = self.image_info[segid]['image_info'][ptype][side][next_seqid]['location'] # do it for index + 1

                                azimuth1 = self.image_info[segid]['image_info'][ptype][side][seqid]['azimuth']
                                azimuth2 = self.image_info[segid]['image_info'][ptype][side][next_seqid]['azimuth']

                                # Get n
                                n = find_n_given_yhat(yhat)

                                # Get d
                                d = find_d_given_n(n)

                                # d can be greate than 320 in error cases. Thisblows up the whole algorithm.
                                # Skip such images
                                if d >= 320:
                                    print("[Skipped images] segid: {}, ptype: {}, side: {}, seqid: {}".format(segid, ptype, side, seqid))
                                    continue

                                # Get the proportion!
                                x, W = find_overlap_gap_given_d(c1, c2, azimuth1, azimuth2, d, 1)
                                self.image_info[segid]['image_info'][ptype][side][seqid]['crop_percent'] = {"x": x, "W": W}
                                avg_gap_overlap[segid].append(x/W)

                                # Is it a gap or overlap?
                                # If perfect sequence
                                if x == 0: # side != "T" and
                                    # print("No need for image processing")
                                    continue

                                # If overlap
                                elif x > 0: # side != "T" and
                                    # print("Overlap!")
                                    # read the image file
                                    imfilename = self.image_info[segid]['image_info'][ptype][side][seqid]['filename']
                                    current_im = cv2.imread(os.path.join(image_path, imfilename))

                                    # Calculate pixel number for retaintion
                                    retain_amount = int(round(((W-x)/W)*640))
                                    # print("retain_amount: ", str(retain_amount))

                                    # Crop the image
                                    cropped_im  = current_im[:,1:retain_amount,:] # height (y), width (x), channel (z)
                                    imfilename_cropped = imfilename[0:-4] + "_mod.jpg"

                                    # Save the image
                                    try:
                                        cv2.imwrite(os.path.join(image_path, imfilename_cropped), cropped_im)
                                    except:
                                        identification = str(segid) + "-" + ptype + "-" + side + "-" + str(seqid)
                                        error_images[identification] = x/W


                                # If gap
                                elif x < 0: # side != "T" and
                                    # print("Gap!")
                                    # read the image file
                                    imfilename = self.image_info[segid]['image_info'][ptype][side][seqid]['filename']
                                    imfilename_stitch = imfilename[0:-4] + "_mod.jpg"
                                    current_im = cv2.imread(os.path.join(image_path, imfilename))

                                    # Get rotate angle
                                    theta = find_rotate_angle(x, W)

                                    # Download additional image in memory
                                    url_base = "https://maps.googleapis.com/maps/api/streetview?size=640x640&pano="
                                    location = self.image_info[segid]['image_info'][ptype][side][seqid]['pano_id'] + "&heading="
                                    heading =  str(self.image_info[segid]['image_info'][ptype][side][seqid]['azimuth'] + theta) + "&pitch=0&key="
                                    api_key = os.environ.get("google_api")
                                    url_full = url_base + location + heading + api_key
                                    rotated_im = url_to_image(url_full)

                                    # Stitch the two
                                    images = [current_im, rotated_im]
                                    stitcher = cv2.Stitcher_create(try_use_gpu = False)

                                    (status, stitched) = stitcher.stitch(images)
                                    self.image_info[segid]['image_info'][ptype][side][seqid]['stitched'] = status
                                    print("status is " + str(status) + "for " + imfilename_stitch)
                                    if status is not 0:
                                        continue
                                    else:
                                        cv2.imwrite(os.path.join(image_path, imfilename_stitch), stitched)

        return avg_gap_overlap, error_images

    # METHOD: a second prediction after the overlaps and gaps are taken care of.
    def predict_2(self, image_path, model_dense):
        """
        Purpose : Make the second prediction using the gap-overlap-processed images and append to the dictionary. Assumes 21 category - full model.
        Argument: None.
        Return  : Appends to the prediction dictionary.
        """
        # Sequentially extract each image
        for segid in self.image_info.keys():
            for ptype in self.image_info[segid]['image_info'].keys():
                for side in self.image_info[segid]['image_info'][ptype].keys():
                    for seqid in self.image_info[segid]['image_info'][ptype][side].keys():

                        # INTERSECTIONS
                        if side in ['I', 'T']:
                            continue

                        # DENSE
                        else:
                            file_name = self.image_info[segid]['image_info'][ptype][side][seqid]['filename']
                            file_name_mod = file_name[0:-4] + "_mod.jpg"
                            full_path_mod = os.path.join(image_path, file_name_mod)
                            # crop_percent may not exist if there was no overlap or gap --
                            if self.image_info[segid]['image_info'][ptype][side][seqid]['crop_percent'] is None:
                                continue
                            x = self.image_info[segid]['image_info'][ptype][side][seqid]['crop_percent']['x']

                            if x == 0: # No overlap or gap
                                continue

                            elif x > 0: # Overlap!
                                # Make prediction
                                image_mod = skimage.io.imread(full_path_mod)
                                yhat_mod = model_dense.detect([image_mod], verbose = 0)[0]

                                # Identify where sidewalk, buffer, and streetlight predictions are.
                                buffer_index = np.where(yhat_mod['class_ids'] == 9)[0]
                                sidewalk_index = np.where(yhat_mod['class_ids'] == 1)[0]
                                buffer_sidewalk_index = list(buffer_index) + list(sidewalk_index)

                                # Delete the corresponding items from the class_ids.
                                np.delete(yhat_mod['class_ids'], buffer_sidewalk_index)
                                np.delete(yhat_mod['rois'], buffer_sidewalk_index)
                                np.delete(yhat_mod['masks'], buffer_sidewalk_index)
                                np.delete(yhat_mod['scores'], buffer_sidewalk_index)

                                # Append
                                self.image_info[segid]['image_info'][ptype][side][seqid]['prediction']['dense'][2] = yhat_mod

                            elif x < 0: # Gap!
                                if self.image_info[segid]['image_info'][ptype][side][seqid][stitched] is 1:
                                    continue
                                else:
                                    # Make prediction
                                    image_mod = skimage.io.imread(full_path_mod)
                                    yhat_mod = model_dense.detect([image_mod], verbose = 0)

                                    # Append
                                    self.image_info[segid]['image_info'][ptype][side][seqid]['prediction']['dense'][2] = yhat_mod



    # METHOD: summary
    def summary_prediction_1(self, to_dataframe = True):
        """
        Purpose : Summarises the computer vision prediction into a format compatible with HS's audit.
        Argument: Nothing
        Return  : Summarized statistics appended to self.image_info['segid']['pred_summary']
        """
        # Sequentially extract each image
        for segid in self.image_info.keys():
            summarise = {"SegID": segid,
                         'walk_signal_S': None,
                         'curb_ramp_S': None,
                         'crosswalk_S': None,
                         'seating': [],
                         'streetlight_num': [],
                         'streetlight_up': [],
                         'lightpole':[],
                         'bad_building': [],
                         'boarded_building': [],
                         'good_building': [],
                         'graffiti': [],
                         'sidewalk': [],
                         'trip_hazard': [],
                         'buffer': [],
                         'buffer_11': [],
                         'walk_signal_E': None,
                         'curb_ramp_E': None,
                         'crosswalk_E': None
                        }

            for ptype in self.image_info[segid]['image_info'].keys():
                for side in self.image_info[segid]['image_info'][ptype].keys():
                    for index, seqid in enumerate(list(self.image_info[segid]['image_info'][ptype][side].keys())):

                        # Prediction info == intersection & seqid == 1
                        if seqid == 1 and list(self.image_info[segid]['image_info'][ptype][side][seqid]['prediction'].keys())[0] == 'intersection':

                            # Extract prediction for intersection 1 (-45?)
                            yhat0 = self.image_info[segid]['image_info'][ptype][side][seqid]['prediction']['intersection'][0][1][0]
                            class_ids0 = yhat0['class_ids']

                            # Extract prediction for intersection 1 (+45?)
                            yhat1 = self.image_info[segid]['image_info'][ptype][side][seqid]['prediction']['intersection'][1][1][0]
                            class_ids1 = yhat1['class_ids']

                            # Intersection azimuth
                            summarise['walk_signal_S'] = np.array(np.where(class_ids0 == 4)).shape[1] + np.array(np.where(class_ids1 == 4)).shape[1]
                            summarise['curb_ramp_S'] = np.array(np.where(class_ids0 == 6)).shape[1] + np.array(np.where(class_ids1 == 6)).shape[1]
                            summarise['crosswalk_S'] = np.array(np.where(class_ids0 == 5)).shape[1] + np.array(np.where(class_ids1 == 5)).shape[1]


                        # Prediction info == intersection & seqid == 1000
                        elif seqid == 1000 and list(self.image_info[segid]['image_info'][ptype][side][seqid]['prediction'].keys())[0] == 'intersection':

                            # Extract prediction for intersection 1
                            yhat0 = self.image_info[segid]['image_info'][ptype][side][seqid]['prediction']['intersection'][0][1][0]
                            class_ids0 = yhat0['class_ids']

                            # Extract prediction for intersection 1000
                            yhat1 = self.image_info[segid]['image_info'][ptype][side][seqid]['prediction']['intersection'][1][1][0]
                            class_ids1 = yhat1['class_ids']

                            # Intersection azimuth
                            summarise['walk_signal_E'] = np.array(np.where(class_ids0 == 4)).shape[1] + np.array(np.where(class_ids1 == 4)).shape[1]
                            summarise['curb_ramp_E'] = np.array(np.where(class_ids0 == 6)).shape[1] + np.array(np.where(class_ids1 == 6)).shape[1]
                            summarise['crosswalk_E'] = np.array(np.where(class_ids0 == 5)).shape[1] + np.array(np.where(class_ids1 == 5)).shape[1]

                        # Prediction info == dense
                        else:
                            # Dense vs Top
                            if 'dense' in self.image_info[segid]['image_info'][ptype][side][seqid]['prediction'].keys():
                                # Extract prediction
                                yhat_original = self.image_info[segid]['image_info'][ptype][side][seqid]['prediction']['dense'][1][0]

                                # Deleting overlapped houseing prediction
                                bad_building_index = np.array(np.where(yhat_original['class_ids'] == 6))[0]
                                boarded_building_index = np.array(np.where(yhat_original['class_ids'] == 18))[0]
                                bad_boarded_index = np.append(bad_building_index, boarded_building_index)
                                good_building_index = np.array(np.where(yhat_original['class_ids'] == 7))[0]

                                delete_index = []

                                if len(bad_boarded_index) > 0 or len(good_building_index) > 0:

                                    # for each good_building_mask, multiply each bad_building_mask and see if it has 1s.
                                    good_building_masks = yhat_original['masks'][:,:,good_building_index]
                                    bad_building_masks = yhat_original['masks'][:,:,bad_building_index]
                                    boarded_building_masks = yhat_original['masks'][:,:,boarded_building_index]

                                    # loop through good building mask * bad building_mask
                                    for i in range(good_building_masks.shape[2]):
                                        good_building_mask = good_building_masks[:,:,i]

                                        # checking against bad building
                                        for j in range(bad_building_masks.shape[2]):
                                            bad_building_mask = bad_building_masks[:,:,j]

                                            # multiply the two masks
                                            good_bad_multiply = good_building_mask*bad_building_mask

                                            # if there is an overlap..
                                            if np.sum(good_bad_multiply) > 0:
                                                # idenfity who they are and what their confidence is
                                                current_good_index = good_building_index[i]
                                                current_bad_index = bad_building_index[j]
                                                good_bldg_score = yhat_original['scores'][current_good_index]
                                                bad_bldg_score = yhat_original['scores'][current_bad_index]

                                                # tell which one won, and delete the lost one
                                                if good_bldg_score > bad_bldg_score:
                                                    delete_index.append(current_bad_index)
                                                else:
                                                    delete_index.append(current_good_index)

                                        # checking against boarded building
                                        for k in range(boarded_building_masks.shape[2]):
                                            boarded_building_mask = boarded_building_masks[:,:,k]

                                            # multiply the two masks
                                            good_boarded_multiply = good_building_mask*boarded_building_mask

                                            # if there is an overlap..
                                            if np.sum(good_boarded_multiply) > 0:
                                                # idenfity who they are and what their confidence is
                                                current_good_index = good_building_index[i]
                                                current_boarded_index = boarded_building_index[k]
                                                good_bldg_score = yhat_original['scores'][current_good_index]
                                                boarded_bldg_score = yhat_original['scores'][current_boarded_index]

                                                # tell which one won, and delete the lost one
                                                if good_bldg_score > boarded_bldg_score:
                                                    delete_index.append(current_boarded_index)
                                                else:
                                                    delete_index.append(current_good_index)

                                yhat_original['class_ids'] = np.delete(yhat_original['class_ids'], delete_index)

                                # Count
                                summarise['bad_building'].append(
                                    np.array(np.where(yhat_original['class_ids'] == 6)).shape[1]
                                ) # bad_buidling == 5
                                summarise['boarded_building'].append(
                                    np.array(np.where(yhat_original['class_ids'] == 18)).shape[1]
                                ) # boarded_building == 13
                                summarise['good_building'].append(
                                    np.array(np.where(yhat_original['class_ids'] == 7)).shape[1]
                                ) # good_building == 6
                                summarise['seating'].append(
                                    np.array(np.where(yhat_original['class_ids'] == 11)).shape[1]
                                ) # seating == 10
                                summarise['streetlight_num'].append(
                                    np.array(np.where(yhat_original['class_ids'] == 10)).shape[1]
                                ) # streetlight == 9
                                summarise['lightpole'].append(
                                    np.array(np.where(yhat_original['class_ids'] == 17)).shape[1]
                                ) # lightpole == 12
                                summarise['graffiti'].append(
                                    np.array(np.where(yhat_original['class_ids'] == 15)).shape[1]
                                ) # graffiti == 11

                                # Presence
                                summarise['sidewalk'].append(np.array(np.where(yhat_original['class_ids'] == 1)).shape[1] > 0) # sidewalk == 1

                                # Trip hazard and buffer are dependent on the presence of sidewalk
                                if np.array(np.where(yhat_original['class_ids'] == 1)).shape[1] > 0:
                                    summarise['buffer'].append(len(np.where(yhat_original['class_ids'] == 9)[0])) # buffer == 8

                                    ## Trip hazards are only counted if it is overlapped with sidewalk mask
                                    # trip hazard
                                    if len(np.array(np.where(yhat_original['class_ids'] == 5))[0]) > 0:
                                        trip_hazard_index = np.array(np.where(yhat_original['class_ids'] == 5))[0]
                                        trip_hazard_masks = yhat_original['masks'][:,:,trip_hazard_index]

                                        # sidewalk
                                        sidewalk_index = np.array(np.where(yhat_original['class_ids'] == 1))[0]
                                        sidewalk_masks = yhat_original['masks'][:,:,sidewalk_index]

                                        # merge all trip hazard mask into one piece
                                        for i in range(sidewalk_masks.shape[2]):
                                            if i == 0:
                                                continue
                                            else:
                                                sidewalk_masks[:,:,0] = sidewalk_masks[:,:,0] + sidewalk_masks[:,:,i]

                                        sidewalk_masks = sidewalk_masks[:,:,0]

                                        # Check whether they overlap by multiplying
                                        trip_hazard_counter = 0
                                        for i in range(trip_hazard_masks.shape[2]):
                                            th_by_sidewalk = trip_hazard_masks[:,:,i] * sidewalk_masks

                                        # Check overlap exists
                                        if np.sum(th_by_sidewalk) > 0.5*np.sum(trip_hazard_masks[:,:,i]):
                                            trip_hazard_counter += 1
                                        else:
                                            continue

                                        summarise['trip_hazard'].append(trip_hazard_counter)

                            # Dense vs Top
                            elif 'top' in self.image_info[segid]['image_info'][ptype][side][seqid]['prediction'].keys():
                                yhat = self.image_info[segid]['image_info'][ptype][side][seqid]['prediction']['top'][1][0]
                                class_ids = yhat['class_ids']
                                summarise['streetlight_up'].append(np.array(np.where(class_ids == 2)).shape[1])

            # Summarise them all by each segment
            # Count
            summarise['seating'] = sum(summarise['seating'])
            summarise['streetlight_num'] = sum(summarise['streetlight_num'])
            summarise['streetlight_up'] = sum(summarise['streetlight_up'])
            summarise['lightpole'] = sum(summarise['lightpole'])
            summarise['bad_building'] = sum(summarise['bad_building'])
            summarise['boarded_building'] = sum(summarise['boarded_building'])
            summarise['good_building'] = sum(summarise['good_building'])

            # Presence
            summarise['graffiti'] = sum(summarise['graffiti'])
            summarise['trip_hazard'] = sum(summarise['trip_hazard'])

            # Sidewalk - checking whether there are more than two images
            # that sequentially have sidewalks
            sidewalk_str = str()
            for element in summarise['sidewalk']:
                sidewalk_str += str(element*1)

            # If there are at least two consecutive images with sidewalks, give 'sidewalk' 1
            if "11" in sidewalk_str:
                summarise['sidewalk'] = True
            else:
                summarise['sidewalk'] = False

            # Buffers - checking whether there are more than two images
            # that sequentially have buffers
            buffer_str = str()
            for element in summarise['buffer']:
                buffer_str += str(element*1)

            # If there are at least two consecutive images with sidewalks, give 'sidewalk' 1
            if "11" in buffer_str:
                summarise['buffer_11'] = True
            else:
                summarise['buffer_11'] = False

            summarise['buffer'] = sum(summarise['buffer'])

            # convert to df if the argument is given
            if to_dataframe:
                try:
                    current_df = pd.concat([current_df, pd.DataFrame.from_dict(summarise, orient = 'index').T])
                except NameError:
                    current_df = pd.DataFrame.from_dict(summarise, orient = 'index').T

        return current_df


    # METHOD: summary
    def summary_prediction_2(self, to_dataframe = True):
        """
        Purpose : Summarises the computer vision prediction into a format compatible with HS's audit.
        Argument: Nothing
        Return  : Summarized statistics appended to self.image_info['segid']['pred_summary']
        """
        # Sequentially extract each image
        for segid in self.image_info.keys():
            summarise = {"SegID": segid,
                         'walk_signal_S': None,
                         'curb_ramp_S': None,
                         'crosswalk_S': None,
                         'seating': [],
                         'streetlight_num': [],
                         'streetlight_up': [],
                         'lightpole':[],
                         'bad_building': [],
                         'boarded_building': [],
                         'good_building': [],
                         'graffiti': [],
                         'sidewalk': [],
                         'trip_hazard': [],
                         'buffer': [],
                         'buffer_11': [],
                         'walk_signal_E': None,
                         'curb_ramp_E': None,
                         'crosswalk_E': None
                        }

            for ptype in self.image_info[segid]['image_info'].keys():
                for side in self.image_info[segid]['image_info'][ptype].keys():
                    for index, seqid in enumerate(list(self.image_info[segid]['image_info'][ptype][side].keys())):

                        # Prediction info == intersection & seqid == 1
                        if seqid == 1 and list(self.image_info[segid]['image_info'][ptype][side][seqid]['prediction'].keys())[0] == 'intersection':

                            # Extract prediction for intersection 1 (-45?)
                            yhat0 = self.image_info[segid]['image_info'][ptype][side][seqid]['prediction']['intersection'][0][1][0]
                            class_ids0 = yhat0['class_ids']

                            # Extract prediction for intersection 1 (+45?)
                            yhat1 = self.image_info[segid]['image_info'][ptype][side][seqid]['prediction']['intersection'][1][1][0]
                            class_ids1 = yhat1['class_ids']

                            # Intersection azimuth
                            summarise['walk_signal_S'] = np.array(np.where(class_ids0 == 4)).shape[1] + np.array(np.where(class_ids1 == 4)).shape[1]
                            summarise['curb_ramp_S'] = np.array(np.where(class_ids0 == 6)).shape[1] + np.array(np.where(class_ids1 == 6)).shape[1]
                            summarise['crosswalk_S'] = np.array(np.where(class_ids0 == 5)).shape[1] + np.array(np.where(class_ids1 == 5)).shape[1]


                        # Prediction info == intersection & seqid == 1000
                        elif seqid == 1000 and list(self.image_info[segid]['image_info'][ptype][side][seqid]['prediction'].keys())[0] == 'intersection':

                            # Extract prediction for intersection 1
                            yhat0 = self.image_info[segid]['image_info'][ptype][side][seqid]['prediction']['intersection'][0][1][0]
                            class_ids0 = yhat0['class_ids']

                            # Extract prediction for intersection 1000
                            yhat1 = self.image_info[segid]['image_info'][ptype][side][seqid]['prediction']['intersection'][1][1][0]
                            class_ids1 = yhat1['class_ids']

                            # Intersection azimuth
                            summarise['walk_signal_E'] = np.array(np.where(class_ids0 == 4)).shape[1] + np.array(np.where(class_ids1 == 4)).shape[1]
                            summarise['curb_ramp_E'] = np.array(np.where(class_ids0 == 6)).shape[1] + np.array(np.where(class_ids1 == 6)).shape[1]
                            summarise['crosswalk_E'] = np.array(np.where(class_ids0 == 5)).shape[1] + np.array(np.where(class_ids1 == 5)).shape[1]

                        # Prediction info == dense
                        else:
                            # Dense vs Top
                            if 'dense' in self.image_info[segid]['image_info'][ptype][side][seqid]['prediction'].keys():
                                # Extract prediction
                                yhat_original = self.image_info[segid]['image_info'][ptype][side][seqid]['prediction']['dense'][1][0]
                                yhat_mod = self.image_info[segid]['image_info'][ptype][side][seqid]['prediction']['dense'][2][0]

                                # Deleting overlapped houseing prediction
                                bad_building_index = np.array(np.where(yhat_original['class_ids'] == 6))[0]
                                boarded_building_index = np.array(np.where(yhat_original['class_ids'] == 18))[0]
                                bad_boarded_index = np.append(bad_building_index, boarded_building_index)
                                good_building_index = np.array(np.where(yhat_original['class_ids'] == 7))[0]

                                delete_index = []

                                if len(bad_boarded_index) > 0 or len(good_building_index) > 0:

                                    # for each good_building_mask, multiply each bad_building_mask and see if it has 1s.
                                    good_building_masks = yhat_original['masks'][:,:,good_building_index]
                                    bad_building_masks = yhat_original['masks'][:,:,bad_building_index]
                                    boarded_building_masks = yhat_original['masks'][:,:,boarded_building_index]

                                    # loop through good building mask * bad building_mask
                                    for i in range(good_building_masks.shape[2]):
                                        good_building_mask = good_building_masks[:,:,i]

                                        # checking against bad building
                                        for j in range(bad_building_masks.shape[2]):
                                            bad_building_mask = bad_building_masks[:,:,j]

                                            # multiply the two masks
                                            good_bad_multiply = good_building_mask*bad_building_mask

                                            # if there is an overlap..
                                            if np.sum(good_bad_multiply) > 0:
                                                # idenfity who they are and what their confidence is
                                                current_good_index = good_building_index[i]
                                                current_bad_index = bad_building_index[j]
                                                good_bldg_score = yhat_original['scores'][current_good_index]
                                                bad_bldg_score = yhat_original['scores'][current_bad_index]

                                                # tell which one won, and delete the lost one
                                                if good_bldg_score > bad_bldg_score:
                                                    delete_index.append(current_bad_index)
                                                else:
                                                    delete_index.append(current_good_index)

                                        # checking against boarded building
                                        for k in range(boarded_building_masks.shape[2]):
                                            boarded_building_mask = boarded_building_masks[:,:,k]

                                            # multiply the two masks
                                            good_boarded_multiply = good_building_mask*boarded_building_mask

                                            # if there is an overlap..
                                            if np.sum(good_boarded_multiply) > 0:
                                                # idenfity who they are and what their confidence is
                                                current_good_index = good_building_index[i]
                                                current_boarded_index = boarded_building_index[k]
                                                good_bldg_score = yhat_original['scores'][current_good_index]
                                                boarded_bldg_score = yhat_original['scores'][current_boarded_index]

                                                # tell which one won, and delete the lost one
                                                if good_bldg_score > boarded_bldg_score:
                                                    delete_index.append(current_boarded_index)
                                                else:
                                                    delete_index.append(current_good_index)

                                yhat_original['class_ids'] = np.delete(yhat_original['class_ids'], delete_index)

                                # Count
                                summarise['bad_building'].append(
                                    np.array(np.where(yhat_original['class_ids'] == 6)).shape[1] + np.array(np.where(yhat_mod['class_ids'] == 6)).shape[1]
                                ) # bad_buidling == 5
                                summarise['boarded_building'].append(
                                    np.array(np.where(yhat_original['class_ids'] == 18)).shape[1] + np.array(np.where(yhat_mod['class_ids'] == 18)).shape[1]
                                ) # boarded_building == 13
                                summarise['good_building'].append(
                                    np.array(np.where(yhat_original['class_ids'] == 7)).shape[1] + np.array(np.where(yhat_mod['class_ids'] == 7)).shape[1]
                                ) # good_building == 6
                                summarise['seating'].append(
                                    np.array(np.where(yhat_original['class_ids'] == 11)).shape[1] + np.array(np.where(yhat_mod['class_ids'] == 11)).shape[1]
                                ) # seating == 10
                                summarise['streetlight_num'].append(
                                    np.array(np.where(yhat_original['class_ids'] == 10)).shape[1] + np.array(np.where(yhat_mod['class_ids'] == 10)).shape[1]
                                ) # streetlight == 9
                                summarise['lightpole'].append(
                                    np.array(np.where(yhat_original['class_ids'] == 17)).shape[1] + np.array(np.where(yhat_mod['class_ids'] == 17)).shape[1]
                                ) # lightpole == 12
                                summarise['graffiti'].append(
                                    np.array(np.where(yhat_original['class_ids'] == 15)).shape[1] + np.array(np.where(yhat_mod['class_ids'] == 15)).shape[1]
                                ) # graffiti == 11

                                # Presence
                                summarise['sidewalk'].append(np.array(np.where(yhat_original['class_ids'] == 1)).shape[1] > 0) # sidewalk == 1

                                # Trip hazard and buffer are dependent on the presence of sidewalk
                                if np.array(np.where(yhat_original['class_ids'] == 1)).shape[1] > 0:
                                    summarise['buffer'].append(len(np.where(yhat_original['class_ids'] == 9)[0])) # buffer == 8

                                    ## Trip hazards are only counted if it is overlapped with sidewalk mask
                                    # trip hazard
                                    if len(np.array(np.where(yhat_original['class_ids'] == 5))[0]) > 0:
                                        trip_hazard_index = np.array(np.where(yhat_original['class_ids'] == 5))[0]
                                        trip_hazard_masks = yhat_original['masks'][:,:,trip_hazard_index]

                                        # sidewalk
                                        sidewalk_index = np.array(np.where(yhat_original['class_ids'] == 1))[0]
                                        sidewalk_masks = yhat_original['masks'][:,:,sidewalk_index]

                                        # merge all trip hazard mask into one piece
                                        for i in range(sidewalk_masks.shape[2]):
                                            if i == 0:
                                                continue
                                            else:
                                                sidewalk_masks[:,:,0] = sidewalk_masks[:,:,0] + sidewalk_masks[:,:,i]

                                        sidewalk_masks = sidewalk_masks[:,:,0]

                                        # Check whether they overlap by multiplying
                                        trip_hazard_counter = 0
                                        for i in range(trip_hazard_masks.shape[2]):
                                            th_by_sidewalk = trip_hazard_masks[:,:,i] * sidewalk_masks

                                        # Check overlap exists
                                        if np.sum(th_by_sidewalk) > 0.5*np.sum(trip_hazard_masks[:,:,i]):
                                            trip_hazard_counter += 1
                                        else:
                                            continue

                                        summarise['trip_hazard'].append(trip_hazard_counter + np.array(np.where(yhat_mod['class_ids'] == 5)).shape[1])

                            # Dense vs Top
                            elif 'top' in self.image_info[segid]['image_info'][ptype][side][seqid]['prediction'].keys():
                                yhat = self.image_info[segid]['image_info'][ptype][side][seqid]['prediction']['top'][1][0]
                                class_ids = yhat['class_ids']
                                summarise['streetlight_up'].append(np.array(np.where(class_ids == 2)).shape[1])

            # Summarise them all by each segment
            # Count
            summarise['seating'] = sum(summarise['seating'])
            summarise['streetlight_num'] = sum(summarise['streetlight_num'])
            summarise['streetlight_up'] = sum(summarise['streetlight_up'])
            summarise['lightpole'] = sum(summarise['lightpole'])
            summarise['bad_building'] = sum(summarise['bad_building'])
            summarise['boarded_building'] = sum(summarise['boarded_building'])
            summarise['bad_building'] = summarise['bad_building'] + summarise['boarded_building']
            summarise['good_building'] = sum(summarise['good_building'])

            # Presence
            summarise['graffiti'] = sum(summarise['graffiti'])
            summarise['trip_hazard'] = sum(summarise['trip_hazard'])

            # Sidewalk - checking whether there are more than two images
            # that sequentially have sidewalks
            sidewalk_str = str()
            for element in summarise['sidewalk']:
                sidewalk_str += str(element*1)

            # If there are at least two consecutive images with sidewalks, give 'sidewalk' 1
            if "11" in sidewalk_str:
                summarise['sidewalk'] = True
            else:
                summarise['sidewalk'] = False

            # Buffers - checking whether there are more than two images
            # that sequentially have buffers
            buffer_str = str()
            for element in summarise['buffer']:
                buffer_str += str(element*1)

            # If there are at least two consecutive images with sidewalks, give 'sidewalk' 1
            if "11" in buffer_str:
                summarise['buffer_11'] = True
            else:
                summarise['buffer_11'] = False

            summarise['buffer'] = sum(summarise['buffer'])

            # convert to df if the argument is given
            if to_dataframe:
                try:
                    current_df = pd.concat([current_df, pd.DataFrame.from_dict(summarise, orient = 'index').T])
                except NameError:
                    current_df = pd.DataFrame.from_dict(summarise, orient = 'index').T

        return current_df

    # Class attributes
    class_code_dense = {0:"BG",
                        1:"sidewalk",
                        2:"road",
                        3:"planter",
                        4:"landscape",
                        5:"trip_hazard",
                        6:"bad_building",
                        7:"good_building",
                        8:"utility_pole",
                        9:"buffer",
                        10:"street_light",
                        11:"seating",
                        12:"walk_signal",
                        13:"crosswalk",
                        14:"curb_ramp",
                        15:"graffiti",
                        16:"bike_mark",
                        17:"lightpole",
                        18:"boarded_house",
                        19:"wall",
                        20:"driveway"}

    class_code_intersection = {0:"BG",
                               1:"utility_pole",
                               2:"street_light",
                               3:"seating",
                               4:"walk_signal",
                               5:"crosswalk",
                               6:"curb_ramp",
                               7:"graffiti",
                               8:"bike_mark",
                               9:"lightpole"}

    class_code_top = {0: "BG",
                      1:"utility_pole",
                      2:"streetlight_up"}
