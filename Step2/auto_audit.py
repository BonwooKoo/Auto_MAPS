import os
from  urllib import request
import pandas as pd
import numpy as np
import sys
# import joblib
import time
import matplotlib.pyplot as plt
# import random
# import math
# import pickle
import skimage
MASK_RCNN_DIR = "/content/Mask_RCNN" # Assuming using Google Colab

sys.path.append(MASK_RCNN_DIR)
from mrcnn import visualize

class auto_audit_df:
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

                        # Append
                        self.image_info[segid]['image_info'][ptype]["I"][seq_id] = {
                            'filename': [
                                # row 1
                                "micro" + "_SegID-" + str(segid) + "_type-"+ ptype + "_side-I_seqid-" + str(row1.sequence_id) + "_panoid-" + row1.pano_id + "_azi-" + str(round(row1.azimuth, 2)) + ".jpg",
                                # row 2
                                "micro" + "_SegID-" + str(segid) + "_type-"+ ptype + "_side-I_seqid-" + str(row2.sequence_id) + "_panoid-" + row2.pano_id + "_azi-" + str(round(row2.azimuth, 2)) + ".jpg"],
                            'pano_id': row1.pano_id,
                            'y_coord': str(row1.y_coord),
                            'x_coord': str(row1.x_coord),
                            'location': str(row1.y_coord) + "," + str(row1.x_coord),
                            'azimuth': [row1.azimuth, row2.azimuth],
                            'length': None,
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
                                'filename': "micro" + "_SegID-" + str(segid) + "_type-"+ ptype + "_side-" + df_by_seq_id.LR.iat[0] + "_seqid-" + str(df_by_seq_id.sequence_id.iat[0]) + "_panoid-" + df_by_seq_id.pano_id.iat[0] + "_azi-" + str(round(df_by_seq_id.azimuth.iat[0],2)) + ".jpg",
                                'pano_id': df_by_seq_id.pano_id.iat[0],
                                'y_coord': str(df_by_seq_id.y_coord.iat[0]),
                                'x_coord': str(df_by_seq_id.x_coord.iat[0]),
                                'location': str(df_by_seq_id.y_coord.iat[0]) + "," + str(df_by_seq_id.x_coord.iat[0]),
                                'azimuth': df_by_seq_id.azimuth.iat[0],
                                'length': None,
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
    def download_gsv(self, download_path, api_key):
        # Downloads gsv images
        # argument: download_path - path to the folder where images will be downloaded
        #           api_key       - Google Street View API key
        # return:   img - a jpg image

        for segid in self.image_info.keys():
            for ptype in self.image_info[segid]['image_info'].keys():
                for side in self.image_info[segid]['image_info'][ptype].keys():
                    for seqid in self.image_info[segid]['image_info'][ptype][side].keys():

                        # Extract needed parameters
                        y_coord = self.image_info[segid]['image_info'][ptype][side][seqid]['y_coord']
                        x_coord = self.image_info[segid]['image_info'][ptype][side][seqid]['x_coord']
                        pano_id = self.image_info[segid]['image_info'][ptype][side][seqid]['pano_id']
                        azimuth = self.image_info[segid]['image_info'][ptype][side][seqid]['azimuth']

                        # Parameters
                        base_url = "https://maps.googleapis.com/maps/api/streetview?size=640x640&"
                        location = "location={},{}&".format(y_coord, x_coord)
                        key = "key=" + api_key

                        # Different heading for dense vs. intersection
                        if side in ["L", "R"]:
                            pitch = "pitch=0&"
                            heading = "heading={}&".format(azimuth)
                            full_url = base_url + location + heading + pitch + key

                            # File name
                            file_name = "micro" + "_SegID-" + str(segid) + "_type-"+ ptype + "_side-" + side + "_seqid-" + str(seqid) + "_panoid-" + pano_id + "_azi-" + str(round(azimuth,2)) + ".jpg"
                            full_path = os.path.join(download_path, file_name)

                            # Download if not done already
                            if os.path.exists(full_path) is False:
                                request.urlretrieve(full_url, full_path)

                        elif side == "T":
                            pitch = "pitch=90&"
                            heading = "heading={}&".format(azimuth)
                            full_url = base_url + location + heading + pitch + key

                            # File name
                            file_name = "micro" + "_SegID-" + str(segid) + "_type-"+ ptype + "_side-" + side + "_seqid-" + str(seqid) + "_panoid-" + pano_id + "_azi-" + str(round(azimuth,2)) + ".jpg"
                            full_path = os.path.join(download_path, file_name)

                            # Download if not done already
                            if os.path.exists(full_path) is False:
                                request.urlretrieve(full_url, full_path)

                        elif side == "I": # when the point is intersection
                            pitch = "pitch=0&"
                            azimuth_1 = "heading={}&".format(azimuth[0])
                            azimuth_2 = "heading={}&".format(azimuth[1])
                            full_url_1 = base_url + location + azimuth_1 + pitch + key
                            full_url_2 = base_url + location + azimuth_2 + pitch + key

                            # File name
                            file_name_1 = "micro" + "_SegID-" + str(segid) + "_type-"+ ptype + "_side-" + side + "_seqid-" + str(seqid) + "_panoid-" + pano_id + "_azi-" + str(round(azimuth[0],2)) + ".jpg"
                            file_name_2 = "micro" + "_SegID-" + str(segid) + "_type-"+ ptype + "_side-" + side + "_seqid-" + str(seqid) + "_panoid-" + pano_id + "_azi-" + str(round(azimuth[1],2)) + ".jpg"
                            full_path_1 = os.path.join(download_path, file_name_1)
                            full_path_2 = os.path.join(download_path, file_name_2)

                            # Download if not done already
                            if os.path.exists(full_path_1) is False:
                                request.urlretrieve(full_url_1, full_path_1)

                            if os.path.exists(full_path_2) is False:
                                request.urlretrieve(full_url_2, full_path_2)


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

    # METHOD: summary
    def prediction_summary(self, to_dataframe = True):
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
                                    summarise['trip_hazard'].append(len(np.where(yhat_original['class_ids'] == 5)[0])) # trip_hazard == 5

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
