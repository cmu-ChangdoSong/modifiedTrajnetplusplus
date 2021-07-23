""" Categorization of Primary Pedestrian """

import numpy as np
import pysparkling

# import trajnetplusplustools
import data
from kalman import predict as kalman_predict
from interactions import check_interaction, group
from interactions import get_interaction_type

import pickle
# from orca_helper import predict_all

from data_loader import DataLoader as dl


def get_type(scene, args):
    '''
    Categorization of Single Scene
    :param scene: All trajectories as TrackRows, args
    :return: The type of the traj
    '''

    def interaction(rows, pos_range, dist_thresh, obs_len):
        '''
        :return: Determine if interaction exists and type (optionally)
        '''
        interaction_matrix = check_interaction(rows, pos_range=pos_range,
                                               dist_thresh=dist_thresh, obs_len=obs_len)
        return np.any(interaction_matrix)

    # Category Tags
    mult_tag = []
    sub_tag = []

    # check for interactions
    if interaction(scene, args.inter_pos_range, args.inter_dist_thresh, args.obs_len) \
            or np.any(group(scene, args.grp_dist_thresh, args.grp_std_thresh, args.obs_len)):
        mult_tag.append(3)

    # Non-Linear (No explainable reason)
    else:
        mult_tag.append(4)

    # Interaction Types
    if mult_tag[0] == 3:
        type, data = get_interaction_type(scene, args.inter_pos_range,
                                          args.inter_dist_thresh, args.obs_len)
    else:
        type = []
        data = {}

    return mult_tag[0], mult_tag, type, data


def trajectory_type(track_id=0, args=None):
    """ Categorization of all scenes """

    # Construct scene for each of pedestrian (360 in total for eth dataset)
    # Each scene is created over the duration of each pedestrian's apearance, from starting frame to ending frame,
    # and the scene will contain paths of all people that are in those frames
    def constructScenes():
        # use data_loader here
        dataloader = dl()

        frame_ped_positions = dataloader.frameId_people_positions
        ped_id = dataloader.personIdListSorted
        start_frames = dataloader.people_start_frame
        end_frames = dataloader.people_end_frame

        scenes = {}
        # create scene for each pedestrian
        for i in range(len(ped_id)):
            primary_id = ped_id[i]
            start = start_frames[primary_id]
            end = end_frames[primary_id]

            pos_in_timeframe = {}
            max_columnlen = 0
            frame = start
            # collect pedestrian's position from the begining frame until the end frame
            while frame <= end:
                ped_positions = frame_ped_positions[frame]

                for id in ped_positions.keys():
                    coord = ped_positions[id]
                    if id not in pos_in_timeframe.keys():
                        pos_in_timeframe[id] = []
                    pos_in_timeframe[id].append(coord)
                    max_columnlen = len(pos_in_timeframe[id]) if len(
                        pos_in_timeframe[id]) > max_columnlen else max_columnlen

                # the original data has frameids with 6 intervals in between
                # but we will take into account the interpolated positions between
                # each frame which has been calculated in data_loader
                frame += 1

            scene = np.empty((len(pos_in_timeframe), max_columnlen, 2))
            scene[:] = np.NaN

            # always add primary pedestrian's path to the 0 index so that it confroms
            # with original data format
            scene[0][:len(pos_in_timeframe[primary_id])] = np.asarray(
                pos_in_timeframe[primary_id])

            # add neighbor's paths to the scene
            i = 1
            for id, _ in pos_in_timeframe.items():
                if id == primary_id:
                    continue
                scene[i][:len(pos_in_timeframe[id])] = np.asarray(
                    pos_in_timeframe[id])
                i += 1

            scenes[primary_id] = scene

        return scenes

    scenes = constructScenes()
    # Filtered Frames and Scenes
    new_frames = set()
    new_scenes = []

    # Initialize Tag Stats to be collected
    tags = {1: [], 2: [], 3: [], 4: []}
    mult_tags = {1: [], 2: [], 3: [], 4: []}
    sub_tags = {1: [], 2: [], 3: [], 4: []}
    col_count = 0

    if len(scenes) == 0:
        raise Exception('No scenes found')

    leader_follower_res = []
    collision_avoidance_res = []
    group_res = 0
    others_res = 0

    for index, key in enumerate(scenes):
        if (index+1) % 50 == 0:
            print(index)

        primary_id = key
        scene = scenes[key]

        # Get Tag
        tag, mult_tag, sub_tag, data = get_type(scene, args)

        if 1 in data.keys():
            leader_follower_res.append(data[1])

        if 2 in data.keys():
            collision_avoidance_res.append(data[2])

        if 3 in data.keys():
            group_res += 1

        if 4 in data.keys():
            others_res += 1

        if np.random.uniform() < args.acceptance[tag - 1]:
            # Update Tags
            tags[tag].append(track_id)
            for tt in mult_tag:
                mult_tags[tt].append(track_id)
            for st in sub_tag:
                sub_tags[st].append(track_id)

            # Define Scene_Tag
            scene_tag = []
            scene_tag.append(tag)
            scene_tag.append(sub_tag)

            track_id += 1

    # Number of collisions found
    print("Col Count: ", col_count)

    if scenes:
        print("Total Scenes: ", index)

        # Types:
        print("Main Tags")
        print("Type 1: ", len(tags[1]), "Type 2: ", len(tags[2]),
              "Type 3: ", len(tags[3]), "Type 4: ", len(tags[4]))
        print("Sub Tags")
        print("LF: ", len(sub_tags[1]), "CA: ", len(sub_tags[2]),
              "Group: ", len(sub_tags[3]), "Others: ", len(sub_tags[4]))

    return track_id
