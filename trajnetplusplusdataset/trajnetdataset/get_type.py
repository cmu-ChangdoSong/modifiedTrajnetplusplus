""" Categorization of Primary Pedestrian """

import numpy as np
import pysparkling

import data
from kalman import predict as kalman_predict
from interactions import check_interaction, group
from interactions import get_interaction_type

import pickle

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

    def split_by_size(arr, size):
        res = np.split(arr, np.arange(size, len(arr), size))
        # delete elements that has length less than (size)
        d = []
        for (i, e) in enumerate(res):
            if len(e) < 40:
                d.append(i)
        return np.delete(res, d, axis=0)

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
        pred_velocity = dataloader.people_velocity_complete

        scenes = {}
        # create scene for each pedestrian
        for i in range(len(ped_id)):
            primary_id = ped_id[i]
            start = start_frames[primary_id]
            end = end_frames[primary_id]

            # get all pedestrian id that appears in the timeframe
            ped_timeframe = set()

            timeframe_len = end - start + 1
            for f in range(timeframe_len):
                f_id = start + f
                ped_timeframe.update(
                    set(frame_ped_positions[f_id].keys()))

            # collect all pedestrian's position from the begining frame until the end frame
            scene_pre = {}
            for p_id in ped_timeframe:
                scene_pre[p_id] = [np.NaN] * timeframe_len

            for f in range(timeframe_len):
                f_id = start + f

                for p_id in scene_pre:
                    if p_id in frame_ped_positions[f_id].keys():
                        scene_pre[p_id][f] = frame_ped_positions[f_id][p_id]
                    else:
                        # take care of the NaNs.
                        # We will use the pedestrians id's start & end frame
                        # to calculate this.

                        # First get first velocity if current frame is before start frame,
                        # get last velocity if current frame is after end frame,
                        if f_id < start_frames[p_id]:
                            # TODO:might need to double-check if velocity calculations?
                            velocity = pred_velocity[p_id][0]
                            first_coord = frame_ped_positions[start_frames[p_id]][p_id]
                            frame_diff = start_frames[p_id] - f_id
                            predict_x = first_coord[0] - \
                                frame_diff * velocity[0]
                            predict_y = first_coord[1] - \
                                frame_diff * velocity[1]
                        elif f_id > end_frames[p_id]:
                            velocity = pred_velocity[p_id][-1]
                            last_coord = frame_ped_positions[end_frames[p_id]][p_id]
                            predict_x = last_coord[0] + \
                                frame_diff * velocity[0]
                            predict_y = last_coord[1] + \
                                frame_diff * velocity[1]
                        scene_pre[p_id][f] = [predict_x, predict_y]

            scene = np.empty(
                shape=[timeframe_len, len(scene_pre), 2])

            # always add primary pedestrian's path to the 0 column so that it confroms
            # with original data format
            for f in range(timeframe_len):
                scene[f][0] = scene_pre[primary_id][f]

            # add neighbor's paths to the scene
            ped = 1
            for p_id in scene_pre:
                if p_id == primary_id:
                    continue
                for f in range(timeframe_len):
                    scene[f][ped] = scene_pre[p_id][f]
                ped += 1

            # create a scene for this primary_id
            scenes[primary_id] = scene

        return scenes

    scenes = constructScenes()

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

    for index, primary_id in enumerate(scenes):
        if (index+1) % 50 == 0:
            print(index)

        # divide into a list of 40 frame-scenes
        # (Social Gan requires a 8 second worth of data which in this case
        # will be equivalent to 40 frames since the raw data has a 2.5 fps)
        scene_40f = split_by_size(scenes[primary_id], 40)

        for scene in scene_40f:
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
