""" Categorization of Primary Pedestrian """
import pickle
import numpy as np
import data
from interactions import check_interaction, group
from interactions import get_interaction_type


from data_loader import DataLoader as dl
import pandas as pd


def save2npy(input, path):
    if type(input) is dict:
        np.save(path, np.array(list(input.items())))
    elif type(input) is list:
        np.save(path, np.array(input))
    elif type(input) is np.ndarray:
        np.save(path, input)


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
        i_type, i_data = get_interaction_type(scene, args.inter_pos_range,
                                              args.inter_dist_thresh, args.obs_len)
    else:
        i_type = []
        i_data = {}

    return mult_tag[0], mult_tag, i_type, i_data


def trajectory_type(track_id=0, args=None):
    """ Categorization of all scenes """

    def split_by_size(arr, size):
        res = np.split(arr, np.arange(size, len(arr), size))
        # discard elements that has less than (size)'s length
        del_row = []
        for (i, e) in enumerate(res):
            if len(e) < size:
                del_row.append(i)
        return np.delete(res, del_row, axis=0)

    # Construct scene for each of pedestrian (360 in total for eth dataset)
    # Each scene is created over the duration of each pedestrian's apearance - from
    # its start frame to end frame, and the scene contains the paths for all people
    # in that time window
    def construct_scenes(args):
        # use data_loader here
        dataloader = dl(path=args.read_path)

        fps = args.fps
        frame_ped_positions = dataloader.frameId_people_positions
        ped_id = dataloader.personIdListSorted
        start_frames = dataloader.people_start_frame
        end_frames = dataloader.people_end_frame
        ped_velocity = dataloader.people_velocity_complete
        total_ped_num = 0

        scenes = {}
        # create scene for each primary pedestrian
        for i in range(len(ped_id)):
            primary_id = ped_id[i]
            start = start_frames[primary_id]
            end = end_frames[primary_id]

            # collection of pedestrians that appears in this timewindow
            ped_collection = set()

            timewindow_len = end - start + 1
            for tf in range(timewindow_len):
                frame = start + tf
                ped_collection.update(
                    set(frame_ped_positions[frame].keys()))

            # number of all pedestrians that appears in the timewindow
            ped_num = len(ped_collection)

            # keep track of maximum number of pedestrians that appears in all timewindows.
            # The purpose of this value is to create a dataset
            if ped_num > total_ped_num:
                total_ped_num = ped_num


            # collect all pedestrian's positions from the begining until the end
            scene_pre = {}
            for p_id in ped_collection:
                scene_pre[p_id] = [np.NaN] * timewindow_len

            for f in range(timewindow_len):
                f_id = start + f

                for p_id in scene_pre:
                    if p_id in frame_ped_positions[f_id].keys():
                        scene_pre[p_id][f] = frame_ped_positions[f_id][p_id]
                    else:
                        # Replace NaNs with interpolated positions.
                        # We will use the pedestrians id's start & end frame
                        # to calculate this.

                        # If the frame in this for-loop iteration is before the start
                        # frame then use the first velocity, else if the
                        # frame goes beyond the end frame use the last velocity.
                        # Note that velocity is converted from m/s into m/frame
                        if f_id < start_frames[p_id]:
                            vel_mpf = [
                                vel / fps for vel in ped_velocity[p_id][0]]
                            first_coord = frame_ped_positions[start_frames[p_id]][p_id]
                            frame_diff = start_frames[p_id] - f_id
                            predict_x = first_coord[0] - \
                                frame_diff * vel_mpf[0]
                            predict_y = first_coord[1] - \
                                frame_diff * vel_mpf[1]
                        elif f_id > end_frames[p_id]:
                            vel_mpf = [
                                vel / fps for vel in ped_velocity[p_id][-1]]
                            last_coord = frame_ped_positions[end_frames[p_id]][p_id]
                            predict_x = last_coord[0] + \
                                frame_diff * vel_mpf[0]
                            predict_y = last_coord[1] + \
                                frame_diff * vel_mpf[1]
                        scene_pre[p_id][f] = [predict_x, predict_y]

            scene = np.empty(
                shape=[timewindow_len, len(scene_pre), 2])

            # always place primary pedestrian's path in the 0th column so that it confroms
            # with the trajnetplusplus's original dataset format
            for f in range(timewindow_len):
                scene[f][0] = scene_pre[primary_id][f]

            # add primary pedestrian's neighbor's path
            next_ped = 1
            for p_id in scene_pre:
                if p_id == primary_id:
                    continue
                for f in range(timewindow_len):
                    scene[f][next_ped] = scene_pre[p_id][f]
                next_ped += 1

            # generate a scene for this primary_id
            scenes[primary_id] = scene

        return scenes, total_ped_num


    # scenes, scenes_rel, total_ped = construct_scenes(args)
    scenes, total_ped = construct_scenes(args)

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

    computed_scenes = []
    computed_scenes_rel = []
    result_interaction_yn = []
    result_interaction_type = []

    for index, primary_id in enumerate(scenes):
        if (index+1) % 50 == 0:
            print(index)

        # divide into a list of 16-frame scenes
        # (the default observed length for Social Gan is 8 but 8-frame window 
        # doesn't detect any interactions)
        sgan_fsize = 16
        scene_list = split_by_size(scenes[primary_id], sgan_fsize)

        for scene in scene_list:

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

            # Note: As we're collecting data we will transfer the scene and
            # scene_rel data to a fixed-size numpy array
            
            # collect scenes 
            tmp_array_scene = np.empty((sgan_fsize, total_ped, 2))  
            tmp_array_scene[:, :scene.shape[1], :] = scene[:, :, :]
            computed_scenes.append(tmp_array_scene)
            
            # collect scenes_rel (collection of position delta between frames)
            tmp_array_scene_rel = np.empty((sgan_fsize, total_ped, 2))
            scene_rel = scene[1:, :, :] - scene[:-1, :, :] 
            tmp_array_scene_rel[:scene_rel.shape[0], :scene_rel.shape[1], :] = scene_rel[:, :, :]
            tmp_array_scene_rel[scene_rel.shape[0], :scene_rel.shape[1], :] = scene_rel[-1, :, :]   # this line is perfomred to fill in the empty element with filler data
            computed_scenes_rel.append(tmp_array_scene_rel)           
            
            # collect interaction data
            result_interaction_yn.append([1 if tag == 3 else 0])
            result_interaction_type.append(sub_tag)

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

    # save all data to local file
    path = args.save_path if args.save_path is not None else ''
    save2npy(np.array(computed_scenes), str(path) + 'scenes')
    save2npy(np.array(computed_scenes_rel), str(path) + 'scenes_rel')     
    save2npy(np.array(result_interaction_yn),
             str(path) + 'result_interaction_yn')
    save2npy(np.array(result_interaction_type),
             str(path) + 'result_interaction_type')   

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

        print("Number of scenes saved to file: ", len(computed_scenes))
        print("Number of results saved to file: ", len(result_interaction_yn))

    return track_id
