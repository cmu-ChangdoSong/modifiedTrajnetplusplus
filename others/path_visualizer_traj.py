# [Introduction]
# The purpose of this script is to visualize the EWAP DATASET to better understand
# and validate the Trajnet++ interaction labeling. This is written for course deliverables
# purposes only (course: Independent Study - summer 2021 @ CMU/The Robotics Institute/TBD Lab)
#
# [Details]
# Pedestrian's navigational trajectoy has been drawn on top of the original footage
# and it is color-coded with specific interaction types.
# For example, if a trajectory of pedestrian A is marked with having
# an interaction type - "leader follower", the trajectory line will be color-coded with
# the corresponding color.
#
# [Note]
# There are 2 views that you can toggle between by using the spacebar key.
# First is the normal view that displays the high-level interaction types such as follows:
#   1: static
#   2: linear
#   3: non-linear
#   4: others
# Second is the interaction view that displays the lower-level of interaction subtypes:
#   1: leader follower
#   2: collision avoidance
#   3: group
#   4: others
#
# Also keep in mind that you require a full resources of EWAP DATASET. This includes original
# footage, homography file, trjectory file etc.

import numpy as np
import cv2 as cv
import random
import json


eth_postfix = 'eth'
hotel_postfix = 'hotel'

biwi_url = '/home/cds/Documents/summer2021/IndependentStudy/code/ewap_dataset/'
# select one of the following datasets: 'eth' or 'hotel'
select = 'eth'
# select = 'hotel'
timestep = 80

biwi_postfix = eth_postfix if select == 'eth' else hotel_postfix
biwi_url = biwi_url + 'seq_' + biwi_postfix + '/'
print("biwi URL = ", biwi_url)

# trajectory line settings
line_width = 2

# colors in BGR
colorMap = {'red': [0, 0, 255],
            'green': [0, 255, 0],
            'blue': [255, 0, 0],
            'yellow': [7, 255, 223],
            'dimgrey': [87, 85, 83],
            'grey': [115, 108, 103],
            'purple': [184, 35, 55],
            'pink': [170, 9, 214],
            'black': [0, 0, 0],
            'orange': [0, 145, 255],
            'cyan': [233, 250, 0]}

# import homography
H_data = np.loadtxt(biwi_url + "H.txt", dtype=float)
print("Homography = ", H_data)

# get inverse homography
Inv_H = np.linalg.inv(H_data)

# read from file
lines = []
with open(biwi_url + '/import/biwi_' + biwi_postfix + '.ndjson') as f:
    lines = f.readlines()

p_info = {}
t_info = {}
for line in lines:
    d = json.loads(line)

    if 'scene' in d:
        info = d['scene']
        tag = info['tag']
        p = info['p']
        s = info['s']
        e = info['e']
        id = info['id']
        if p in p_info:
            p_info[p].append({'s': s, 'e': e, 'tag': tag, 'id': id})
        else:
            p_info[p] = [{'s': s, 'e': e, 'tag': tag, 'id': id}]
    elif 'track' in d:
        info = d['track']
        f = info['f']
        p = info['p']
        x = info['x']
        y = info['y']
        if f not in t_info:
            t_info[f] = {}

        t_info[f][p] = {'x': x, 'y': y}


# <Interaction types>
# 1: static
# 2: linear
# 3: non-linear
    # 1: leader follower
    # 2: collision avoidance
    # 3: group
    # 4: others
# 4: others
typeColor = {1: colorMap['green'],   # static
             2: colorMap['black'],  # linear
             3: colorMap['red'],    # non-linear
             4: colorMap['dimgrey']}   # others
subtypeColor = {1: colorMap['pink'],    # leader follower
                2: colorMap['yellow'],  # collision avoidance
                3: colorMap['cyan'],    # group
                4: colorMap['grey']}    # others

# A dedicated canvas will be created for each pedestrian of interest (pedestrain that has a recorded 'type' data)
canvas = {}  # view1
interactionCanvas = {}  # view2

vid_path = biwi_url + 'seq_' + biwi_postfix + '.avi'
cap = cv.VideoCapture(vid_path)
videFrameCnt = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
print("Frame Count = ", videFrameCnt)

prevWarppedPos = {}
f_index = 0
showOnlyInteraction = False

# read frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # if frame data does NOT exist for this frame id, continue
    if f_index not in t_info:
        f_index = f_index + 1
        continue

    # (input)toggle between showing all trajectories or just the trajectories with interactions
    input = cv.waitKey(timestep)
    if input == ord(' '):
        showOnlyInteraction = not showOnlyInteraction
    elif input == ord('q'):
        break

    p_position_list = t_info[f_index]

    pidOutOfFrame = set(canvas) - set(p_position_list)
    for id in pidOutOfFrame:
        remove_message = canvas.pop(
            id, ('No canvas found for pid = ' + str(id)))
        print(remove_message)

    pidOutOfFrame = set(interactionCanvas) - set(p_position_list)
    for id in pidOutOfFrame:
        remove_message = interactionCanvas.pop(
            id, ('No interaction-canvas found for pid = ' + str(id)))
        print(remove_message)

    for key, val in p_position_list.items():
        pid = key   # pedestrian id
        pos = val   # pedestrian position for this frame id

        if pid not in p_info:
            continue    # continue if there is no interaction data for this pedestrian id

        tag = None  # pedestrian interaction type
        info = p_info[pid]  # pedestrian info (contains type data)
        for taginfo in info:  # check if current frame has tag info
            startFrame = taginfo['s']
            endFrame = taginfo['e']
            sceneid = taginfo['id']

            if f_index >= startFrame and f_index <= endFrame:
                tag = taginfo['tag']

        if tag == None:  # no more recorded data for this pid
            continue

        # NOW draw line
        if pid not in canvas:
            canvas[pid] = np.zeros_like(frame)  # create canvas for this pid

        if pid not in interactionCanvas:
            interactionCanvas[pid] = np.zeros_like(frame)

        warppedXY = Inv_H @ [pos['x'], pos['y'], 1]
        warppedXY = warppedXY / warppedXY[2]

        # the raw position data seems to have x and y switched hence the following code will reverse it
        y = warppedXY[0]
        x = warppedXY[1]

        x_prev = x if (pid not in prevWarppedPos.keys()
                       ) else prevWarppedPos[pid]['x']
        y_prev = y if (pid not in prevWarppedPos.keys()
                       ) else prevWarppedPos[pid]['y']

        tag_type = tag[0]
        # NOTE: an interaction can be defined by multiple subtypes of interactions
        tag_subtype = tag[1]
        canvas[pid] = cv.line(canvas[pid], (int(x), int(y)), (int(x_prev), int(y_prev)),
                              typeColor[tag_type], line_width)

        if tag_type == 3:
            # interactionCanvas[pid] = cv.line(interactionCanvas[pid], (int(x), int(y)), (int(x_prev), int(y_prev)),
            #                                  subtypeColor[tag_subtype[0]], line_width)
            lineOffset = 0
            for t in tag_subtype:
                interactionCanvas[pid] = cv.line(interactionCanvas[pid],
                                                 (int(x + lineOffset),
                                                  int(y + lineOffset)),
                                                 (int(x_prev + lineOffset),
                                                  int(y_prev + lineOffset)),
                                                 subtypeColor[t], line_width)
                lineOffset += (line_width + 1)

        prevWarppedPos[pid] = {'x': x, 'y': y}

    if showOnlyInteraction:
        for id, pid_canvas in interactionCanvas.items():
            frame = cv.add(frame, pid_canvas)
    else:
        for id, pid_canvas in canvas.items():
            frame = cv.add(frame, pid_canvas)

    cv.imshow('frame', frame)
    f_index = f_index + 1

cap.release()
cv.destroyAllWindows()
