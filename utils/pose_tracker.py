import os
import json
import shutil
import subprocess
import numpy as np
import os.path as osp

def run_openpose(video_file, output_folder, staf_folder, vis=False):
    """
    Run OpenPose on a video file to extract pose keypoints.

    Args:
    - video_file (str): Path to the input video file.
    - output_folder (str): Path to the folder where OpenPose JSON outputs will be saved.
    - staf_folder (str): Path to the OpenPose installation directory.
    - vis (bool, optional): Whether to enable visualization during OpenPose execution. Default is False.

    Returns:
    - None
    """
    pwd = os.getcwd()  # Get current working directory
    os.chdir(staf_folder)  # Change directory to where OpenPose binaries are located

    render = 1 if vis else 0  # Set render mode based on vis flag
    display = 2 if vis else 0  # Set display mode based on vis flag
    cmd = [
        'build/samples/openpose/openpose.bin',
        '--model_pose', 'BODY_21A',
        '--tracking', '1',
        '--render_pose', str(render),
        '--video', video_file,
        '--write_json', output_folder,
        '--display', str(display)
    ]

    print('Executing', ' '.join(cmd))  # Print the command being executed
    subprocess.run(cmd)  # Run OpenPose command
    os.chdir(pwd)  # Restore original working directory

def read_posetrack_keypoints(output_folder):
    """
    Read pose keypoints from OpenPose JSON outputs.

    Args:
    - output_folder (str): Path to the folder containing OpenPose JSON outputs.

    Returns:
    - people (dict): Dictionary containing extracted pose keypoints and associated frames.
    """
    people = dict()  # Initialize dictionary to store people's keypoints

    # Iterate through all JSON files in the output folder
    for idx, result_file in enumerate(sorted(os.listdir(output_folder))):
        json_file = os.path.join(output_folder, result_file)
        data = json.load(open(json_file))  # Load JSON data from file

        # Process each person detected in the JSON data
        for person in data['people']:
            person_id = person['person_id'][0]  # Get person ID
            joints2d = person['pose_keypoints_2d']  # Get 2D pose keypoints

            # If person ID exists in dictionary, append keypoints and frame number
            if person_id in people.keys():
                people[person_id]['joints2d'].append(joints2d)
                people[person_id]['frames'].append(idx)
            else:
                # If person ID does not exist, initialize new entry
                people[person_id] = {
                    'joints2d': [],
                    'frames': [],
                }
                people[person_id]['joints2d'].append(joints2d)
                people[person_id]['frames'].append(idx)

    # Convert lists to numpy arrays for efficiency
    for k in people.keys():
        people[k]['joints2d'] = np.array(people[k]['joints2d']).reshape((len(people[k]['joints2d']), -1, 3))
        people[k]['frames'] = np.array(people[k]['frames'])

    return people  # Return dictionary containing extracted keypoints and frames

def run_posetracker(video_file, staf_folder, posetrack_output_folder='/home/jd/tmp', display=False):
    """
    Run PoseTracker pipeline on a video file.

    Args:
    - video_file (str): Path to the input video file.
    - staf_folder (str): Path to the OpenPose installation directory.
    - posetrack_output_folder (str, optional): Path to the folder where PoseTrack outputs will be temporarily stored. Default is '/home/jd/tmp'.
    - display (bool, optional): Whether to enable visualization during PoseTracker execution. Default is False.

    Returns:
    - people_dict (dict): Dictionary containing extracted pose keypoints and associated frames.
    """
    posetrack_output_folder = os.path.join(
        posetrack_output_folder,
        f'{os.path.basename(video_file)[:-4]}_posetrack'
    )  # Construct output folder path for PoseTrack outputs

    # Run OpenPose to extract pose keypoints
    run_openpose(
        video_file,
        posetrack_output_folder,
        vis=display,
        staf_folder=staf_folder
    )

    # Read pose keypoints from OpenPose outputs
    people_dict = read_posetrack_keypoints(posetrack_output_folder)

    # Clean up temporary PoseTrack output folder
    shutil.rmtree(posetrack_output_folder)

    return people_dict  # Return dictionary containing extracted keypoints and frames
