from pymo.parsers import BVHParser
from pymo.viz_tools import *
from pymo.preprocessing import *
from matplotlib import pyplot as plt

import bvh
import quat
from train_common import load_database

import numpy as np
import os.path


def vis_skeleton(path, out_path_prefix, frames=(0, 100, 200, 300, 400, 500)):
    parser = BVHParser()
    parsed_data = parser.parse(path)
    parsed_data.skeleton = {k: v for k, v in parsed_data.skeleton.items() if not k.endswith('_Nub')}
    for skeleton in parsed_data.skeleton.values():
        skeleton['children'] = [c for c in skeleton['children'] if not c.endswith('_Nub')]

    mp = MocapParameterizer('position')

    positions = mp.fit_transform([parsed_data])
    joints_to_visualize = [j for j in parsed_data.skeleton.keys() if not j.startswith("joint_0")]

    for frame in frames:
        draw_stickfigure(positions[0], frame=frame, joints=joints_to_visualize)
        plt.savefig(out_path_prefix + str(frame))


def save_original_bvh(path):
    database = load_database('./database.bin')

    parents = database['bone_parents']

    Ypos = database['bone_positions'].astype(np.float32)
    Yrot = database['bone_rotations'].astype(np.float32)
    nbones = Ypos.shape[1]

    try:
        bvh.save(path, {
            'rotations': np.degrees(quat.to_euler(Yrot)),
            'positions': 100.0 * Ypos,
            'offsets': 100.0 * Ypos[0],
            'parents': parents,
            'names': ['joint_%i' % i for i in range(nbones)],
            'order': 'zyx'
        })
    except IOError as e:
        print(e)


if __name__ == '__main__':
    original_path = "/mnt/c/Users/nengn/Documents/Motion-Matching/resources/decompressor_Ygnd.bvh"
    decompressed_path = "/mnt/c/Users/nengn/Documents/Motion-Matching/resources/decompressor_Ytil.bvh"
    # if not os.path.exists(original_path):
    #     save_original_bvh(original_path)

    vis_skeleton(original_path, "/mnt/c/Users/nengn/Documents/thesis/motion_data/original_fk_enabled")
    vis_skeleton(decompressed_path, "/mnt/c/Users/nengn/Documents/thesis/motion_data/decompressed_fk_enabled")
    # vis_skeleton(decompressed_path, "/mnt/c/Users/nengn/Documents/thesis/motion_data/decompressed_fk_disabled")