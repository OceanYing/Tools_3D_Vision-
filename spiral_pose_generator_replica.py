"""
spiral pose generator
Edited by haiyang, 20221226
Based on LLFF spiral pose generator
"""
import numpy as np
import os
import pdb

def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, down, pos):
    vec2 = normalize(z)
    vec1_avg = down
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def poses_avg(poses):
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    down = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, down, center), hwf], 1)
    return c2w


def render_path_spiral(c2w, down, rads, focal, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:, 4:5]

    for theta in np.linspace(0., 2. * np.pi * rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(np.dot(c2w[:3, :4], np.array([0, 0, +focal, 1.])) - c)    # z-axis
        render_poses.append(np.concatenate([viewmatrix(z, down, c), hwf], 1))
    return render_poses


def read_cameras_scannet(datadir, frame_ids, crop=10):
    frame_len = len(frame_ids)  # np.arange(0, 600)
    rgb_files = []
    c2w_mats = []
    W = 640
    H = 480
    W_rawimg = 1296
    H_rawimg = 968

    # pdb.set_trace()
    intrinsics = np.loadtxt(os.path.join(datadir, "intrinsic/intrinsic_color.txt"), delimiter=' ')
    intrinsics[0, :] = intrinsics[0, :] * (W / W_rawimg)
    intrinsics[1, :] = intrinsics[1, :] * (H / H_rawimg)
    intrinsics[0, 2] = intrinsics[0, 2] - crop
    intrinsics[1, 2] = intrinsics[1, 2] - crop

    for i in frame_ids:
        rgb_file = os.path.join(datadir, "color", "%d.jpg"%(i))
        rgb_files.append(rgb_file)
        c2w_opencv = np.loadtxt(os.path.join(datadir, "pose", "%d.txt"%(i)))
        c2w_mats.append(c2w_opencv)
    c2w_mats = np.array(c2w_mats)
    return rgb_files, np.array([intrinsics]*frame_len), c2w_mats


def free_view_pose_generator(poses, Nviews=60, radii=None, farpoint=3.0):
    c2w = poses_avg(poses)

    ## Get spiral
    # Get average pose
    down = normalize(poses[:, :3, 1].sum(0))    # mean y-axis norm vector

    # Find a reasonable "focus depth" for this dataset
    close_depth, inf_depth = 0.2, 5.0
    # close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
    # dt = .75
    # mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
    # focal = mean_dz
    focal = 3.0 if farpoint is None else farpoint      # meter(m) : poses_avg to scene_center

    # Get radii for spiral path
    if radii is not None:
        rads = radii
    else:
        # tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
        # rads = np.percentile(np.abs(tt), 90, 0)
        # rads = np.array([0.5, 0.3, 0])  # setup render circle size (xc, yc, zc)
        rads = np.array([1.0, 0.6, 0])  # setup render circle size (xc, yc, zc)

    c2w_path = c2w
    N_views = Nviews
    N_rots = 1

    # Generate poses for spiral path
    render_poses = render_path_spiral(c2w_path, down, rads, focal, zrate=.5, rots=N_rots, N=N_views)

    render_poses = np.array(render_poses).astype(np.float32)
    mat_eyes = np.repeat(np.eye(4)[None, ...], repeats=N_views, axis=0)
    mat_eyes[:, :3, :] = render_poses[:, :, :4]
    
    return mat_eyes


def pcwrite(filename, xyzrgb):
  """Save a point cloud to a polygon .ply file.
  """
  xyz = xyzrgb[:, :3]
  rgb = xyzrgb[:, 3:].astype(np.uint8)

  # Write header
  ply_file = open(filename,'w')
  ply_file.write("ply\n")
  ply_file.write("format ascii 1.0\n")
  ply_file.write("element vertex %d\n"%(xyz.shape[0]))
  ply_file.write("property float x\n")
  ply_file.write("property float y\n")
  ply_file.write("property float z\n")
  ply_file.write("property uchar red\n")
  ply_file.write("property uchar green\n")
  ply_file.write("property uchar blue\n")
  ply_file.write("end_header\n")

  # Write vertex list
  for i in range(xyz.shape[0]):
    ply_file.write("%f %f %f %d %d %d\n"%(
      xyz[i, 0], xyz[i, 1], xyz[i, 2],
      rgb[i, 0], rgb[i, 1], rgb[i, 2],
    ))


def render_pose_axis(poses, unit_length=0.1):
    """
    poses: c2w matrix in opencv manner
    unit_length (m:meter): unit axis length for visualization
    axis-x, axis-y, axis-z: red, green, blue
    """
    pose_coord_x = poses[:, :3, -1] + poses[:, :3, 0] * unit_length
    pose_coord_y = poses[:, :3, -1] + poses[:, :3, 1] * unit_length
    pose_coord_z = poses[:, :3, -1] + poses[:, :3, 2] * unit_length
    poses_vis = np.concatenate([poses[:, :3, -1], pose_coord_x, pose_coord_y, pose_coord_z], axis=0)
    poses_rgb = np.concatenate([np.ones([poses.shape[0], 3])*255,
                                np.ones([poses.shape[0], 3])*np.array([255, 0, 0]),
                                np.ones([poses.shape[0], 3])*np.array([0, 255, 0]),
                                np.ones([poses.shape[0], 3])*np.array([0, 0, 255]),
                                ])
    # pcwrite("camera_raw_axis.ply", np.concatenate([poses_vis, poses_rgb], axis=1))
    return poses_vis, poses_rgb


if __name__ == "__main__":
    # folder_path = "/data/haiyang/Experiments/data/scannet/scannetTrainSeq"
    # scene_path = os.path.join(folder_path, "scene0000_00")
    # cam_idx_list = np.arange(0, 600)
    # img_crop = 10

    # rgb_files, intrinsics, poses = read_cameras_scannet(datadir=scene_path, frame_ids=cam_idx_list, crop=img_crop)

    # poses = np.loadtxt("/Users/ocean/Documents/TBSI项目/SLAM/RadianceFusion/data/scalable/livingroom/camera.txt").reshape(-1, 4, 4)



    basedir = "/Users/ocean/Documents/TBSI项目/SLAM/RadianceFusion/0106_NeRF-SLAM/replica_sample/"
    posedir = os.path.join(basedir, "office0/save_traj_gt.txt")  
    poses = np.loadtxt(posedir).reshape(-1, 4, 4)

    # free_poses = free_view_pose_generator(poses)
    # free_poses = np.concatenate([
    #         free_view_pose_generator(poses, Nviews=60, radii=np.array([0.5, 0.3, 0])),
    #         free_view_pose_generator(poses, Nviews=60, radii=np.array([1.0, 0.6, 0]))
    #         ], axis=0)

    pcwrite("camera_raw_gt.ply", np.concatenate([poses[:, :3, -1], np.ones([poses.shape[0], 3])*255], axis=1))
    # pcwrite("camera_spiral.ply", np.concatenate([free_poses[:, :3, -1], np.zeros([free_poses.shape[0], 3])], axis=1))

    poses_vis, poses_rgb= render_pose_axis(poses, unit_length=0.05)
    # free_poses_vis, free_poses_rgb= render_pose_axis(free_poses, unit_length=0.05)
    pcwrite("camera_raw_axis_gt.ply", np.concatenate([poses_vis, poses_rgb], axis=1))
    # pcwrite("camera_spiral_axis.ply", np.concatenate([free_poses_vis, free_poses_rgb], axis=1))
    