from pyvirtualdisplay import Display
#display = Display(visible=False, size=(2560, 1440))
#display.start()

from mayavi import mlab
import mayavi
mlab.options.offscreen = True
print("Set mlab.options.offscreen={}".format(mlab.options.offscreen))

import time, argparse, os.path as osp, os
import torch, numpy as np

import mmcv
from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging import MMLogger
from mmengine.registry import MODELS

import warnings
warnings.filterwarnings("ignore")

def pass_print(*args, **kwargs):
    pass

def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0]) # [0, 1, ..., 256]
    # g_xx = g_xx[::-1]
    g_yy = np.arange(0, dims[1]) # [0, 1, ..., 256]
    # g_yy = g_yy[::-1]
    g_zz = np.arange(0, dims[2]) # [0, 1, ..., 32]

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)
    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])

    coords_grid = (coords_grid * resolution) + resolution / 2

    return coords_grid

def draw(
    voxels,          # semantic occupancy predictions
    pred_pts,        # lidarseg predictions
    vox_origin,
    voxel_size=0.2, #0.2,  # voxel size in the real world
    grid=None,       # voxel coordinates of point cloud
    pt_label=None,   # label of point cloud
    save_dir=None,
    cam_positions=None,
    focal_positions=None,
    timestamp=None,
    mode=0,
    sem=False,
):
    w, h, z = voxels.shape #200 200 16
    
    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    ) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])

    if mode == 0:
        grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
        
        # draw a simple car at the middle
        
        car_vox_range = np.array([
            [w//2 - 2 - 4, w//2 - 2 + 4],
            [h//2 - 2 - 4, h//2 - 2 + 4],
            [z//2 - 2 - 3, z//2 - 2 + 3]
        ], dtype=int)
        car_x = np.arange(car_vox_range[0, 0], car_vox_range[0, 1])
        car_y = np.arange(car_vox_range[1, 0], car_vox_range[1, 1])
        car_z = np.arange(car_vox_range[2, 0], car_vox_range[2, 1])
        
        car_xx, car_yy, car_zz = np.meshgrid(car_x, car_y, car_z)
        car_label = np.zeros([8, 8, 6], dtype=int)
        car_label[:3, :, :2] = 20#17
        car_label[3:6, :, :2] = 18
        car_label[6:, :, :2] = 19
        car_label[:3, :, 2:4] = 18
        car_label[3:6, :, 2:4] = 19
        car_label[6:, :, 2:4] = 20#17
        car_label[:3, :, 4:] = 19
        car_label[3:6, :, 4:] = 20#17
        car_label[6:, :, 4:] = 18
        car_grid = np.array([car_xx.flatten(), car_yy.flatten(), car_zz.flatten()]).T
        car_indexes = car_grid[:, 0] * h * z + car_grid[:, 1] * z + car_grid[:, 2]
        grid_coords[car_indexes, 3] = car_label.flatten()
        
    elif mode == 1:
        indexes = grid[:, 0] * h * z + grid[:, 1] * z + grid[:, 2]
        indexes, pt_index = np.unique(indexes, return_index=True)
        pred_pts = pred_pts[pt_index]
        grid_coords = grid_coords[indexes]
        grid_coords = np.vstack([grid_coords.T, pred_pts.reshape(-1)]).T
    elif mode == 2:
        indexes = grid[:, 0] * h * z + grid[:, 1] * z + grid[:, 2]
        indexes, pt_index = np.unique(indexes, return_index=True)
        gt_label = pt_label[pt_index]
        grid_coords = grid_coords[indexes]
        grid_coords = np.vstack([grid_coords.T, gt_label.reshape(-1)]).T
    else:
        raise NotImplementedError



    grid_coords[grid_coords[:, 3] == 17, 3] = 21
    # Get the voxels inside FOV
    fov_grid_coords = grid_coords

    # Remove empty and unknown voxels
    fov_voxels = fov_grid_coords[
        (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 21)
    ]
    print(len(fov_voxels))
    
    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    voxel_size = sum(voxel_size) / 3
    
    
    
    
    
    
    
    
    car_center_x = (w//2  + w//2) / 2
    car_center_y = (h//2  + h//2 ) / 2
    car_center_z = (z//2 - 2 - 3 + z//2 - 2 + 3) / 2
    print(car_center_x)
    print(car_center_y)
    print(car_center_z)
    car_center_x = car_center_x/10-10
    car_center_y = car_center_y/10-10
    car_center_z = car_center_z/10
    '''
    trajectories = [np.array([[1, 2], [2, 3], [3, 4]]), 
                    np.array([[2, 3], [3, 4], [4, 5]]), 
                    np.array([[3, 4], [4, 5], [5, 6]])]
  
    for traj in trajectories:
        traj_x = car_center_x + traj[:, 0]
        traj_y = car_center_y + traj[:, 1]
        traj_z = np.full_like(traj[:, 0], car_center_z) 
        mlab.points3d(traj_x, traj_z, traj_y, color=(0,0,1), scale_factor=1)
    '''

    
    xy_coords = np.array([
        [0.00333244, 0.00820783],
        [0.00755689, 0.01950348],
        [0.01425542, 0.03879583],
        [0.0283762, 0.08404884],
        [0.04682241, 0.14907806],
        [0.07117051, 0.23827328],
        [0.0965116, 0.3309874],
        [0.12062407, 0.41365507],
        [0.14889929, 0.5034309],
        [0.17333154, 0.57299954],
        [0.19619855, 0.63157153],
        [0.21363494, 0.6717925],
        [0.22401935, 0.69375926],
        [0.22722836, 0.69957423],
        [0.22674589, 0.69808745],
        [0.22665612, 0.69757724],
        [0.22666734, 0.6973573],
        [0.22662808, 0.6970758],
        [0.22993807, 0.7037001],
        [0.23863381, 0.7203005],
        [0.25544748, 0.75014955],
        [0.2763341, 0.78342074],
        [0.31259817, 0.83343303],
        [0.3679761, 0.8953304],
        [0.42739892, 0.9443926],
        [0.5010491, 0.98288935],
        [0.5870024, 1.],
        [0.6824481, 0.9889595],
        [0.78409857, 0.94825464],
        [0.8910507, 0.8779647],
        [1., 0.7873531]
    ])
    
    xy_coords =xy_coords*20
    traj_x = car_center_x + xy_coords[:, 0]
    traj_y = car_center_y + xy_coords[:, 1]
    traj_z = np.full_like(xy_coords[:, 0], car_center_z)  
    
    
    
    angle = np.radians(260)  
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    traj_xy = np.column_stack((traj_x - car_center_x, traj_y - car_center_y))  
    rotated_traj_xy = np.dot(traj_xy, rotation_matrix.T)  
    rotated_traj_x = rotated_traj_xy[:, 0] + car_center_x  
    rotated_traj_y = rotated_traj_xy[:, 1] + car_center_y

    #!!mlab.points3d(rotated_traj_x, rotated_traj_y, traj_z, color=(1, 0, 0), scale_factor=1)

    #mlab.points3d(traj_x, traj_y, traj_z, color=(1, 0, 0), scale_factor=1)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 1],
        fov_voxels[:, 0],
        #fov_voxels[:, 0],
        #fov_voxels[:, 1],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        scale_factor=1.0 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=1,
        vmax=21, # 16
    )

    colors = np.array(
        [
            [255, 120,  50, 255],       # barrier              orange
            [255, 192, 203, 255],       # bicycle              pink
            [255, 255,   0, 255],       # bus                  yellow
            [  0, 150, 245, 255],       # car                  blue
            [  0, 255, 255, 255],       # construction_vehicle cyan
            [255, 127,   0, 255],       # motorcycle           dark orange
            [255,   0,   0, 255],       # pedestrian           red
            [255, 240, 150, 255],       # traffic_cone         light yellow
            [135,  60,   0, 255],       # trailer              brown
            [160,  32, 240, 255],       # truck                purple                
            [255,   0, 255, 255],       # driveable_surface    dark pink
            # [175,   0,  75, 255],       # other_flat           dark red
            [139, 137, 137, 255],
            [ 75,   0,  75, 255],       # sidewalk             dard purple
            [150, 240,  80, 255],       # terrain              light green          
            [230, 230, 250, 255],       # manmade              white
            [  0, 175,   0, 255],       # vegetation           green
            [  0, 255, 127, 255],       # ego car              dark cyan
            [255,  99,  71, 255],       # ego car
            [  0, 191, 255, 255],        # ego car
            [175,   0,  75, 255] 
        ]
    ).astype(np.uint8)
    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors
    '''
    scene = figure.scene
    scene.camera.position = [  0.75131739, -35.08337438,  16.71378558]
    scene.camera.focal_point = [  0.75131739, -34.21734897,  16.21378558]
    scene.camera.view_angle = 40.0
    scene.camera.view_up = [0.0, 0.0, 1.0]
    scene.camera.clipping_range = [0.01, 300.]
    scene.camera.compute_view_plane_normal()
    scene.render()
    '''
 
    
    
    mlab.savefig(os.path.join(save_dir, f'vis_{timestamp}.png'))
    mlab.close()

def main(args):
    # global settings
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir
    
    os.makedirs(args.work_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{cfg.get("data_type", "gts")}_visualize_autoreg_{timestamp}.log')
    logger = MMLogger('genocc', log_file=log_file)
    MMLogger._instance_dict['genocc'] = logger
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    import model
    my_model = MODELS.build(cfg.model)
    my_model.init_weights()
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    my_model = my_model.cuda()
    raw_model = my_model
    logger.info('done ddp model')
    from dataset import get_dataloader
    train_dataset_loader, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_wrapper_config,
        cfg.val_wrapper_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=False)
    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'epoch_164.pth')):#105
        cfg.resume_from = osp.join(args.work_dir, 'epoch_164.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from
    logger.info('resume from: ' + cfg.resume_from)
    logger.info('work dir: ' + args.work_dir)

    epoch = 'last'
    if cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        print(raw_model.load_state_dict(ckpt['state_dict'], strict=False))
        epoch = ckpt['epoch']
        print(f'successfully resumed from epoch {epoch}')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        print(raw_model.load_state_dict(state_dict, strict=False))
        
    # eval
    my_model.eval()
    os.environ['eval'] = 'true'
    recon_dir = os.path.join(args.work_dir, args.dir_name+f'{cfg.get("data_type", "gts")}_autoreg', str(epoch))
    os.makedirs(recon_dir, exist_ok=True)
    dataset = cfg.val_dataset_config['type']
    recon_dir = os.path.join(recon_dir, dataset)
    start_frame = 48
    with torch.no_grad():
        for i_iter_val, (input_occs, target_occs, metas) in enumerate(val_dataset_loader):
            if i_iter_val not in args.scene_idx:
                continue
            if i_iter_val > max(args.scene_idx):
                break
            '''
            if i_iter_val < start_frame:
                continue'''
            input_occs = input_occs.cuda()
            #result = my_model(x=input_occs, metas=metas)
            result = my_model(
                    x=input_occs, metas=metas, 
                    start_frame=cfg.get('start_frame', 0),
                    mid_frame=cfg.get('mid_frame', 5),
                    end_frame=cfg.get('end_frame', 11))
            logits = result['logits']
            n_frames = logits.shape[1]
            dst_dir = os.path.join(recon_dir, str(i_iter_val))
            input_dir = os.path.join(recon_dir, f'{i_iter_val}_input')
            print(result['sem_pred'].shape)
            print(n_frames,input_occs.shape[1])
            #input_occs = result['sem_pred']
            os.makedirs(dst_dir, exist_ok=True)
            os.makedirs(input_dir, exist_ok=True)
            assert n_frames < 33
            for frame in range(n_frames+1):
                input_occ = input_occs[:, frame, ...].squeeze().cpu().numpy()
                draw(input_occ, 
                    None, # predict_pts,
                    [-40, -40, -1], 
                    [0.4] * 3, 
                    None, #  grid.squeeze(0).cpu().numpy(), 
                    None,#  pt_label.squeeze(-1),
                    input_dir,#recon_dir,
                    None, # img_metas[0]['cam_positions'],
                    None, # img_metas[0]['focal_positions'],
                    timestamp=str(i_iter_val) + '_' + str(frame),
                    mode=0,
                    sem=False)
                if frame == n_frames:
                    continue
                logit = logits[:, frame, ...]
                pred = logit.argmax(dim=-1).squeeze().cpu().numpy() # 1, 1, 200, 200, 16
                draw(pred, 
                    None, # predict_pts,
                    [-40, -40, -1], 
                    [0.4] * 3, 
                    None, #  grid.squeeze(0).cpu().numpy(), 
                    None,#  pt_label.squeeze(-1),
                    dst_dir,#recon_dir,
                    None, # img_metas[0]['cam_positions'],
                    None, # img_metas[0]['focal_positions'],
                    timestamp=str(i_iter_val) + '_' + str(frame),
                    mode=0,
                    sem=False)
            logger.info('[EVAL] Iter %5d / %5d'%(i_iter_val, len(val_dataset_loader)))
            logger.info(f'gt_poses_{result["gt_poses_"]}')
            logger.info(f'poses_{result["poses_"]}')
            

if __name__ == '__main__':
    # Eval settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--dir-name', type=str, default='vis')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-trials', type=int, default=10)
    parser.add_argument('--frame-idx', nargs='+', type=int, default=[0, 10])
    parser.add_argument('--scene-idx', nargs='+', type=int, default=[13,16,18,19,87,89,96,101])
    args = parser.parse_args()
    
    ngpus = 1
    args.gpus = ngpus
    print(args)

    main(args)
