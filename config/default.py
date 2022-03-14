from yacs.config import CfgNode as CN

_C = CN()

_C.mode = 'train'
_C.max_epochs = 10
_C.batch_size = 4096
_C.num_workers = 8
_C.learning_rate = 1e-4
_C.logdir = 'logs/'
_C.log_img_freq_iter = 100
_C.num_log_img = 10
_C.log_img_scale = 2
_C.log_img_batch_count = 1
_C.log_mesh_freq_epoch = 1
_C.resume = False
_C.load_ckpt_path = ''

# Dataset
_C.dataset = CN()
_C.dataset.dataset_path = 'test_datasets/tum_rgbd/'
_C.dataset.scene_name = 'rgbd_dataset_freiburg1_desk'
_C.dataset.association_file_name = 'data_association_file.txt'
# _C.dataset.frame_indices = [23, 60, 131, 206, 257, 325, 368, 407, 455, 558]
_C.dataset.frame_indices = 30
_C.dataset.distance_koef = 0.1
_C.dataset.clip_distance_threshold = 4.0

# Model
_C.model = CN()
_C.model.mlp_color = False
_C.model.mlp_num_layers = 4
_C.model.mlp_color_num_layers = 4
_C.model.hidden_features = 256
_C.model.encoding_dimension = 93
_C.model.coarse_sample_bins = 320
_C.model.fine_sample_bins = 16
_C.model.encoding_sigma = 25
_C.model.minimal_depth = 0.01
_C.model.sc_factor = 1.
_C.model.mode = 'sdf'
_C.model.volsdf_ab_fixed = False
_C.model.alpha = 1.
_C.model.beta = 0.1
_C.model.model_type = 'sine'
_C.model.pos_encoding = True
_C.model.truncation = 0.05
_C.model.loss_type = 'l2'
_C.model.depth_loss_koef = 1.
_C.model.color_loss_koef = 5.
_C.model.sdf_loss_koef = 6e4
_C.model.fs_loss_koef = 1e2
_C.model.grad_loss_koef = 1e2
_C.model.upsample = 'default'
_C.model.variance_init = 0.05
_C.model.speed_factor = 1.0


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)

    cfg.freeze()


def check_config(cfg):
    pass
