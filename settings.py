import os

taskname = "supermodel_random_100"
local_root_dir = "/home/xuhang/han/nips/" # root working directory
local_data_dir = "/home/xuhang/han/nips/data" # data root
results_dir = "trained_results"
trained_pickle_file = "trained_models.pkl"
trained_csv_file = "trained_models.csv"
logfile = 'BOGCN_open_domain.log'
io_config = dict(
    trained_pickle_file=os.path.join(local_root_dir, results_dir, taskname, trained_pickle_file),
    trained_csv_file=os.path.join(local_root_dir, results_dir, taskname, trained_csv_file),
)
search_config = dict(
    gcn_epochs=100,
    gcn_lr=0.001,
    loss_num=3,
    generate_num=100,
    iterations=0,
    bo_sample_num=100,
    sample_method="random",
    if_init_samples=True,
    init_num=100,
)

training_config = dict(
    train_supernet_epochs=1,
    data_path=os.path.join(local_data_dir, 'data'),
    super_batch_size=64,
    sub_batch_size=128,
    learning_rate=0.025,
    momentum=0.9,
    weight_decay=3e-4,
    report_freq=50,
    epochs=1,
    init_channels=36,
    layers=20,
    drop_path_prob=0.2,
    seed=0,
    grad_clip=5,
    parallel=False,
    mode='random'
)

distributed = False

#OPS to allow in the search space
OPS = ['input', 'max_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'dil_conv_3x3', 'output']
