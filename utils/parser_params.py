from torch import cuda


def get_args():
    import argparse
    import os
    import torch
    import json
    parser = argparse.ArgumentParser(description='Welcome to the MAML++ training and inference system')

    parser.add_argument('--batch_size', nargs="?", type=int, default=32, help='Batch_size for experiment')
    parser.add_argument('--image_channels', nargs="?", type=int, default=3)
    parser.add_argument('--reset_stored_filepaths', type=str, default="False")
    parser.add_argument('--reverse_channels', type=str, default="False")
    parser.add_argument('--num_of_gpus', type=int, default=8)
    parser.add_argument('--indexes_of_folders_indicating_class', nargs='+', default=[-2, -3])
    parser.add_argument('--samples_per_iter', nargs="?", type=int, default=1)
    parser.add_argument('--labels_as_int', type=str, default="False")
    parser.add_argument('--seed', type=int, default=104)

    parser.add_argument('--gpu_to_use', type=int)
    parser.add_argument('--num_dataprovider_workers', nargs="?", type=int, default=4)
    parser.add_argument('--max_models_to_save', nargs="?", type=int, default=5)
    parser.add_argument('--reset_stored_paths', type=str, default="False")

    parser.add_argument('--experiment_name', nargs="?", type=str, )
    parser.add_argument('--architecture_name', nargs="?", type=str)
    parser.add_argument('--continue_from_epoch', nargs="?", type=str, default='latest', help='Continue from checkpoint of epoch')
    parser.add_argument('--dropout_rate_value', type=float, default=0.3, help='Dropout_rate_value')
    parser.add_argument('--num_target_samples', type=int, default=15, help='Dropout_rate_value')
    parser.add_argument('--second_order', type=str, default="False", help='Dropout_rate_value')
    parser.add_argument('--total_epochs', type=int, default=200, help='Number of epochs per experiment')
    parser.add_argument('--total_iter_per_epoch', type=int, default=500, help='Number of iters per epoch')
    parser.add_argument('--min_learning_rate', type=float, default=0.0001, help='Min learning rate')
    parser.add_argument('--meta_opt_bn', type=str, default="False")
    parser.add_argument('--task_learning_rate', type=float, default=0.0005, help='Learning rate per task gradient step')

    parser.add_argument('--norm_layer', type=str, default="batch_norm")

    parser.add_argument('--max_pooling', type=str, default="False")
    parser.add_argument('--per_step_bn_statistics', type=str, default="False")
    parser.add_argument('--num_classes_per_set', type=int, default=2, help='Number of classes to sample per set')
    parser.add_argument('--cnn_num_blocks', type=int, default=4, help='Number of classes to sample per set')
    parser.add_argument('--number_of_training_steps_per_iter', type=int, default=1, help='Number of classes to sample per set')
    parser.add_argument('--number_of_evaluation_steps_per_iter', type=int, default=1, help='Number of classes to sample per set')
    parser.add_argument('--cnn_num_filters', type=int, default=64, help='Number of classes to sample per set')
    parser.add_argument('--cnn_blocks_per_stage', type=int, default=1,
                        help='Number of classes to sample per set')
    parser.add_argument('--num_samples_per_class', type=int, default=1, help='Number of samples per set to sample')
    parser.add_argument('--name_of_args_json_file', type=str, default="None")

    parser.add_argument('--conv_padding', type=str, default='true')
    parser.add_argument('--num_stages', type=int, default=19)
    parser.add_argument('--learnable_bn_beta', type=str, default='true')
    parser.add_argument('--learnable_bn_gamma', type=str, default='true')
    parser.add_argument('--enable_inner_loop_optimizable_bn_params', type=str, default='true')
    parser.add_argument('--learnable_per_layer_per_step_inner_loop_learning_rate', type=str, default='true')
    parser.add_argument('--multi_step_loss_num_epochs', type=int, default=10)
    ###################
    ###################
    parser.add_argument("--master_addr", default=None, type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    dset = ['FF-DF', 'FF-NT', 'FF-F2F', 'FF-FS', 'ALL']
    parser.add_argument('--meta', default="init", type=str, help='the feature space')
    parser.add_argument('--dset', type=str, choices=dset, help='method in FF++')
    parser.add_argument('--train_batchSize', type=int, default=4, help='input batch size')
    parser.add_argument('--eval_batchSize', type=int, default=32, help='eval batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
    parser.add_argument('--resolution', type=int, default=256, help='the resolution of the output image to network')
    parser.add_argument('--test_batchSize', type=int, default=32, help='test batch size')
    parser.add_argument('--dataname', type=str, default=None, help='dataname')

    # setting
    parser.add_argument('--save_epoch', type=int, default=1, help='the interval epochs for saving models')
    parser.add_argument('--rec_iter', type=int, default=100, help='the interval iterations for recording')

    # trainning config
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="Momentum, Default: 0.0005")

    parser.add_argument("--nEpochs", type=int, default=30,help="number of epochs to train for")
    parser.add_argument("--start_epoch", default=0, type=int, help="Manual epoch number (useful on restarts)")

    # for distributed parallel
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=8, type=int, help='number of gpus per node')
    parser.add_argument('-nr', '--node_rank', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('-mp', '--master_port', default='5555', type=str, help='ranking within the nodes')
    parser.add_argument('--ngpu', type=int, default=8, help='number of GPUs to use')
    parser.add_argument('--logdir', default='/apdcephfs/private_liamclchen/meta/logs', help='folder to output images')
    parser.add_argument('--savedir', default='/apdcephfs/private_liamclchen/meta/logs', help='folder to output images')
    parser.add_argument('--log_embedding', action='store_true', help='log embedding projection')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument("--pretrained", default=None, type=str, help="path to pretrained model (default: none)")
    parser.add_argument('--cuda', default=True, action='store_true', help='enable cuda')

    args = parser.parse_args()
    args_dict = vars(args)
    if args.name_of_args_json_file is not "None":
        args_dict = extract_args_from_json(args.name_of_args_json_file, args_dict)

    for key in list(args_dict.keys()):

        if str(args_dict[key]).lower() == "true":
            args_dict[key] = True
        elif str(args_dict[key]).lower() == "false":
            args_dict[key] = False
        if key == "dataset_path":
            args_dict[key] = os.path.join(os.environ['DATASET_DIR'], args_dict[key])
            print(key, os.path.join(os.environ['DATASET_DIR'], args_dict[key]))

        print(key, args_dict[key], type(args_dict[key]))

    args = Bunch(args_dict)


    args.use_cuda = torch.cuda.is_available()

    return args



class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)

def extract_args_from_json(json_file_path, args_dict):
    import json
    summary_filename = json_file_path
    with open(summary_filename) as f:
        summary_dict = json.load(fp=f)

    for key in summary_dict.keys():
        if "continue_from" in key:
            pass
        elif "gpu_to_use" in key:
            pass
        else:
            args_dict[key] = summary_dict[key]

    return args_dict





