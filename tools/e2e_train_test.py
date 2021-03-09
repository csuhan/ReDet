import os
import os.path as osp


def mkdir_if_not_exists(path):
    if not osp.exists(path):
        os.mkdir(path)


def e2e_train_test(config_dir, confg_list, args=None):
    """End to end train test script for slurm
    """
    mkdir_if_not_exists('e2e_train_test')
    if args is None:
        args = ["" for config in confg_list]
    for arg, config in zip(args, confg_list):
        launch_file = 'e2e_train_test/train_{}.sh'.format(config)
        with open(launch_file, 'w') as f_in:
            f_in.write(r'#!/usr/bin/env bash' + '\n')
            f_in.write(r'module load scl/gcc4.9' + '\n')
            f_in.write(r'module load nvidia/cuda/10.0' + '\n')
            f_in.write(
                '{} ./tools/dist_train.sh {}/{}.py 4\n'.format(
                    arg, config_dir, config))
            f_in.write(
                '{} ./tools/dist_test.sh {}/{}.py work_dirs/{}/latest.pth 4 --out work_dirs/{}/results.pkl\n'.format(
                    arg, config_dir, config, config, config))
            f_in.write('python tools/parse_results.py --config {}/{}.py --type OBB\n'.format(config_dir, config))
        # launch file
        status = os.system(
            'sbatch -A gsxia -p gpu --gres=gpu:4 -c 16 -o train_{}.log {}'.format(config, launch_file))
        print('Train: {}\n'.format(config))
    os.system('rm e2e_train_test -rf')


if __name__ == '__main__':
    config_dir = "configs/ReDet/"
    configs_dota = [
        "ReDet_re18_refpn_1x_dota15",
    ]
    e2e_train_test(config_dir, configs_dota)
