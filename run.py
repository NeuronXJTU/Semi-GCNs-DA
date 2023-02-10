import os
import time
import platform


def train(out_dir):
    '''
    Train
    '''
    '''
    if platform.node() != '/' and platform.node() != 'lab-server':
        node = 86
        server_id = 6
        gpu_num = 1
        command = (
            'srun -p mia -n1 -w SH-IDC1-10-198-{}-{} --mpi=pmi2 '
            '--gres=gpu:{} python -u main.py --batch_size 64 '
            '--pre_epochs 100 --num_workers 8 '
        ).format(server_id, node, gpu_num)
    else:
    '''
    command = 'python main.py --batch_size 64 --pre_epochs 6 --num_workers 4 '
    command += (
       
        '--feat_archs  res18 '
        '--classifi_arch  Mgcnlinear '
        '--channel_size 3 --num_classes 2 --epochs 100 '
        '--optim adam --lr 1e-4 --beta1 0.9 --beta2 0.999 '
        '--weight_decay 1e-6 '
        '--txt_dir txts '
        '--controld 0.90 '
        '--labeled_txt txts/digest_labeled_pc.txt --unlabeled_txt txts/warwick_unlabeled_pc.txt --t_labeled_txt txts/warwick_labeled_pc.txt '
        '--out_dir {}'
    ).format(out_dir)
    return command


def run():
    '''
    Run
    '''
    out_dir = 'digest_warwick'
    command = train(out_dir)
    print(command)
    if not os.path.exists(out_dir):
         os.mkdir(out_dir)
    with open(os.path.join(out_dir, 'log.txt'), 'a') as writer:
        writer.write(
            '\n' + time.asctime(time.localtime(time.time())) + '\n' +
            command + '\n'
        )
    os.system(command)

if __name__ == '__main__':
    run()

