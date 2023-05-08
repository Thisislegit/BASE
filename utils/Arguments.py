import argparse

parser = argparse.ArgumentParser(description='IMDB QO Parameters')


parser.add_argument('--path', default='./Models/now.pth',
                 help='Location to store model')
parser.add_argument('--epoch_start', default=1, type=int)
parser.add_argument('--epoch_end', default=400, type=int)

# DQN parameters
parser.add_argument('--epsilon_start', default=1, type=float)
parser.add_argument('--epsilon_end', default=0.1, type=float)
parser.add_argument('--capacity', default=300, type=int)
parser.add_argument('--epsilon_decay', default=0.98, type=float)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--sync_batch_size', default=50, type=int)

# Net Parameters
parser.add_argument('--learning_rate', default=0.0005, type=float)
parser.add_argument('--max_lr', default=0.001, type=float)
parser.add_argument('--steps_per_epoch', default=100, type=int)



if __name__ == '__main__':
    args = parser.parse_args()
    print(args.path)