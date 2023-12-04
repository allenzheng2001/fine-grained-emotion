import argparse
from lr import train_lr
from bart import train_bart
from eval import eval_lr
from load_datasets import get_datasets, get_loader

parser = argparse.ArgumentParser(description='Example of using FLAGS in Python.')

# Add custom flags
parser.add_argument('--appraisal', action='store_true', help='use appraisals')
parser.add_argument('--labels', type=str, default = 'emotions', help='test on emotions or intensities?')
parser.add_argument('--model', type=str, default = 'lr', help='which model to run')

args = parser.parse_args()
model = args.model

train_data, val_data, test_data, all_data = get_datasets(mode = args.labels, appraisal_flag = args.appraisal) if model == 'lr' else get_loader(mode = args.labels, appraisal_flag = args.appraisal)

if(model == 'lr'):
    model = train_lr(train_data + val_data)
    eval_lr(model, test_data)
elif(model == 'bart'):
    model = train_bart(train_data)
else:
    print("ERROR: model not supported")