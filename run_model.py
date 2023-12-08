import argparse
from lr import train_lr
from bart import train_bart
from eval import eval_lr, eval_bart
from load_datasets import get_datasets, get_loader

parser = argparse.ArgumentParser(description='Example of using FLAGS in Python.')

# Add custom flags
parser.add_argument('--appraisal_embed', type=str, default ='None', help='use appraisals, what type of embedding?')
parser.add_argument('--labels', type=str, default = 'emotions', help='test on emotions or intensities?')
parser.add_argument('--model', type=str, default = 'lr', help='which model to run')

args = parser.parse_args()
model = args.model

train_data, val_data, test_data, all_data = get_datasets(mode = args.labels) if model == 'lr' else get_loader(mode = args.labels, appraisal_embed = args.appraisal_embed)

if(model == 'lr'):
    model = train_lr(train_data + val_data)
    eval_lr(model, test_data)
elif(model == 'bart'):
    model = train_bart(train_data, label_type = args.labels)
    eval_bart(model, test_data)
elif(model == 'bert'):
    model = train_bert(train_data, label_type = args.labels)
else:
    print("ERROR: model not supported")