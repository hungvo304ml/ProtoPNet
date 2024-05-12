import os
import shutil

import torch
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import re
import json

from helpers import makedir
import model
import push
import prune
import train_and_test as tnt
import save
import yaml
from log import create_logger
from preprocess import mean, std, preprocess_input_function
from lightning.pytorch import seed_everything


parser = argparse.ArgumentParser()
parser.add_argument("--config-path", dest="config_path", required=True)
parser.add_argument('-gpuid', nargs=1, type=str, default='0') # python3 main.py -gpuid=0,1,2,3
parser.add_argument("-workers", "--workers", type=int)

parser.add_argument("-arch", "--base_architecture", type=str, default='vgg19')
parser.add_argument("-imsize", "--img_size", type=int, default=224)
parser.add_argument("-protoshape", "--prototype_shape", nargs='+', type=int)
parser.add_argument("-nclass", "--num_classes", type=int, default=200)
parser.add_argument("-proto_act_fn", "--prototype_activation_function", type=str, default='log')
parser.add_argument("-addon_layers", "--add_on_layers_type", type=str, default='regular')
parser.add_argument("-exp", "--experiment_run", type=str)

parser.add_argument("-data_path", "--data_path", type=str, required=True)
parser.add_argument("-train_dir", "--train_dir", type=str, required=True)
parser.add_argument("-test_dir", "--test_dir", type=str, required=True)
parser.add_argument("-train_push_dir", "--train_push_dir", type=str, required=True)
parser.add_argument("-train_bs", "--train_batch_size", type=int, default=80)
parser.add_argument("-test_bs", "--test_batch_size", type=int, default=100)
parser.add_argument("-train_push_bs", "--train_push_batch_size", type=int, default=75)

parser.add_argument("-joint_opt_lrs", "--joint_optimizer_lrs", type=json.loads, default='{"features": 1e-4, "add_on_layers": 3e-3, "prototype_vectors": 3e-3}')
parser.add_argument("-joint_lr_step_size", "--joint_lr_step_size", type=int, default=5)
parser.add_argument("-warm_opt_lrs", "--warm_optimizer_lrs", type=json.loads, default='{"add_on_layers": 3e-3, "prototype_vectors": 3e-3}')
parser.add_argument("--last_layer_optimizer_lr", type=float, default=1e-4)
parser.add_argument("--coefs", type=json.loads)
parser.add_argument("--num_train_epochs", type=int, default=1000)
parser.add_argument("--num_warm_epochs", type=int, default=5)
parser.add_argument("--push_start", type=int, default=10)
parser.add_argument("--push_epochs_freq", type=int, default=10)

    
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]

print(os.environ['CUDA_VISIBLE_DEVICES'])


with open(args.config_path, 'r') as file:
    yaml_cfg = yaml.safe_load(file)


for arg in vars(args):   
    attr = getattr(args, arg)
    
    if not isinstance(attr, dict):
        yaml_cfg[arg] = attr
    else:
        print(arg, attr)
        yaml_cfg[arg] = dict()
        for key, value in attr.items():
            yaml_cfg[arg][key] = value

print(yaml_cfg)   



seed_everything(seed=42, workers=True)

# book keeping namings and code
# from settings import base_architecture, img_size, prototype_shape, num_classes, \
#                      prototype_activation_function, add_on_layers_type, experiment_run
base_architecture, img_size, prototype_shape, num_classes, \
    prototype_activation_function, add_on_layers_type, experiment_run = \
        yaml_cfg['base_architecture'], yaml_cfg['img_size'], yaml_cfg['prototype_shape'], \
        yaml_cfg['num_classes'], yaml_cfg['prototype_activation_function'], yaml_cfg['add_on_layers_type'], \
        yaml_cfg['experiment_run']


base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

model_dir = './saved_models/' + base_architecture + '/' + experiment_run + '/'
makedir(model_dir)
shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
# shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
with open(os.path.join(model_dir, 'settings.yaml'), 'w') as yamlfile:
    data = yaml.dump(yaml_cfg, yamlfile)
    
shutil.copy(src=os.path.join(os.getcwd(), base_architecture_type + '_features.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
img_dir = os.path.join(model_dir, 'img')
makedir(img_dir)
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'

# load the data
# from settings import train_dir, test_dir, train_push_dir, \
#                      train_batch_size, test_batch_size, train_push_batch_size
train_dir, test_dir, train_push_dir, \
        train_batch_size, test_batch_size, train_push_batch_size = \
        yaml_cfg['train_dir'], yaml_cfg['test_dir'], yaml_cfg['train_push_dir'], \
        yaml_cfg['train_batch_size'], yaml_cfg['test_batch_size'], yaml_cfg['train_push_batch_size']

normalize = transforms.Normalize(mean=mean,
                                 std=std)

# all datasets
# train set
train_dataset = datasets.ImageFolder(
    train_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=False)
# push set
train_push_dataset = datasets.ImageFolder(
    train_push_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
    ]))
train_push_loader = torch.utils.data.DataLoader(
    train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=False)

# push set for visualization
train_push_vis_dataset = datasets.ImageFolder(
    train_push_dir,
    transforms.Compose([        
        transforms.Resize(size=(1024, 1024)),
        transforms.ToTensor(),
    ]))
train_push_vis_loader = torch.utils.data.DataLoader(
    train_push_vis_dataset, batch_size=train_push_batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=False)

# test set
test_dataset = datasets.ImageFolder(
    test_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=False)

# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
log('training set size: {0}'.format(len(train_loader.dataset)))
log('push set size: {0}'.format(len(train_push_loader.dataset)))
log('test set size: {0}'.format(len(test_loader.dataset)))
log('batch size: {0}'.format(train_batch_size))

# construct the model
ppnet = model.construct_PPNet(base_architecture=base_architecture,
                              pretrained=True, img_size=img_size,
                              prototype_shape=prototype_shape,
                              num_classes=num_classes,
                              prototype_activation_function=prototype_activation_function,
                              add_on_layers_type=add_on_layers_type)
#if prototype_activation_function == 'linear':
#    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)
class_specific = True

# define optimizer
# from settings import joint_optimizer_lrs, joint_lr_step_size
joint_optimizer_lrs, joint_lr_step_size = yaml_cfg['joint_optimizer_lrs'], yaml_cfg['joint_lr_step_size']

joint_optimizer_specs = \
[{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
 {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
]
joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

# from settings import warm_optimizer_lrs
warm_optimizer_lrs = yaml_cfg['warm_optimizer_lrs']
warm_optimizer_specs = \
[{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
]
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

# from settings import last_layer_optimizer_lr
last_layer_optimizer_lr = yaml_cfg['last_layer_optimizer_lr']
last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

# weighting of different training losses
# from settings import coefs
coefs = yaml_cfg['coefs']

# number of training epochs, number of warm epochs, push start epoch, push epochs
# from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs
num_train_epochs, num_warm_epochs, push_start, push_epochs_freq = yaml_cfg['num_train_epochs'], yaml_cfg['num_warm_epochs'], \
                                                            yaml_cfg['push_start'], yaml_cfg['push_epochs_freq']

push_epochs = [i for i in range(num_train_epochs) if i % push_epochs_freq == 0]

# train the model
log('start training')
import copy
for epoch in range(num_train_epochs):
    log('epoch: \t{0}'.format(epoch))

    if epoch < num_warm_epochs:
        tnt.warm_only(model=ppnet_multi, log=log)
        _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log)
    else:
        tnt.joint(model=ppnet_multi, log=log)
        joint_lr_scheduler.step()
        _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log)

    accu, auc, ap = tnt.test(model=ppnet_multi, dataloader=test_loader,
                    class_specific=class_specific, log=log)
    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu, auc=auc, ap=ap,
                                target_accu=0, log=log)

    if epoch >= push_start and epoch in push_epochs:
        push.push_prototypes(
            train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
            vis_loader=train_push_vis_loader,
            prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
            class_specific=class_specific,
            preprocess_input_function=preprocess_input_function, # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
            epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True,
            log=log)
        accu, auc, ap = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=log)
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu, auc=auc, ap=ap,
                                    target_accu=0, log=log)

        if prototype_activation_function != 'linear':
            tnt.last_only(model=ppnet_multi, log=log)
            for i in range(20):
                log('iteration: \t{0}'.format(i))
                _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                              class_specific=class_specific, coefs=coefs, log=log)
                accu, auc, ap = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                class_specific=class_specific, log=log)
                save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=accu, auc=auc, ap=ap,
                                            target_accu=0, log=log)
   
logclose()

