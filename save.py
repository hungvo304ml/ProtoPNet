import os
import torch

def save_model_w_condition(model, model_dir, model_name, accu, auc, ap, target_accu, log=print):
    '''
    model: this is not the multigpu model
    '''
    print("[+] In function save_model_w_condition")
    if accu > target_accu:
        log('\tabove {0:.2f}%'.format(target_accu * 100))
        # torch.save(obj=model.state_dict(), f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))
        torch.save(obj=model, f=os.path.join(model_dir, (model_name + 'acc-{0:.4f}_auc-{1:.4f}_ap-{2:.4f}.pth').format(accu, auc, ap)))

        print("[+] Model is actually saved")
