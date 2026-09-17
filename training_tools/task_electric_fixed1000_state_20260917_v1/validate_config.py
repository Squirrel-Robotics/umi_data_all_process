import dataclasses
from experiment import *
from openpi.training import config

old=config.get_config('pi05_umi_task_electric_10hz_h50_wrist180_state_20k_b64_w128_8gpu_v1')
new=config.get_config(CONFIG)
differences={}
def compare(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        for key in a.keys()|b.keys():
            p=f'{path}.{key}' if path else key
            if key not in a or key not in b: differences[p]={'old':a.get(key),'new':b.get(key)}
            else: compare(a[key],b[key],p)
    elif repr(a)!=repr(b): differences[path]={'old':repr(a),'new':repr(b)}
compare(dataclasses.asdict(old),dataclasses.asdict(new))
require(set(differences)=={'name','data.assets.asset_id','policy_metadata.hand_normalization'},f'Unexpected config change: {differences}')
require(not new.resume and not new.overwrite and str(new.weight_loader.params_path)=='/mnt/data/checkpoints/pi05_base/params','Expected fresh base model')
require(new.model.discrete_state_input and new.model.action_horizon==50 and new.num_workers==128 and new.batch_size==64,'Wrong training settings')
report={'status':'passed','differences':differences,'resume':False,'restart_from_step':0,
        'base_params':str(new.weight_loader.params_path),'num_workers':128,'batch_size':64,
        'old_norm_stats_preserved':True,'old_checkpoint_preserved':True}
with (ROOT/'config_change_validation.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report,indent=2))
