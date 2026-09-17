"""Read-only audit; test fixed raw-count 0..1000 bounds in memory only."""
import copy
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, '/home/dzq/openpi/src')
from openpi import transforms
from openpi.shared import normalize

ROOT=Path('/mnt/data/dzq/umi_v2/datasets/task_electric_lerobot_10hz_h50_right_eye_wrist180_v1')
SOURCE=Path('/mnt/data/dzq/umi_v2/data/task_electric')
ASSET=Path('/mnt/data/dzq/openpi/data/assets/umi_task_electric_10hz_h50_right_eye_wrist180_masked_v1')
contract=json.loads((ROOT/'meta/umi_conversion.json').read_text())
old=normalize.deserialize_json((ASSET/'norm_stats.json').read_text())
fixed=copy.deepcopy(old)
for key in ('state','actions'):
    fixed[key].q01[18:30]=0.
    fixed[key].q99[18:30]=100.
old_transform=transforms.Normalize(old,use_quantiles=True)
new_transform=transforms.Normalize(fixed,use_quantiles=True)
inverse=transforms.Unnormalize(fixed,use_quantiles=True)
raw_min=np.full(12,np.inf);raw_max=np.full(12,-np.inf)
packet_counts=[0,0];raw_outside=np.zeros(12,dtype=np.int64)
formats=set(); packet_errors=[]
state_count=0;action_count=0;pad_count=0
state_min=np.full(12,np.inf);state_max=np.full(12,-np.inf)
action_min=np.full(12,np.inf);action_max=np.full(12,-np.inf)
energies={key:np.zeros(30) for key in ('old_state','new_state','old_actions','new_actions')}
norm_ranges={key:[np.full(12,np.inf),np.full(12,-np.inf)] for key in energies}
roundtrip=0.;raw_grid_residual=0.;future_finger_error=0.;eef_change=0.
outside_state=0;outside_action=0
episodes=[]

def array(column, shape):
    value=column.combine_chunks()
    if hasattr(value,'storage'):value=value.storage
    while hasattr(value,'values'):value=value.values
    return value.to_numpy(zero_copy_only=False).reshape(shape).astype(np.float64)

for ep in contract['episodes']:
    src=SOURCE/ep['source_episode_id']
    for side_index,side in enumerate(('left','right')):
        sl=slice(side_index*6,(side_index+1)*6)
        with (src/'serial'/f'{side}_rx_packets.csv').open() as f:
            for line,row in enumerate(csv.DictReader(f),2):
                if row['packet_type_name']!='HAND_STATE_FRAME' or row['payload_status']!='full':continue
                vals=np.array([int(x) for x in row['decoded_values'].split(';')[:6]])
                fmt=row['decoded_values_format'];formats.add(fmt)
                if 'u16_position_0p1deg' not in fmt or len(vals)!=6:
                    packet_errors.append({'source':str(src),'side':side,'line':line,'format':fmt})
                packet_counts[side_index]+=1
                raw_min[sl]=np.minimum(raw_min[sl],vals)
                raw_max[sl]=np.maximum(raw_max[sl],vals)
                raw_outside[sl]+=((vals<0)|(vals>1000))
    tab=pq.read_table(ROOT/ep['parquet']['path'],columns=['observation.state','action','action_is_pad'])
    n=len(tab)
    s=array(tab['observation.state'],(n,30))
    a=array(tab['action'],(n,50,30))
    pad=array(tab['action_is_pad'],(n,50)).astype(bool)
    assert np.isfinite(s).all() and np.isfinite(a).all()
    future=np.minimum(np.arange(n)[:,None]+np.arange(1,51)[None,:],n-1)
    future_finger_error=max(future_finger_error,float(np.max(np.abs(a[...,18:]-s[future,18:]))))
    v=a[~pad]
    state_count+=n;action_count+=len(v);pad_count+=int(pad.sum())
    raw_grid_residual=max(raw_grid_residual,float(np.max(np.abs(s[:,18:]*10-np.rint(s[:,18:]*10)))))
    state_min=np.minimum(state_min,s[:,18:].min(axis=0));state_max=np.maximum(state_max,s[:,18:].max(axis=0))
    action_min=np.minimum(action_min,v[:,18:].min(axis=0));action_max=np.maximum(action_max,v[:,18:].max(axis=0))
    outside_state+=int(np.count_nonzero((s[:,18:]<0)|(s[:,18:]>100)))
    outside_action+=int(np.count_nonzero((v[:,18:]<0)|(v[:,18:]>100)))
    before={'state':s.copy(),'actions':v.copy()}
    prev=old_transform(copy.deepcopy(before));after=new_transform(copy.deepcopy(before))
    restored=inverse(copy.deepcopy(after))
    for key in ('state','actions'):
        roundtrip=max(roundtrip,float(np.max(np.abs(restored[key]-before[key]))))
        eef_change=max(eef_change,float(np.max(np.abs(prev[key][...,:18]-after[key][...,:18]))))
        for prefix,value in [('old',prev[key]),('new',after[key])]:
            dest=f'{prefix}_{key}'
            energies[dest]+=np.sum(value**2,axis=0)
            norm_ranges[dest][0]=np.minimum(norm_ranges[dest][0],value[:,18:].min(axis=0))
            norm_ranges[dest][1]=np.maximum(norm_ranges[dest][1],value[:,18:].max(axis=0))
    episodes.append({'episode':ep['episode_index'],'source':ep['source_episode_id'],
        'old_action_target_mean_square_32':float(np.sum(prev['actions']**2)/len(v)/32),
        'fixed_action_target_mean_square_32':float(np.sum(after['actions']**2)/len(v)/32)})
    if (ep['episode_index']+1)%20==0:
        print(f'Checked {ep["episode_index"]+1}/{len(contract["episodes"])} episodes',flush=True)

names=[f'{side}_{joint}' for side in ('left','right') for joint in ('thumb_flex','thumb_aux','index','middle','ring','little')]
summary=[]
for i,name in enumerate(names):
    summary.append({'joint':name,'raw_packet_min':raw_min[i],'raw_packet_max':raw_max[i],
        'raw_count_outside_0_1000':int(raw_outside[i]),
        'state_deg_min':state_min[i],'state_deg_max':state_max[i],
        'action_deg_min':action_min[i],'action_deg_max':action_max[i],
        'current_q01_action':float(old['actions'].q01[18+i]),
        'current_q99_action':float(old['actions'].q99[18+i]),
        'old_action_norm_min':norm_ranges['old_actions'][0][i],
        'old_action_norm_max':norm_ranges['old_actions'][1][i],
        'fixed_action_norm_min':norm_ranges['new_actions'][0][i],
        'fixed_action_norm_max':norm_ranges['new_actions'][1][i]})

result={'mode':'read_only_in_memory_candidate_no_config_or_asset_changes','episodes':len(episodes),
    'frames':state_count,'real_action_slots':action_count,'excluded_padding_slots':pad_count,
    'raw_packet_counts':dict(zip(('left','right'),packet_counts)),
    'raw_packet_formats':sorted(formats),'packet_format_errors':packet_errors,
    'state_finger_values_outside_0_100_deg':outside_state,'action_finger_values_outside_0_100_deg':outside_action,
    'max_dataset_raw_integer_residual':raw_grid_residual,'future_finger_vs_future_state_max_error':future_finger_error,
    'normalize_unnormalize_roundtrip_max_error':roundtrip,'eef_normalization_difference':eef_change,
    'candidate':{'raw_count_range':[0,1000],'dataset_degree_range':[0,100],
        'dimensions':list(range(18,30)),'applies_to':['state','actions'],
        'norm_q01':0,'norm_q99':100,'formula_ideal':'normalized = 2 * degrees / 100 - 1 = 2 * raw / 1000 - 1',
        'inverse_ideal':'degrees = (normalized + 1) * 50; raw = degrees * 10',
        'epsilon_in_current_openpi':1e-6,'clipping':False},
    'target_mean_square_over_32_dims':{
        k:float(v.sum()/(state_count if k.endswith('state') else action_count)/32) for k,v in energies.items()},
    'per_joint':summary,'episode72':episodes[72],
    'not_a_model_loss_measurement':True}
assert not packet_errors and not raw_outside.any() and not outside_state and not outside_action
assert future_finger_error<1e-5 and roundtrip<1e-10 and eef_change==0
print('AUDIT_JSON_BEGIN',flush=True)
print(json.dumps(result,indent=2),flush=True)
