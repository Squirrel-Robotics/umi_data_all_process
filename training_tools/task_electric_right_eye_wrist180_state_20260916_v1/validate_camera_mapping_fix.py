"""Prove video-only changes preserve every selected state/action/alignment value."""
import argparse
import ast
import hashlib
import importlib.util
import sys
from experiment import *

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);sys.modules[name]=result;spec.loader.exec_module(result)
    return result

def main():
    import numpy as np
    old_path=ROOT/'backups/before_camera_mapping_fix/converter_electric.py'
    new_path=ROOT/'converter_electric.py'
    old=module('electric_before_camera_fix',old_path)
    new=module('electric_after_camera_fix',new_path)
    trees=[ast.parse(p.read_text()) for p in (old_path,new_path)]
    funcs=[{n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))} for t in trees]
    require(funcs[0].keys()==funcs[1].keys(),'Function set changed')
    changed=[n for n in funcs[0] if funcs[0][n]!=funcs[1][n]]
    require(changed==['convert'],'Only conversion video metadata may change; math functions must stay identical')
    require(new.VIDEO_KEYS=={'observation.images.head_rgb':'head','observation.images.left_wrist_rgb':'cam1',
                            'observation.images.right_wrist_rgb':'cam0'},'Incorrect camera mapping')
    args=argparse.Namespace(source=SOURCE,episode_list=SOURCE/'pass_sessions.txt',fps=10,action_horizon=50,
        max_alignment_ms=100.,max_hand_age_ms=100.,hand_alignment='nearest')
    fields=('state','action','action_is_pad','e6_source_row_indices','cam0_frame_indices','cam1_frame_indices',
            'cam0_signed_delta_ns','cam1_signed_delta_ns')
    def collect(mod):
        episodes,hidden,snapshot,plans,failures=mod.collect_plans(args)
        require(not failures and len(plans)==115,'Selected episodes failed preflight')
        hashes=[]
        for plan in plans:
            rec={'episode':plan.output_episode_id,'frames':plan.output_count}
            for field in fields:
                a=np.ascontiguousarray(getattr(plan,field))
                rec[field]={'shape':list(a.shape),'dtype':str(a.dtype),'sha256':hashlib.sha256(a.tobytes()).hexdigest()}
            hashes.append(rec)
        return snapshot,hashes
    old_snapshot,old_hashes=collect(old)
    print('Original numeric values hashed for all 115 sessions',flush=True)
    new_snapshot,new_hashes=collect(new)
    require(old_snapshot==new_snapshot,'Raw source changed while comparing')
    require(old_hashes==new_hashes,'Numeric state/action/masks/alignment changed')
    require(sum(x['frames'] for x in new_hashes)==24182,'Unexpected frame count')
    report={'status':'complete','old_generator_sha256':digest(old_path),'new_generator_sha256':digest(new_path),
        'unchanged_function_and_class_count':len(funcs[0])-1,'changed_functions':changed,
        'camera_acquisition_mapping':{'cam0':'right_wrist','cam1':'left_wrist'},'numeric_hand_channels_swapped':False,
        'numeric_frames_compared':24182,'numeric_episodes_compared':115,'compared_fields':fields,'episode_hashes':new_hashes}
    with (ROOT/'camera_mapping_fix_validation.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='episode_hashes'},indent=2),flush=True)

if __name__=='__main__':main()
