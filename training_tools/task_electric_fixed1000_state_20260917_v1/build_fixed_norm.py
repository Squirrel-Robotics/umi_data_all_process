#!/usr/bin/env python3
"""Immutable task-local norms: empirical EEF bounds, fixed finger 0..100 degrees.

For OpenPI compatibility q01/q99 encode the fixed finger bounds, NOT empirical
percentiles for those 12 dimensions. Raw packets use 0.1 degree/count. This is
not the robot SDK's separate Normalized-position command convention.
"""
import copy
import importlib.util
import os
import subprocess
import sys
import tempfile
from experiment import *
import numpy as np
from openpi.shared import normalize

spec=importlib.util.spec_from_file_location('base_masked_norm',BASE_NORM_TOOL)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
validate_contract=base.validate_contract

def compute(*args,**kwargs):
    result,counts=base.compute(*args,**kwargs)
    for key in ('state','actions'):
        result[key].q01[18:30]=0.
        result[key].q99[18:30]=100.
    return result,counts

def main():
    require(not ASSET.exists(),'Fixed-bound asset already exists; never overwrite')
    require(digest(BASE_NORM_TOOL)=='650123cf421631125adc47a61b99edd08ddea7fb4300c54172bfeda5235103fe','Base generator changed')
    require(digest(BASE_ASSET/'norm_stats.json')=='b7d1b92a451a1dc2b03638dec992d1150906e71b649994cdc99625eee3b27760','Original norms changed')
    # Repeat the all-raw-packets/all-parquet audit with the real Normalize and
    # Unnormalize implementations, without loading a model or touching GPUs.
    checked=subprocess.run([sys.executable,str(ROOT/'audit_fixed_finger_scale.py')],
        env=os.environ.copy(),text=True,capture_output=True,check=True,timeout=600)
    require('AUDIT_JSON_BEGIN\n' in checked.stdout,'Missing raw/fixed-range audit')
    scale_report=json.loads(checked.stdout.split('AUDIT_JSON_BEGIN\n',1)[1])
    with (ROOT/'finger_scale_audit.json').open('x') as f:json.dump(scale_report,f,indent=2)
    contract=read_json(DATASET/'meta/umi_conversion.json');info=read_json(DATASET/'meta/info.json')
    dims=validate_contract(DATASET,contract,info,expected_fps=10,expected_horizon=50,expected_task=PROMPT)
    raw,counts=base.compute(DATASET,contract,state_dim=30,horizon=50,episodes=115,frames=24182,real_slots=dims[4])
    parent=normalize.deserialize_json((BASE_ASSET/'norm_stats.json').read_text())
    for key in ('state','actions'):
        for field in ('mean','std','q01','q99'):
            np.testing.assert_allclose(getattr(raw[key],field),getattr(parent[key],field),rtol=1e-12,atol=1e-12)
    fixed=copy.deepcopy(raw)
    for key in ('state','actions'):
        fixed[key].q01[18:30]=0.;fixed[key].q99[18:30]=100.
    audit=copy.deepcopy(read_json(BASE_ASSET/'norm_stats_audit.json'))
    for key,value in counts.items():require(audit[key]==value,f'Count changed: {key}')
    staging=Path(tempfile.mkdtemp(prefix=f'.{ASSET_ID}.building-',dir=ASSET.parent))
    normfile=staging/'norm_stats.json'
    with normfile.open('x') as f:f.write(normalize.serialize_json(fixed)+'\n')
    bounds={'raw_count_bounds':[0,1000],'raw_units':'0.1 degree/count',
        'dataset_degree_bounds':[0,100],'dimensions':list(range(18,30)),
        'applies_to':['state','actions'],'kind':'fixed_linear_bounds_not_empirical_quantiles',
        'clip_training_targets':False,'model_range':[-1,1],
        'forward_ideal':'2*degrees/100-1','inverse_ideal':'(normalized+1)*50 degrees',
        'eef_first_18_dimensions':'unchanged_empirical_quantiles'}
    audit.update(asset_id=ASSET_ID,norm_stats_path=str(ASSET/'norm_stats.json'),
        norm_stats_sha256=digest(normfile),generator=str(NORM_TOOL),generator_sha256=digest(NORM_TOOL),
        base_generator_sha256=digest(BASE_NORM_TOOL),parent_asset=str(BASE_ASSET),
        parent_norm_stats_sha256=digest(BASE_ASSET/'norm_stats.json'),
        quantiles='EEF: exact empirical 0.01/0.99; hand: explicit fixed 0/100-degree bounds',
        finger_scaling=bounds,finger_scale_audit_sha256=digest(ROOT/'finger_scale_audit.json'))
    with (staging/'norm_stats_audit.json').open('x') as f:json.dump(audit,f,indent=2)
    staging.chmod(0o755)
    require(not ASSET.exists(),'Asset appeared during generation')
    staging.rename(ASSET)
    print(json.dumps({'status':'complete','asset':str(ASSET),'sha256':digest(ASSET/'norm_stats.json'),
        'target_mean_square_over_32_dims':scale_report['target_mean_square_over_32_dims'],
        'finger_scaling':bounds},indent=2),flush=True)

if __name__=='__main__':main()
