#!/usr/bin/env python3
"""Reuse the completed full video audit only if EVERY recorded byte is unchanged.

Then execute both four-GPU configs' real CPU transforms and dataloaders. No GPU
initialization or model-weight loading is performed by this script.
"""
import ast
import copy
import dataclasses
import importlib.util
import json
import sys
from experiment import *


def fingerprints():
    prior = read_json(OPENPI / 'logs' / (BASE_EXP+'.launch.json'))
    frozen = prior['evidence']['fingerprints']
    result = {}
    for name, expected in frozen.items():
        path=Path(name)
        if path == OPENPI/'src/openpi/training/config.py': continue
        actual=digest(path)
        require(actual==expected, f'Previously verified file changed: {path}')
        result[name]=actual
    for name in ('experiment.py','preflight.py','launch_pair.py','serve_wrist180.py','wrist180_inference.py'):
        path=PACKAGE/name; result[str(path)]=digest(path)
    for path in (OPENPI/'src/openpi/training/config.py', PACKAGE/'backups/config.before_dual4gpu.py',
                 OPENPI/'logs'/(BASE_EXP+'.smoke.log'), OPENPI/'logs'/(BASE_EXP+'.launch.json')):
        result[str(path)]=digest(path)
    return result


def main():
    import jax
    import numpy as np
    from openpi import transforms
    from openpi.training import config, data_loader
    require(all(d.platform=='cpu' for d in jax.devices()), 'Run preflight with JAX_PLATFORMS=cpu')
    frozen=fingerprints()
    # Verify that installation only added the two new configs.
    before=ast.parse((PACKAGE/'backups/config.before_dual4gpu.py').read_text())
    current=ast.parse((OPENPI/'src/openpi/training/config.py').read_text())
    for job in JOBS:
        found=[n for n in ast.walk(current) if isinstance(n,ast.Call) and any(
            k.arg=='name' and isinstance(k.value,ast.Constant) and k.value.value==job['config'] for k in n.keywords)]
        require(len(found)==1, 'Expected exactly one independent config')
        for n in ast.walk(current):
            if isinstance(n,ast.List) and found[0] in n.elts: n.elts.remove(found[0])
    require(ast.dump(before)==ast.dump(current), 'Existing configuration changed')
    log=(OPENPI/'logs'/(BASE_EXP+'.smoke.log')).read_text()
    decoder=json.JSONDecoder(); prior_smoke=None
    for index, char in enumerate(log):
        if char!='{': continue
        try: value,_=decoder.raw_decode(log,index)
        except ValueError: continue
        if isinstance(value,dict) and value.get('status')=='complete' and value.get('config')==BASE_CONFIG:
            prior_smoke=value; break
    require(prior_smoke is not None and prior_smoke['wrist_rotation']['all_yuv_frames_verified'] is True,
            'No successful full video release gate to reuse')
    require(prior_smoke['episodes']==85 and prior_smoke['frames']==9995, 'Wrong verified data')
    require(read_json(DATASET/'meta/umi_conversion.json')['status']=='complete', 'Dataset not published')
    require(digest(ASSET/'norm_stats.json')==prior_smoke['norm_stats_sha256'], 'Norm changed')
    spec=importlib.util.spec_from_file_location('verified_previous_smoke', BASE_TOOLS/'smoke_training.py')
    old_checks=importlib.util.module_from_spec(spec); sys.modules[spec.name]=old_checks; spec.loader.exec_module(old_checks)
    model_path=old_checks.no_state_model_paths(OPENPI/'src/openpi/models/pi0.py', OPENPI/'src/openpi/models/model.py')
    from wrist180_inference import self_test
    adapter_test=self_test()
    base=config.get_config(BASE_CONFIG)
    configs=[config.get_config(job['config']) for job in JOBS]
    no,yes=configs
    require(dataclasses.replace(yes,name=no.name,model=no.model,policy_metadata=no.policy_metadata)==no,
            'Pair differs outside model state flag/name/metadata')
    require(dataclasses.replace(yes.model,discrete_state_input=False)==no.model, 'Unexpected model differences')
    samples=[]
    print('Full 85-episode video audit reused after all file hashes passed.',flush=True)
    for job,cfg in zip(JOBS,configs):
        require(dataclasses.replace(cfg,name=base.name,model=base.model,policy_metadata=base.policy_metadata,
                                    batch_size=64,fsdp_devices=8)==base, 'Recipe differs from baseline')
        require(cfg.batch_size==32 and cfg.num_workers==32 and cfg.fsdp_devices==4 and cfg.num_train_steps==10000,
                'Wrong per-job resource recipe')
        require(cfg.save_interval==cfg.keep_period==1000 and not cfg.resume and not cfg.overwrite, 'Unsafe run settings')
        require(cfg.model.discrete_state_input is job['state'] and cfg.policy_metadata['state_values_used_for_conditioning'] is job['state'],
                'State metadata mismatch')
        require(Path(cfg.data.repo_id)==DATASET and cfg.data.use_delta_eef_actions is False, 'Wrong dataset/action semantics')
        data=cfg.data.create(cfg.assets_dirs,cfg.model)
        require(not data.action_sequence_keys and data.action_padding_mask_key=='action_is_pad' and data.sample_filter_key is None,
                'Wrong H50 padding/anchor contract')
        token=[t for t in data.model_transforms.inputs if isinstance(t,transforms.TokenizePrompt)]
        require(len(token)==1 and token[0].discrete_state_input is job['state'], 'Wrong tokenizer state mode')
        before_model=transforms.compose([*data.repack_transforms.inputs,*data.data_transforms.inputs,
            transforms.Normalize(data.norm_stats,use_quantiles=data.use_quantile_norm)])
        to_model=transforms.compose(data.model_transforms.inputs)
        raw=data_loader.create_torch_dataset(data,50,cfg.model)
        transformed=data_loader.transform_dataset(raw,data)
        require(len(raw)==len(transformed)==9995, 'Rows filtered unexpectedly')
        for row in (0,4997,9994):
            first=before_model(copy.deepcopy(raw[row])); altered=copy.deepcopy(first)
            altered['state']=np.linspace(-0.875,0.875,30,dtype=np.float64)
            a,b=to_model(copy.deepcopy(first)),to_model(altered)
            expected_tokens,expected_mask=token[0].tokenizer.tokenize(cfg.data.default_prompt,first['state'] if job['state'] else None)
            np.testing.assert_array_equal(a['tokenized_prompt'],expected_tokens)
            np.testing.assert_array_equal(a['tokenized_prompt_mask'],expected_mask)
            same=np.array_equal(a['tokenized_prompt'],b['tokenized_prompt'])
            require(same is (not job['state']), f'State sensitivity wrong: {job["variant"]}/{row}')
            np.testing.assert_array_equal(a['actions'],b['actions'])
            np.testing.assert_array_equal(a['action_is_pad'],b['action_is_pad'])
            require(np.asarray(a['state']).shape==(32,) and np.asarray(a['actions']).shape==(50,32), 'Wrong numeric shapes')
            require(np.asarray(a['action_is_pad']).shape==(50,), 'Wrong mask shape')
            for key,image in a['image'].items():
                require(np.asarray(image).shape==(224,224,3), 'Wrong image shape')
                np.testing.assert_array_equal(image,b['image'][key])
            reference=transformed[row]
            for key in ('tokenized_prompt','tokenized_prompt_mask','state','actions','action_is_pad'):
                np.testing.assert_array_equal(a[key],reference[key])
            other=data_loader.transform_dataset(raw,configs[1 if not job['state'] else 0].data.create(cfg.assets_dirs,configs[1 if not job['state'] else 0].model))[row]
            for key,image in a['image'].items(): np.testing.assert_array_equal(image,other['image'][key])
            samples.append({'variant':job['variant'],'row':row,'state_perturbation_changes_tokens':not same,
                            'token_count':int(np.asarray(expected_mask).sum()),'images_actions_unchanged':True})
        loader=data_loader.create_data_loader(dataclasses.replace(cfg,batch_size=1,num_workers=0),shuffle=False,num_batches=1,framework='jax')
        obs,actions=next(iter(loader))
        require(obs.state.shape==(1,32) and actions.shape==(1,50,32) and obs.action_is_pad.shape==(1,50), 'Wrong real batch')
        print(f'{job["variant"]}: real CPU dataloader and state-conditioning test passed.',flush=True)
    require(fingerprints()==frozen, 'Files changed during preflight')
    report={'status':'complete','fingerprints':frozen,'jobs':JOBS,'episodes':85,'frames':9995,
            'full_video_audit_reused':True,'previous_full_smoke_sha256':digest(OPENPI/'logs'/(BASE_EXP+'.smoke.log')),
            'samples':samples,'no_continuous_state_branch_pi05':model_path,'inference_adapter_test':adapter_test}
    with (PACKAGE/'preflight.report.json').open('x') as f: json.dump(report,f,indent=2)
    print('PAIR_PREFLIGHT_COMPLETE',flush=True)


if __name__=='__main__': main()
