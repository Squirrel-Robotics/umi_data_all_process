#!/usr/bin/env python3
"""CPU release gate: approved sessions, all numeric targets, video proofs, state input."""
import argparse
import copy
import dataclasses
import importlib.util
import json
import sys
from experiment import *


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec); sys.modules[name]=value; spec.loader.exec_module(value)
    return value


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); args=p.parse_args()
    require(args.config==CONFIG,'Wrong training configuration')
    import jax
    import numpy as np
    import pyarrow.parquet as pq
    from openpi import transforms
    from openpi.training import config,data_loader
    from openpi.shared import normalize
    import converter_electric as converter
    from wrist180_inference import self_test
    require(all(d.platform=='cpu' for d in jax.devices()),'CPU-only smoke required')
    cfg=config.get_config(CONFIG)
    for key,value in {'batch_size':64,'num_workers':32,'num_train_steps':20000,'save_interval':1000,
                      'keep_period':1000,'fsdp_devices':8,'resume':False,'overwrite':False}.items():
        require(getattr(cfg,key)==value,f'Wrong training setting {key}')
    require(cfg.model.pi05 and cfg.model.discrete_state_input is True and cfg.model.action_horizon==50
            and cfg.model.action_dim==32,'Expected Pi0.5/state/H50/32')
    require(Path(cfg.data.repo_id)==DATASET and cfg.data.assets.asset_id==ASSET_ID
            and cfg.data.default_prompt==PROMPT,'Wrong dataset/norm/prompt')
    require(cfg.data.head_crop_normalized is None and cfg.data.use_head_camera
            and not cfg.data.use_delta_eef_actions and not cfg.data.output_absolute_eef_actions,'Wrong image/action transforms')
    require(cfg.policy_metadata['wrist_rotation_degrees']=={'left':180,'right':180}
            and cfg.policy_metadata['additional_training_wrist_rotation'] is False,'Double/missing wrist rotation')
    for path in (SOURCE/'pass_sessions.txt',ROOT/'pass_sessions.txt',DATASET/'meta/episode_selection.txt'):
        require(digest(path)==SELECTION_SHA,'Approved session list changed')
    selected=(ROOT/'pass_sessions.txt').read_text().splitlines()
    require(len(selected)==len(set(selected))==115,'Expected all 115 selected sessions')
    contract=read_json(DATASET/'meta/umi_conversion.json'); info=read_json(DATASET/'meta/info.json')
    require(contract['status']=='complete' and contract['target']==str(DATASET) and contract['source']==str(SOURCE),'Wrong publication')
    require(contract['task']==PROMPT and contract['total_frames']==24182 and contract['total_episodes']==115,'Wrong task/counts')
    require([e['source_episode_id'] for e in contract['episodes']]==selected
            and [e['output_episode_id'] for e in contract['episodes']]==selected,'Session loss, reordering or splitting')
    for name in ('converter_electric.py','publish_repaired_dataset.py'):
        require(digest(ROOT/name)==digest(DATASET/'meta'/name),'Generator provenance changed')
    require(contract['video_preprocessing']['wrist_rotation_degrees']=={
        'observation.images.left_wrist_rgb':180,'observation.images.right_wrist_rgb':180},'Wrong encoded wrist rotation')
    require(contract['video_preprocessing']['camera_acquisition_mapping']=={
        'cam0':'right_wrist','cam1':'left_wrist'},'Wrong user-confirmed wrist camera mapping')
    require(contract['video_preprocessing']['numeric_hand_channels_swapped'] is False,'Numeric hands must not be swapped')
    require(contract['video_keys']=={'observation.images.head_rgb':'head',
        'observation.images.left_wrist_rgb':'cam1','observation.images.right_wrist_rgb':'cam0'},'Wrong video key mapping')
    norm=module('electric_norm',NORM_TOOL)
    dimensions=norm.validate_contract(DATASET,contract,info,expected_fps=10,expected_horizon=50,expected_task=PROMPT)
    require(dimensions[:4]==(30,50,115,24182),'Invalid dataset dimensional contract')
    audit=read_json(ASSET/'norm_stats_audit.json')
    require(audit['status']=='complete' and audit['dataset']==str(DATASET)
            and audit['dataset_contract_sha256']==digest(DATASET/'meta/umi_conversion.json')
            and audit['asset_id']==ASSET_ID and audit['norm_stats_sha256']==digest(ASSET/'norm_stats.json')
            and audit['generator_sha256']==digest(NORM_TOOL),'Wrong normalization provenance')
    recomputed,counts=norm.compute(DATASET,contract,state_dim=30,horizon=50,episodes=115,frames=24182,real_slots=dimensions[4])
    saved=normalize.deserialize_json((ASSET/'norm_stats.json').read_text())
    for key in ('state','actions'):
        for field in ('mean','std','q01','q99'):
            np.testing.assert_allclose(getattr(saved[key],field),getattr(recomputed[key],field),rtol=1e-12,atol=1e-12)
    for key,value in counts.items(): require(audit[key]==value,f'Norm count mismatch {key}')
    # Recompute trajectories/targets from raw source using the unchanged numeric
    # conversion implementation, never by taking future dataset action rows.
    _,_,snapshot,plans,failures=converter.collect_plans(argparse.Namespace(
        source=SOURCE,episode_list=SOURCE/'pass_sessions.txt',fps=10,action_horizon=50,
        max_alignment_ms=100.,max_hand_age_ms=100.,hand_alignment='nearest'))
    require(not failures and len(plans)==115 and snapshot==contract['source_snapshot'],'Raw inputs changed or failed')
    checked_videos=0
    for plan,manifest in zip(plans,contract['episodes']):
        table=pq.read_table(DATASET/manifest['parquet']['path'])
        require(digest(DATASET/manifest['parquet']['path'])==manifest['parquet']['sha256'],'Parquet hash mismatch')
        np.testing.assert_allclose(np.asarray(table['observation.state'].to_pylist()),plan.state,rtol=1e-6,atol=1e-7)
        np.testing.assert_allclose(np.asarray(table['action'].to_pylist()),plan.action,rtol=1e-6,atol=1e-7)
        np.testing.assert_array_equal(np.asarray(table['action_is_pad'].to_pylist()),plan.action_is_pad)
        np.testing.assert_allclose(table['timestamp'].to_numpy(),np.arange(plan.output_count)/10,rtol=1e-6,atol=1e-6)
        head=manifest['head_video_preprocessing']
        require(head['crop_xywh']==[head['actual_width']//2,0,head['actual_width']//2,head['actual_height']]
                and head['selected_eye']=='right','Wrong E6 eye crop')
        require(head==plan.quality['head_video_preprocessing'],'Head source/probe mismatch')
        sources={'observation.images.head_rgb':(plan.files.head_video,plan.e6_source_row_indices,0),
                 'observation.images.left_wrist_rgb':(plan.files.cam1_video,plan.cam1_frame_indices,180),
                 'observation.images.right_wrist_rgb':(plan.files.cam0_video,plan.cam0_frame_indices,180)}
        for key,(source,indices,rotation) in sources.items():
            rec=manifest['videos'][key]; video=DATASET/rec['path']
            require(video.resolve().is_relative_to(DATASET.resolve()),'Video path escapes dataset')
            require(rec['sha256']==digest(video) and rec['size_bytes']==video.stat().st_size,'Video bytes changed')
            require(rec['source_video']==str(source) and rec['rotation_degrees']==rotation,'Video source/rotation mismatch')
            require(rec['frames']==plan.output_count and rec['fps']==10 and (rec['width'],rec['height'])==(640,480),'Wrong video shape/FPS/count')
            require(rec['source_frame_indices_sha256']==__import__('hashlib').sha256(np.asarray(indices,dtype=np.int64).tobytes()).hexdigest(), 'Video selection changed')
            require(rec['crf']==0 and rec['decoded_reference_verified'] is True
                    and rec['presentation_timestamps_verified'] is True and len(rec['decoded_frame_hashes_sha256'])==64,'Missing full lossless video/timestamp proof')
            if rotation: require(rec['preprocessing_filters'][-2:]==['hflip','vflip'],'Wrong wrist pixel operation')
            else: require(rec['source_crop']==head['source_crop'],'Head crop was not encoded')
            checked_videos+=1
        if (manifest['episode_index']+1)%20==0: print(f'Raw/numeric/video release gate {manifest["episode_index"]+1}/115',flush=True)
    del plans
    data=cfg.data.create(cfg.assets_dirs,cfg.model)
    require(not data.action_sequence_keys and data.sample_filter_key is None and data.action_padding_mask_key=='action_is_pad','Wrong loader H50 masking')
    raw=data_loader.create_torch_dataset(data,50,cfg.model); transformed=data_loader.transform_dataset(raw,data)
    require(len(raw)==len(transformed)==24182,'Unexpected sample filtering')
    before_model=transforms.compose([*data.repack_transforms.inputs,*data.data_transforms.inputs,
        transforms.Normalize(data.norm_stats,use_quantiles=data.use_quantile_norm)])
    model_transform=transforms.compose(data.model_transforms.inputs)
    tok=[v for v in data.model_transforms.inputs if isinstance(v,transforms.TokenizePrompt)]
    require(len(tok)==1 and tok[0].discrete_state_input is True,'State tokenizer is disabled')
    samples=[]
    for row in (0,12091,24180,24181):
        sample=transformed[row]; before=before_model(copy.deepcopy(raw[row]))
        actual=model_transform(copy.deepcopy(before)); perturbed=copy.deepcopy(before)
        perturbed['state']=np.linspace(-.875,.875,30); changed=model_transform(perturbed)
        require(not np.array_equal(actual['tokenized_prompt'],changed['tokenized_prompt']),'State does not affect model conditioning')
        tokens,mask=tok[0].tokenizer.tokenize(PROMPT,before['state'])
        np.testing.assert_array_equal(sample['tokenized_prompt'],tokens)
        np.testing.assert_array_equal(sample['tokenized_prompt_mask'],mask)
        require(np.asarray(sample['state']).shape==(32,) and np.asarray(sample['actions']).shape==(50,32),'Wrong model numeric shapes')
        np.testing.assert_array_equal(sample['state'][30:],0)
        np.testing.assert_array_equal(sample['actions'][:,30:],0)
        np.testing.assert_array_equal(actual['actions'],changed['actions'])
        np.testing.assert_array_equal(actual['action_is_pad'],changed['action_is_pad'])
        require(set(sample['image'])=={'base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb'},'Camera missing')
        for key,value in sample['image'].items():
            require(np.asarray(value).shape==(224,224,3),'Wrong image shape')
            np.testing.assert_array_equal(value,changed['image'][key])
        samples.append({'row':row,'state_changes_tokens':True,'token_count':int(np.sum(mask)),
                        'real_action_slots':int((~np.asarray(sample['action_is_pad'])).sum())})
    require(samples[-2]['real_action_slots']==1 and samples[-1]['real_action_slots']==0,'Terminal masks changed')
    loader=data_loader.create_data_loader(dataclasses.replace(cfg,batch_size=1,num_workers=0),shuffle=False,num_batches=1,framework='jax')
    obs,actions=next(iter(loader))
    require(obs.state.shape==(1,32) and actions.shape==(1,50,32) and obs.action_is_pad.shape==(1,50),'Wrong real loader batch')
    train=module('electric_train',OPENPI/'scripts/train.py')
    require(float(train.masked_action_loss_mean(np.array([[1.,100.,3.]]),np.array([[False,True,False]])))==2.,'Padding not excluded from loss')
    report={'status':'complete','config':CONFIG,'episodes':115,'frames':24182,'verified_videos':checked_videos,
            'state_input':True,'samples':samples,'dataset_contract_sha256':digest(DATASET/'meta/umi_conversion.json'),
            'norm_stats_sha256':digest(ASSET/'norm_stats.json'),'inference_rotation_test':self_test()}
    with (ROOT/'smoke.report.json').open('x') as f: json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
