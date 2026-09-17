#!/usr/bin/env python3
"""Finish this existing conversion, compute fresh stats, validate and launch once."""
import fcntl
import json
import os
import subprocess
import time
from experiment import *


def status(phase,**details):
    value={'status':phase,'pid':os.getpid(),'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),**details}
    temp=ROOT/f'.pipeline.{os.getpid()}.tmp'
    with temp.open('w') as f: json.dump(value,f,indent=2)
    os.replace(temp,ROOT/'pipeline.status.json')
    print(json.dumps(value),flush=True)


def cpu_env():
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',PYTHONPATH=str(OPENPI/'src'),
               XLA_PYTHON_CLIENT_PREALLOCATE='false',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    return env


def main():
    with (ROOT/'.pipeline.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not (ROOT/'pipeline.status.json').exists(),'Pipeline already started; inspect before retry')
        require(not ASSET.exists(),'Fresh normalization asset already exists')
        require(digest(SOURCE/'pass_sessions.txt')==SELECTION_SHA,'Approved selection changed')
        receipt=read_json(ROOT/'conversion.launch.json')
        require(receipt['target']==str(DATASET),'Wrong conversion receipt')
        status('waiting_for_conversion',conversion_pid=receipt['pid'])
        deadline=time.monotonic()+6*3600
        while not DATASET.exists():
            process=Path(f'/proc/{receipt["pid"]}')
            require(process.exists(),'Conversion exited before publishing; inspect conversion.log; no automatic restart')
            raw=(process/'stat').read_text()
            require(int(raw[raw.rfind(')')+2:].split()[19])==receipt['process_start_ticks'],'Conversion PID reused')
            require(time.monotonic()<deadline,'Conversion wait exceeded six hours')
            time.sleep(10)
        # The publication reservation can briefly exist before atomic rename.
        for _ in range(12):
            if (DATASET/'meta/umi_conversion.json').exists() and not list(DATASET.parent.glob(f'.{DATASET.name}.building-*')): break
            time.sleep(5)
        require(not list(DATASET.parent.glob(f'.{DATASET.name}.building-*')),'Conversion staging remains; refusing training')
        contract=read_json(DATASET/'meta/umi_conversion.json')
        require(contract['status']=='complete' and contract['target']==str(DATASET)
                and contract['task']==PROMPT and contract['total_episodes']==115 and contract['total_frames']==24182,'Unexpected published dataset')
        require(contract['video_keys']=={'observation.images.head_rgb':'head',
            'observation.images.left_wrist_rgb':'cam1','observation.images.right_wrist_rgb':'cam0'},'Incorrect wrist camera assignment')
        require(contract['video_preprocessing']['numeric_hand_channels_swapped'] is False,'Numeric hands unexpectedly swapped')
        for name,sha in receipt['generator_files'].items():
            require(digest(ROOT/name)==sha and digest(DATASET/'meta'/name)==sha,'Conversion code changed')
        require(digest(DATASET/'meta/episode_selection.txt')==SELECTION_SHA,'Published selection changed')
        evidence=digest(DATASET/'meta/umi_conversion.json')
        status('computing_fresh_normalization',dataset=str(DATASET),episodes=115,frames=24182)
        command=[str(PYTHON),str(NORM_TOOL),'--dataset',str(DATASET),'--asset-id',ASSET_ID,
                 '--expected-fps','10','--expected-horizon','50','--expected-task',PROMPT]
        with (ROOT/'normalization.log').open('x') as output:
            subprocess.run(command,cwd=OPENPI,env=cpu_env(),stdout=output,stderr=subprocess.STDOUT,check=True,timeout=1800)
        require(digest(DATASET/'meta/umi_conversion.json')==evidence,'Dataset changed during normalization')
        status('cpu_smoke_then_eight_gpu_queue',dataset_contract_sha256=evidence)
        subprocess.run(['bash',str(ROOT/'start_training.sh')],cwd=ROOT,check=True)
        status('training_launched',log=str(OPENPI/'logs'/(EXP+'.log')),config=CONFIG,exp_name=EXP)


if __name__=='__main__':
    try: main()
    except Exception as error:
        status('failed',error=f'{type(error).__name__}: {error}',note='No dataset/checkpoint deletion; no automatic retry')
        raise
