"""One-time recovery of the stopped, unpublished electric conversion. No deletions."""
import argparse
import os
import shutil
import subprocess
import time
from experiment import *

OLD_PID = 3515137
OLD_TICKS = 98378019
STAGE = DATASET.parent / '.task_electric_lerobot_10hz_h50_right_eye_wrist180_v1.building-k5tf4xre'
ARCHIVED_STAGE = DATASET.parent / '.task_electric_lerobot_10hz_h50_right_eye_wrist180_v1.abandoned-wrong-camera-map-k5tf4xre'
BACKUP = ROOT / 'backups/before_camera_mapping_fix'

def identity(pid):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
        return int(raw[raw.rfind(')')+2:].split()[19])
    except FileNotFoundError:
        return None

def prepare():
    require(identity(OLD_PID) != OLD_TICKS and not Path('/proc/3515910').exists(), 'Previous processes must be stopped')
    active = subprocess.check_output(['ps','-eo','pid,pgid,args'], text=True)
    require(not any(len(v:=line.split())>=2 and v[1]==str(OLD_PID) for line in active.splitlines()[1:]), 'Conversion children remain')
    receipt = read_json(ROOT/'conversion.launch.json')
    require(receipt['pid']==OLD_PID and receipt['process_start_ticks']==OLD_TICKS and receipt['target']==str(DATASET),'Unexpected old receipt')
    require(read_json(ROOT/'pipeline.status.json')['status']=='waiting_for_conversion','Pipeline went beyond conversion')
    require(not DATASET.exists() and not ASSET.exists(),'Dataset or norm asset already published')
    require(STAGE.is_dir() and not STAGE.is_symlink() and not ARCHIVED_STAGE.exists(),'Unexpected staging paths')
    require(not BACKUP.exists(),'Recovery backup already exists')
    require(digest(ROOT/'converter_electric.py')=='9fe998a742d01406c7b12c824dfc79687bcbdbcc8dfaa530cbfd8149b07136eb','Old converter changed')
    require(digest(SOURCE/'pass_sessions.txt')==SELECTION_SHA,'Approved source selection changed')
    for suffix in ('.log','.launch.json','.smoke.log'):
        require(not (OPENPI/'logs'/(EXP+suffix)).exists(),'Training artifacts already exist')
    BACKUP.mkdir()
    for name in ('converter_electric.py','smoke_training.py','launch_training.py','continue_pipeline.py'):
        shutil.copy2(ROOT/name,BACKUP/name)
    for name in ('conversion.log','conversion.launch.json','pipeline.log','pipeline.status.json'):
        os.rename(ROOT/name,BACKUP/name)
    os.rename(STAGE,ARCHIVED_STAGE)
    print(json.dumps({'status':'preserved_before_fix','stage':str(ARCHIVED_STAGE),'backup':str(BACKUP)}),flush=True)

def start():
    require(BACKUP.is_dir() and ARCHIVED_STAGE.is_dir(),'Recovery backup missing')
    require(not DATASET.exists() and not ASSET.exists(),'Final outputs already exist')
    require(not list(DATASET.parent.glob(f'.{DATASET.name}.building-*')),'Unfinished staging remains')
    for name in ('conversion.log','conversion.launch.json','pipeline.log','pipeline.status.json'):
        require(not (ROOT/name).exists(),'Process artifacts exist; inspect before retry')
    import converter_electric
    require(converter_electric.VIDEO_KEYS=={'observation.images.head_rgb':'head',
        'observation.images.left_wrist_rgb':'cam1','observation.images.right_wrist_rgb':'cam0'},'Video mapping not corrected')
    proof=read_json(ROOT/'camera_mapping_fix_validation.json')
    require(proof['status']=='complete' and proof['numeric_frames_compared']==24182
            and proof['new_generator_sha256']==digest(ROOT/'converter_electric.py'),'Missing numeric equivalence proof')
    previous=read_json(BACKUP/'conversion.launch.json')
    command=previous['command']
    require(command[command.index('--video-workers')+1]=='16' and command[command.index('--target')+1]==str(DATASET),'Wrong command')
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    with (ROOT/'conversion.log').open('x') as output:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=output,
                               stderr=subprocess.STDOUT,start_new_session=True)
    receipt={'pid':child.pid,'process_start_ticks':identity(child.pid),'command':command,'target':str(DATASET),
        'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'source_selection_sha256':SELECTION_SHA,
        'generator_files':{n:digest(ROOT/n) for n in ('converter_electric.py','publish_repaired_dataset.py')},
        'camera_acquisition_mapping':{'cam0':'right_wrist','cam1':'left_wrist'},'numeric_hand_channels_swapped':False,
        'previous_stage_preserved':str(ARCHIVED_STAGE),'previous_process_artifacts':str(BACKUP),
        'numeric_equivalence_proof':str(ROOT/'camera_mapping_fix_validation.json')}
    with (ROOT/'conversion.launch.json').open('x') as f:json.dump(receipt,f,indent=2)
    with (ROOT/'pipeline.log').open('x') as output:
        pipeline=subprocess.Popen([str(PYTHON),'-u','continue_pipeline.py'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                  stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps({'conversion':receipt,'pipeline_pid':pipeline.pid}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('prepare','start'))
    {'prepare':prepare,'start':start}[p.parse_args().phase]()
