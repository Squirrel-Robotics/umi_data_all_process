#!/usr/bin/env python3
"""Launch one fresh state/no-state pair on disjoint GPUs; never kill other jobs."""
import fcntl
import json
import os
import subprocess
import time
from experiment import *


def cpu_env():
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',PYTHONPATH=str(OPENPI/'src'),
               OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1',
               XLA_PYTHON_CLIENT_PREALLOCATE='false')
    return env


def snapshot():
    def query(kind,fields):
        text=subprocess.check_output(['nvidia-smi',f'--query-{kind}={fields}','--format=csv,noheader,nounits'],text=True,timeout=30)
        return [[x.strip() for x in line.split(',')] for line in text.splitlines() if line.strip()]
    gpus=query('gpu','index,uuid,memory.used,utilization.gpu')
    require([int(r[0]) for r in gpus]==list(range(8)), 'Expected physical GPUs 0-7')
    apps=query('compute-apps','gpu_uuid,pid,used_gpu_memory')
    return {'gpus':gpus,'apps':apps,'idle':not apps and all(int(r[2])<=1024 and int(r[3])<=5 for r in gpus)}


def active_training():
    found=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit(): continue
        try:
            argv=(entry/'cmdline').read_text().split('\0')
            if str(OPENPI/'scripts/train.py') in argv or ('scripts/train.py' in argv and (entry/'cwd').resolve()==OPENPI):
                found.append(int(entry.name))
        except (OSError,RuntimeError): pass
    return found


def status(phase,**details):
    value={'status':phase,'time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'launcher_pid':os.getpid(),**details}
    path=PACKAGE/'pair.status.json'; temp=PACKAGE/f'.pair.status.{os.getpid()}.tmp'
    with temp.open('w') as f: json.dump(value,f,indent=2)
    os.replace(temp,path)
    print(json.dumps(value),flush=True)


def main():
    outputs=[]
    for job in JOBS:
        outputs += [OPENPI/'logs'/(job['exp']+'.log'), OPENPI/'logs'/(job['exp']+'.launch.json'),
                    Path('/mnt/data/dzq/openpi/checkpoints')/job['config']/job['exp']]
    outputs += [PACKAGE/'preflight.report.json',PACKAGE/'preflight.log',PACKAGE/'pair.status.json']
    with (PACKAGE/'.pair.lock').open('a+') as pair_lock:
        fcntl.flock(pair_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not any(p.exists() for p in outputs),'Fresh pair output already exists; no overwrite/resume allowed')
        status('cpu_preflight',jobs=JOBS)
        with (PACKAGE/'preflight.log').open('x') as output:
            subprocess.run([str(PYTHON),str(PACKAGE/'preflight.py')],cwd=OPENPI,env=cpu_env(),
                           stdout=output,stderr=subprocess.STDOUT,check=True,timeout=1800)
        report=read_json(PACKAGE/'preflight.report.json')
        require(report['status']=='complete','Preflight failed')
        with (OPENPI/'logs/.openpi_training_launch.lock').open('a+') as global_lock:
            fcntl.flock(global_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for _ in range(3):
                current=snapshot()
                require(current['idle'] and not active_training(),'Another process is using GPUs or initializing training')
                time.sleep(2)
            for path,sha in report['fingerprints'].items():
                require(digest(path)==sha,f'Preflight input changed: {path}')
            require(not any(p.exists() for p in outputs[:6]),'Experiment outputs appeared before launch')
            processes=[]
            for job in JOBS:
                env=os.environ.copy()
                env.pop('JAX_PLATFORMS',None); env.pop('JAX_PLATFORM_NAME',None)
                env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,job['gpus'])),CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    PYTHONPATH=str(OPENPI/'src'),PYTHONUNBUFFERED='1',XLA_PYTHON_CLIENT_PREALLOCATE='true',
                    XLA_PYTHON_CLIENT_MEM_FRACTION='0.90')
                command=[str(PYTHON),str(OPENPI/'scripts/train.py'),job['config'],'--exp-name',job['exp']]
                log=OPENPI/'logs'/(job['exp']+'.log')
                with log.open('x') as stream:
                    process=subprocess.Popen(command,cwd=OPENPI,env=env,stdin=subprocess.DEVNULL,
                        stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                raw=Path(f'/proc/{process.pid}/stat').read_text()
                receipt={**job,'pid':process.pid,'process_start_ticks':int(raw[raw.rfind(')')+2:].split()[19]),
                    'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'command':command,'cwd':str(OPENPI),
                    'log':str(log),'checkpoint_root':str(Path('/mnt/data/dzq/openpi/checkpoints')/job['config']/job['exp']),
                    'mode':'fresh','batch_size':32,'num_workers':32,'num_train_steps':10000,'save_interval':1000,
                    'memory_fraction':0.90,'dataset':str(DATASET),'preflight_report':str(PACKAGE/'preflight.report.json'),
                    'preflight_sha256':digest(PACKAGE/'preflight.report.json')}
                with (OPENPI/'logs'/(job['exp']+'.launch.json')).open('x') as f: json.dump(receipt,f,indent=2)
                processes.append((job,process,receipt))
            status('training_initializing',jobs=[r for _,_,r in processes])
            deadline=time.monotonic()+900
            while True:
                current=snapshot(); uuid_by_index={int(r[0]):r[1] for r in current['gpus']}
                claimed={}
                for job,process,receipt in processes:
                    require(process.poll() is None,f'{job["variant"]} exited during startup; see {receipt["log"]}')
                    actual={r[0] for r in current['apps'] if r[1]==str(process.pid)}
                    expected={uuid_by_index[i] for i in job['gpus']}
                    require(actual<=expected,f'{job["variant"]} claimed unassigned GPUs')
                    claimed[job['variant']]=len(actual)
                if all(n==4 for n in claimed.values()):
                    status('both_processes_running',jobs=[r for _,_,r in processes],claimed_gpu_counts=claimed)
                    return
                if time.monotonic()>deadline:
                    status('startup_needs_inspection',claimed_gpu_counts=claimed,jobs=[r for _,_,r in processes])
                    return
                time.sleep(10)


if __name__=='__main__':
    try: main()
    except Exception as error:
        # Preserve any already started job and all artifacts; never silently retry.
        status('failed',error=f'{type(error).__name__}: {error}')
        raise
