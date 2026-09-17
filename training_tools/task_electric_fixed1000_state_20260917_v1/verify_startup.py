"""One-shot, read-only verification of this restart; never retry or stop training."""
import math
import os
import re
import subprocess
import time
from experiment import *

def status(phase,**details):
    value={'status':phase,'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),**details}
    temp=ROOT/f'.startup_verification.{os.getpid()}.tmp'
    with temp.open('w') as f:json.dump(value,f,indent=2)
    os.replace(temp,ROOT/'startup_verification.json')
    print(json.dumps(value),flush=True)

def main():
    require(not (ROOT/'startup_verification.json').exists(),'Startup verification already began')
    receipt=OPENPI/'logs'/(EXP+'.launch.json');queue=OPENPI/'logs'/(EXP+'.queue.json')
    deadline=time.monotonic()+3600;previous=None;max_workers=0
    status('waiting_for_training_launcher',config=CONFIG)
    while time.monotonic()<deadline:
        q=read_json(queue) if queue.exists() else {}
        require(q.get('status')!='failed',f'Launcher failed: {q.get("error")}')
        if receipt.exists():
            r=read_json(receipt);pid=r['pid'];p=Path(f'/proc/{pid}')
            require(p.exists(),'Training exited before the first step; inspect its log')
            raw=(p/'stat').read_text();f=raw[raw.rfind(')')+2:].split()
            require(int(f[19])==r['process_start_ticks'] and f[0] not in ('Z','X'),'Training PID changed or exited')
            cfg=r['evidence']['config']
            require(cfg['num_workers']==128 and cfg['batch_size']==64 and cfg['discrete_state_input'] is True,'Wrong launch settings')
            children=subprocess.run(['ps','--ppid',str(pid),'-o','args='],capture_output=True,text=True,check=False).stdout
            workers=sum('multiprocessing.spawn' in line and 'spawn_main' in line for line in children.splitlines())
            max_workers=max(max_workers,workers)
            log=OPENPI/'logs'/(EXP+'.log')
            if log.exists():
                with log.open('rb') as stream:
                    stream.seek(max(0,log.stat().st_size-24000));text=stream.read().decode(errors='replace')
                require('Traceback (most recent call last)' not in text,'Training traceback; inspect log')
                matches=re.findall(r'Step (\d+):[^\n]*loss=([^,\s]+)',text)
                if matches:
                    step,value=matches[-1];loss=float(value)
                    require(math.isfinite(loss),'Non-finite startup loss')
                    require(max_workers==128,'First step reached without observing all 128 workers')
                    if int(step)>=50:
                        require(loss<5.,'First 50-step loss remains above the numeric-scale sanity threshold (5); inspect before continuing')
                        status('first_50_steps_verified',pid=pid,observed_workers=max_workers,
                               step=int(step),loss=loss,config=CONFIG,log=str(log));return
            milestone=(pid,workers//32,workers==128)
            if milestone!=previous:
                status('initializing_workers_or_model',pid=pid,observed_workers=workers,
                       configured_workers=128,log=str(log));previous=milestone
        time.sleep(10)
    raise TimeoutError('First step not observed within one hour; training left untouched')

if __name__=='__main__':
    try:main()
    except Exception as e:
        status('needs_attention',error=f'{type(e).__name__}: {e}',training_was_not_modified=True)
        raise
