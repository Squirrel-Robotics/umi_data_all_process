"""Start exactly one guarded launcher and one bounded read-only startup check."""
import os
import subprocess
import time
from experiment import *

for name in ('launcher.log','launcher.start.json','startup_verification.json'):
    require(not (ROOT/name).exists(),f'Dispatch already attempted: {name}')
require(read_json(ROOT/'config_change_validation.json')['status']=='passed','Config validation missing')
require((ASSET/'norm_stats.json').is_file(),'Fixed-bound asset missing')
with (ROOT/'launcher.log').open('x') as out:
    p=subprocess.Popen(['bash',str(ROOT/'start_training.sh')],cwd=ROOT,stdin=subprocess.DEVNULL,
                       stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
raw=Path(f'/proc/{p.pid}/stat').read_text();ticks=int(raw[raw.rfind(')')+2:].split()[19])
with (ROOT/'startup_verification.log').open('x') as out:
    check=subprocess.Popen([str(PYTHON),str(ROOT/'verify_startup.py')],cwd=ROOT,stdin=subprocess.DEVNULL,
                       stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
record={'status':'dispatched','queue_pid':p.pid,'queue_start_ticks':ticks,'verification_pid':check.pid,
        'config':CONFIG,'exp':EXP,'time':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
with (ROOT/'launcher.start.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record,indent=2))
