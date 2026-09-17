"""Install an append-only config variant after verifying the exact predecessor."""
import ast
import os
import shutil
from experiment import *

target=OPENPI/'src/openpi/training/config.py'
candidate=ROOT/'config.fixed1000.py'
require(digest(target)=='36f74e116e253a4a11f03b859e890e58a678e71f7d4df8ae046e2d1e61d9f792','Config changed; inspect before installation')
old=ast.parse(target.read_text());new=ast.parse(candidate.read_text())
def is_new(node):
    if isinstance(node,ast.Assign):
        return any(isinstance(t,ast.Name) and t.id=='_electric_fixed1000_base' for t in node.targets)
    return (isinstance(node,ast.Expr) and isinstance(node.value,ast.Call)
        and any(isinstance(n,ast.Constant) and n.value==CONFIG for n in ast.walk(node)))
new.body=[n for n in new.body if not is_new(n)]
require(ast.dump(old)==ast.dump(new),'Unexpected non-append config changes')
backup=ROOT/'backups';backup.mkdir(exist_ok=False)
shutil.copy2(target,backup/'config.before_fixed1000.py')
temporary=target.with_name(f'.config.fixed1000.{os.getpid()}.tmp')
with temporary.open('xb') as f:f.write(candidate.read_bytes())
temporary.chmod(target.stat().st_mode&0o777)
require(digest(target)==digest(backup/'config.before_fixed1000.py'),'Concurrent config change')
os.replace(temporary,target)
print(json.dumps({'status':'installed','config':CONFIG,'sha256':digest(target),'historical_configs_preserved':True}))
