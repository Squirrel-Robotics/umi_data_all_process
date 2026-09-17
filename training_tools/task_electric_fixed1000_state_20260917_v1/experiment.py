from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
OPENPI = Path('/home/dzq/openpi')
PYTHON = OPENPI/'.venv/bin/python'
SOURCE = Path('/mnt/data/dzq/umi_v2/data/task_electric')
DATASET = Path('/mnt/data/dzq/umi_v2/datasets/task_electric_lerobot_10hz_h50_right_eye_wrist180_v1')
ASSET_ID = 'umi_task_electric_10hz_h50_right_eye_wrist180_fixed1000_v1'
ASSET = Path('/mnt/data/dzq/openpi/data/assets')/ASSET_ID
CONFIG = 'pi05_umi_task_electric_10hz_h50_wrist180_state_fixed1000_20k_b64_w128_8gpu_v1'
EXP = 'task_electric_right_eye_wrist180_state_fixed1000_10hz_h50_20k_b64_w128_8gpu_20260917_v1'
PROMPT = 'Insert the battery into the empty slot in the middle.'
SELECTION_SHA = '0145c1b2005d0a7d93f02e465dba74d156fc9c01474d55ae15712f32b362aa16'
NORM_TOOL = ROOT/'build_fixed_norm.py'
BASE_NORM_TOOL = Path('/home/dzq/data_deal/training_tools/compute_masked_norm_stats.py')
BASE_ASSET = ASSET.parent/'umi_task_electric_10hz_h50_right_eye_wrist180_masked_v1'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def read_json(path): return json.loads(Path(path).read_text())

def require(condition,message):
    if not condition: raise ValueError(message)
