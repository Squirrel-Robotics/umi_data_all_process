from pathlib import Path
import hashlib
import json

OPENPI = Path('/home/dzq/openpi')
PYTHON = OPENPI / '.venv/bin/python'
PACKAGE = Path(__file__).resolve().parent
BASE_TOOLS = Path('/home/dzq/data_deal/training_tools/task_v2_x2_high_wrist180_no_state_v1')
BASE_CONFIG = 'pi05_umi_task_v2_x2_high_10hz_h50_wrist180_no_state_10k_b64_w32_8gpu_v1'
BASE_EXP = 'task_v2_x2_high_wrist180_no_state_10hz_h50_10k_b64_w32_8gpu_20260911_v1'
DATASET = Path('/mnt/data/dzq/umi_v2/datasets/task_v2_x2_high_lerobot_10hz_h50_right_eye_v2_wrist180')
ASSET = Path('/mnt/data/dzq/openpi/data/assets/umi_task_v2_x2_high_hand_pose_10hz_h50_right_eye_wrist180_masked_v1')
JOBS = [dict(variant=variant, state=state, gpus=gpus,
    config=f'pi05_umi_task_v2_x2_high_10hz_h50_wrist180_{variant}_10k_b32_w32_4gpu_v1',
    exp=f'task_v2_x2_high_wrist180_{variant}_10hz_h50_10k_b32_w32_4gpu_20260911_v1')
    for variant, state, gpus in [('no_state', False, [0,1,2,3]), ('state', True, [4,5,6,7])]]

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(4*1024*1024),b''): h.update(part)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text())

def require(condition, message):
    if not condition: raise ValueError(message)
