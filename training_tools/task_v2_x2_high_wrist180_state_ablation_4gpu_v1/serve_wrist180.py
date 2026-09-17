#!/usr/bin/env python3
"""Matching inference entry for either four-GPU wrist180 state ablation."""
import argparse
import dataclasses
from experiment import *


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=('state','no_state'),required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--input-orientation',choices=('raw','rotated180'),required=True)
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=8000)
    args=p.parse_args()
    from openpi import transforms
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config
    from wrist180_inference import RotateRawWristImages, self_test
    self_test()
    job=next(j for j in JOBS if j['variant']==args.variant)
    cfg=config.get_config(job['config'])
    checkpoint=args.checkpoint.resolve(strict=True)
    root=(Path(cfg.checkpoint_base_dir)/job['config']/job['exp']).resolve()
    require(checkpoint.parent==root and checkpoint.name.isdigit(),'Checkpoint must belong to the selected variant')
    require(all((checkpoint/n).exists() for n in ('params','assets','train_state')),'Incomplete checkpoint')
    require(cfg.model.discrete_state_input is job['state'],'State mode mismatch')
    require(cfg.policy_metadata['wrist_rotation_degrees']=={'left':180,'right':180},'Wrist orientation mismatch')
    cfg=dataclasses.replace(cfg,policy_metadata={**cfg.policy_metadata,'server_input_orientation':args.input_orientation,
        'server_applies_wrist_rotation':args.input_orientation=='raw'})
    repack=transforms.Group(inputs=[RotateRawWristImages()] if args.input_orientation=='raw' else [])
    policy=policy_config.create_trained_policy(cfg,str(checkpoint),repack_transforms=repack)
    websocket_policy_server.WebsocketPolicyServer(policy,host=args.host,port=args.port,metadata=policy.metadata).serve_forever()


if __name__=='__main__': main()
