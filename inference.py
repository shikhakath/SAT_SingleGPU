import os
import datetime
import random

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / "model"))

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path

from data.inference_dataset import Inference_Dataset, collate_fn

from model.build_model import build_maskformer, load_checkpoint
from model.text_encoder import Text_Encoder

from train.dist import is_master

from evaluate.inference_engine import inference
from evaluate.params import parse_args

def set_seed(config):
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    # new seed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def main(args):
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    local_rank = int(os.environ.get("LOCAL_RANK", 0))  # Default to 0 if not set
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.set_device(local_rank)

    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        torch.distributed.init_process_group(backend="nccl", init_method='env://', timeout=datetime.timedelta(seconds=7200))
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        print(f"Distributed initialized on rank {rank}")
    else:
        rank = 0
        world_size = 1
        print("Running in single GPU mode.")
  
    # torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    # device=torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    # gpu_id = int(os.environ["LOCAL_RANK"])
    # torch.distributed.init_process_group(backend="nccl", init_method='env://', timeout=datetime.timedelta(seconds=7200))   # might takes a long time to sync between process
    
    # Display information
    if rank == 0:  # Master node or single GPU mode
        print('** GPU NUM ** : ', torch.cuda.device_count())
        print('** WORLD SIZE ** : ', world_size)
        print("** DDP ** : Start running in single GPU mode or master rank.")

    # Create the directory for saving results
    if rank == 0:
        Path(args.rcd_dir).mkdir(exist_ok=True, parents=True)
        print(f'Inference Results will be Saved to ** {args.rcd_dir} **')
    
    # Dataset and DataLoader
    testset = Inference_Dataset(args.datasets_jsonl, args.max_queries, args.batchsize_3d)

    if world_size > 1:
        # Distributed mode
        sampler = DistributedSampler(testset)
        testloader = DataLoader(testset, sampler=sampler, batch_size=1, pin_memory=args.pin_memory, num_workers=args.num_workers, collate_fn=collate_fn)
        sampler.set_epoch(0)
    else:
        # Single GPU mode
        testloader = DataLoader(testset, batch_size=1, pin_memory=args.pin_memory, num_workers=args.num_workers, collate_fn=collate_fn)
    
    # set model
    gpu_id = getattr(args, 'gpu_id', None)
    if gpu_id is None:
        gpu_id = local_rank
    model = build_maskformer(args, device, gpu_id)
    
    # load knowledge encoder
    text_encoder = Text_Encoder(
        text_encoder=args.text_encoder,
        checkpoint=args.text_encoder_checkpoint,
        partial_load=args.text_encoder_partial_load,
        open_bert_layer=12,
        open_modality_embed=False,
        gpu_id=gpu_id,
        device=device
    )
    
    # load checkpoint if specified
    model, _, _ = load_checkpoint(
        checkpoint=args.checkpoint,
        resume=False,
        partial_load=args.partial_load,
        model=model, 
        device=device
    )
    
    # choose how to evaluate the checkpoint
    inference(model=model,
              text_encoder=text_encoder,
              device=device,
              testset=testset,
              testloader=testloader,
              nib_dir=args.rcd_dir)

if __name__ == '__main__':
    # get configs
    args = parse_args()
    
    main(args)

    
    
    
        
    
    