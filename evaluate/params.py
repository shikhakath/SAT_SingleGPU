import argparse

def str2bool(v):
    return v.lower() in ('true', 't')

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Exp Controller
    
    parser.add_argument(
        "--rcd_dir",
        type=str,
        default = './demoResults',
        help="save the evaluation results (in a directory)",
    )
    parser.add_argument(
        "--rcd_file",
        type=str,
        help="save the evaluation results (in a csv/xlsx file)",
    )
    parser.add_argument(
        "--visualization",
        type=str2bool,
        default=False,
        help="save the visualization for each case (img, gt, pred)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default = "/content/drive/My Drive/SAT_LFG/UNET-BaseBERT/SAT_Nano_UNET.pth",
        help="Checkpoint path",
    )
    parser.add_argument(
        "--partial_load",
        type=str2bool,
        default=True,
        help="Allow to load partial paramters from checkpoint",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--resume",
        type=str2bool,
        default=True,
        help="Inherit medial results from an interrupted evaluation (no harm even if you evaluate from scratch)",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100
    )
    
    # Metrics
    
    parser.add_argument(
        "--dice",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--nsd",
        type=str2bool,
        default=True,
    )
    
    # Med SAM Dataset
    
    parser.add_argument(
        "--datasets_jsonl",
        type=str,
        default="SAT/demo/inference_demo/demo.jsonl",  # Relative path from the current directory
        help="Path to the JSONL file containing inference dataset",
    )
    
    # Sampler and Loader
    
    parser.add_argument(
        "--online_crop",
        type=str2bool,
        default='False',
        help='load pre-cropped image patches directly, or crop online',
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        nargs='+',
        default=[288, 288, 96],
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--batchsize_3d",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--pin_memory",
        type=str2bool,
        default=False,
        help='load data to gpu to accelerate'
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4
    )
    
    # Knowledge Encoder
    parser.add_argument(
        "--text_encoder_partial_load",
        type=str2bool,
        default=True,
        help="Allow to load partial paramters from checkpoint",
    )
    parser.add_argument(
        "--text_encoder_checkpoint",
        type=str,
        default="/content/drive/My Drive/SAT_LFG/UNET-BaseBERT/BaseBERT.pth"
    )
    parser.add_argument(
        "--text_encoder",
        type=str,
        default="basebert",
        help="Specify the text encoder to use ('ours', 'medcpt', 'basebert')."
    )

    # MaskFormer
    
    parser.add_argument(
        "--vision_backbone",
        type=str,
        default = "UNET",
        help='UNET UNET-L UMamba or SwinUNETR'
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        nargs='+',
        default=[32, 32, 32],
        help='patch size on h w and d'
    )
    parser.add_argument(
        "--deep_supervision",
        type=str2bool,
        default=False,
    )
    
    args = parser.parse_args()
    return args
