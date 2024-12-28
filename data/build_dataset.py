import os

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .train_dataset import Med_SAM_Dataset  # Import Med_SAM_Dataset
from .collect_fn import collect_fn
from .CTDataset import CTDataset  # Import CTDataset
import json  # To load JSON configuration


def build_dataset(args):
    """
    Builds the dataset, sampler, and dataloader based on the specified dataset name in args.
    Supports CT-RATE-Text Reports based on Images and CT-Rate-Images.
    """
    # Load the dataset-specific configuration if provided
    # if args.dataset_config_path:
    #     with open(args.dataset_config_path, 'r') as f:
    #         dataset_config = json.load(f)
    # else:
    #     dataset_config = {}
    if args.dataset_name == "CT-RATE-Text":
        # Build CTDataset
        label_columns = ["VolumneName", "ClinicalInformation_EN", "Technique_EN", "Findings_EN", "Impressions_EN"]
        max_token_length = 512
        dataset = CT_Rate_Text(
            data_files= "Users/shikhakathrani/SAT/data/train_reports.csv"  # Input data files (e.g., pandas DataFrame)
            class_count= len(label_columns),
            label_cols=label_columns,  # Label columns
            max_length= max_token_length,  # Max tokenization length
            augment=dataset_config.get("augment", args.augment),  # Enable text augmentation
            infer= False,  # Whether in inference mode
        )
    elif args.dataset_name == "CT-RATE-Images":
        dataset = CT_RATE_Images_Dataset(
            data_dir= "Users/shikhakathrani/SAT/data/convertedToDicom",
            
            preprocess_config=args.preprocess_config,
            crop_size=args.crop_size,
            augmentations=args.augmentations
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    # Create a sampler for distributed training
    sampler = DistributedSampler(dataset)

    # Build the DataLoader
    if args.num_workers is not None:
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=args.batch_size,
            collate_fn=collect_fn,
            pin_memory=args.pin_memory,
            num_workers=args.num_workers,
        )
    else:
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=args.batch_size,
            collate_fn=collect_fn,
            pin_memory=args.pin_memory,
        )

    return dataset, dataloader, sampler
