"""Data loader for QAFC-PhysFormer training.

Extends STVENLoader to support:
- Quality pair sampling (two different CRF levels for same video)
- Ranking loss data preparation

Returns batches as tuple for compatibility with base class:
- video_high: Higher quality video (lower CRF)
- video_low: Lower quality video (higher CRF)
- bvp_label: Ground truth BVP signal
- crf_info: Dict with 'crf_current' and 'crf_paired'
"""

import os
import re
import random
import numpy as np
import torch
from dataset.data_loader.STVENLoader import STVENLoader


class QAFCPhysFormerLoader(STVENLoader):
    """
    Data loader for QAFC-PhysFormer training.

    Extends STVENLoader to support quality pair sampling for ranking loss.
    For each video, samples a paired CRF level and returns both versions.
    """

    def __init__(self, name, data_path, config_data, device=None):
        """Initializes QAFC-PhysFormer dataloader.

        Args:
            name (str): Name of the dataloader
            data_path (str): Path to dataset (ignored, uses CRF_DATASETS)
            config_data (CfgNode): Data configuration
            device (torch.device): Device for data loading
        """
        super().__init__(name, data_path, config_data, device)

        # Validate CRF_LEVELS is configured
        if not hasattr(self.config_data, 'CRF_LEVELS') or not self.config_data.CRF_LEVELS:
            raise ValueError("QAFCPhysFormerLoader requires CRF_LEVELS in config")

        self.crf_levels = self.config_data.CRF_LEVELS

    def __getitem__(self, index):
        """
        Returns a batch with quality pairs for ranking loss.

        For each video at current CRF, samples a different CRF and returns:
        - video_high: Higher quality video (lower CRF)
        - video_low: Lower quality video (higher CRF)
        - bvp_label: Ground truth BVP signal
        - crf_current: Current CRF level (int)
        - crf_paired: Paired CRF level (int)
        """
        # Load current video data
        compressed_data = np.load(self.inputs[index])

        # Handle data format
        if self.data_format == 'NDCHW':
            compressed_data = np.transpose(compressed_data, (0, 3, 1, 2))
        elif self.data_format == 'NCDHW':
            compressed_data = np.transpose(compressed_data, (3, 0, 1, 2))
        elif self.data_format == 'NDHWC':
            pass
        else:
            raise ValueError('Unsupported Data Format!')

        compressed_data = np.float32(compressed_data)

        # Parse current CRF from filename
        item_path = self.inputs[index]
        item_path_filename = item_path.split(os.sep)[-1]

        match = re.search(r'_crf(\d+)_', item_path_filename)
        if match:
            current_crf = int(match.group(1))
        else:
            raise ValueError(f"Could not parse CRF level from filename: {item_path_filename}")

        # Sample a different CRF level for ranking pair
        other_crfs = [c for c in self.crf_levels if c != current_crf]
        if not other_crfs:
            # Fallback if only one CRF level available
            paired_crf = current_crf
        else:
            paired_crf = random.choice(other_crfs)

        # Load paired video
        paired_filename_base = item_path_filename.replace(f"_crf{current_crf}_", f"_crf{paired_crf}_")
        paired_path = os.path.join(self.cached_path, paired_filename_base)

        if os.path.exists(paired_path):
            paired_data = np.load(paired_path)
            # Apply same format transformation
            if self.data_format == 'NDCHW':
                paired_data = np.transpose(paired_data, (0, 3, 1, 2))
            elif self.data_format == 'NCDHW':
                paired_data = np.transpose(paired_data, (3, 0, 1, 2))
            paired_data = np.float32(paired_data)
        else:
            # Fallback: use current video as paired (loss will be zero)
            print(f"Warning: Paired CRF {paired_crf} file not found: {paired_path}")
            paired_data = compressed_data.copy()

        # Determine which is higher quality (lower CRF = higher quality)
        if current_crf < paired_crf:
            video_high = compressed_data
            video_low = paired_data
        else:
            video_high = paired_data
            video_low = compressed_data

        # Load BVP label (from current video's label path)
        label_path = self.labels[index]
        bvp_label = np.load(label_path)
        bvp_label = np.float32(bvp_label)

        # Return as tuple for base class compatibility
        # Format: (video_high, video_low, bvp_label, crf_info)
        crf_info = {'crf_current': current_crf, 'crf_paired': paired_crf}
        return video_high, video_low, bvp_label, crf_info
