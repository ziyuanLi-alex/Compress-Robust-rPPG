"""Trainer for QAFC-PhysFormer.

Follows rPPG-Toolbox BaseTrainer pattern with:
- Two-phase training support (separate configs for Phase 1 and Phase 2)
- Phase 1: Joint training of quality branch + FiLM + backbone
- Phase 2: Fine-tune FiLM + backbone (quality branch frozen, loads Phase 1 checkpoint)
- Parameter groups with different learning rates
- Quality ranking loss for self-supervised quality learning
"""

import os
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from neural_methods.trainer.BaseTrainer import BaseTrainer
from neural_methods.model.QAFCPhysFormer import QAFCPhysFormer
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.loss.QAFCPhysFormerLoss import QAFCLoss, QualityRankingLoss
from evaluation.metrics import calculate_metrics


class QAFCPhysFormerTrainer(BaseTrainer):
    """
    Trainer for QAFC-PhysFormer.

    Supports two training modes via TRAIN.QAFC.TRAINING_PHASE config:
    - Phase 1 (TRAINING_PHASE=1): Joint training of quality branch + FiLM + backbone
    - Phase 2 (TRAINING_PHASE=2): Fine-tune FiLM + backbone (quality branch frozen)
      Requires PHASE1_CHECKPOINT path to load Phase 1 weights
    """

    def __init__(self, config, data_loader):
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.frame_rate = config.TRAIN.DATA.FS
        self.config = config

        # Logging
        self.min_valid_loss = None
        self.best_epoch = 0

        # Get training phase (1 or 2)
        self.training_phase = config.TRAIN.QAFC.TRAINING_PHASE

        # Initialize QAFCPhysFormer model
        self.model = QAFCPhysFormer(
            physformer_config=config.MODEL.QAFC_PHYSFORMER,
            quality_spatial_channels=config.MODEL.QAFC_PHYSFORMER.QUALITY_CHANNELS,
            quality_temporal_hidden=config.MODEL.QAFC_PHYSFORMER.HIDDEN_SIZE,
            use_block_pattern=config.MODEL.QAFC_PHYSFORMER.USE_BLOCK_PATTERN,
        ).to(self.device)

        # Load pretrained PhysFormer weights if specified
        self._load_pretrained_physformer(config)

        # Phase 2: Load Phase 1 checkpoint and freeze quality branch
        if self.training_phase == 2:
            self._load_phase1_checkpoint(config)
            self._freeze_quality_branch()
            self.optimizer = self._create_optimizer_phase2(config)
            print(f"Phase 2 initialized: Quality branch frozen, fine-tuning FiLM + backbone")
        else:
            # Phase 1: Normal joint training
            self.optimizer = self._create_optimizer(config)
            print(f"Phase 1 initialized: Joint training of all components")

        # Loss functions
        self.criterion_Pearson = Neg_Pearson()
        self.criterion_Ranking = QualityRankingLoss(margin=config.TRAIN.QAFC.RANKING_MARGIN)

    def _load_pretrained_physformer(self, config):
        """Load pretrained PhysFormer backbone weights."""
        if config.MODEL.QAFC_PHYSFORMER.PRETRAINED_PATH:
            print(f"Loading PhysFormer weights from {config.MODEL.QAFC_PHYSFORMER.PRETRAINED_PATH}")
            phys_state = torch.load(config.MODEL.QAFC_PHYSFORMER.PRETRAINED_PATH, map_location=self.device)
            if 'state_dict' in phys_state:
                phys_state = phys_state['state_dict']
            # Remove module. prefix if present (from DataParallel)
            new_phys_state = {}
            for k, v in phys_state.items():
                if k.startswith('module.'):
                    new_phys_state[k[7:]] = v
                else:
                    new_phys_state[k] = v
            # Load into backbone only
            backbone_keys = {k.replace('quality_aware_backbone.physformer.', ''): v
                           for k, v in new_phys_state.items()
                           if k.startswith('quality_aware_backbone.physformer.')}
            if backbone_keys:
                self.model.quality_aware_backbone.physformer.load_state_dict(backbone_keys, strict=False)
                print(f"Loaded pretrained PhysFormer weights from {config.MODEL.QAFC_PHYSFORMER.PRETRAINED_PATH}")
            else:
                print("Warning: No matching backbone keys found in pretrained weights")
        else:
            print("Warning: No pretrained PhysFormer path provided. Training from scratch.")

    def _create_optimizer(self, config):
        """Create optimizer with parameter groups for Phase 1 (all components trainable)."""
        # Quality branch parameters
        quality_params = list(self.model.quality_spatial_encoder.parameters()) + \
                        list(self.model.quality_temporal_encoder.parameters()) + \
                        list(self.model.quality_head.parameters())
        # FiLM parameters
        film_params = list(self.model.quality_aware_backbone.film_spatial.parameters()) + \
                     list(self.model.quality_aware_backbone.film_temporal.parameters())
        # Backbone parameters
        backbone_params = list(self.model.quality_aware_backbone.physformer.parameters())

        return optim.AdamW([
            {'params': quality_params, 'lr': config.TRAIN.LR},
            {'params': film_params, 'lr': config.TRAIN.LR},
            {'params': backbone_params, 'lr': config.TRAIN.LR * 0.1},  # Backbone learns slower
        ], weight_decay=1e-4)

    def _load_phase1_checkpoint(self, config):
        """Load Phase 1 checkpoint for Phase 2 fine-tuning."""
        phase1_ckpt_path = config.TRAIN.QAFC.PHASE1_CHECKPOINT
        if phase1_ckpt_path and os.path.exists(phase1_ckpt_path):
            print(f"Loading Phase 1 checkpoint from {phase1_ckpt_path}")
            phase1_state = torch.load(phase1_ckpt_path, map_location=self.device)
            self.model.load_state_dict(phase1_state)
            print("Phase 1 checkpoint loaded successfully")
        else:
            raise ValueError(f"Phase 1 checkpoint not found: {phase1_ckpt_path}")

    def _freeze_quality_branch(self):
        """Freeze quality branch parameters for Phase 2 fine-tuning."""
        print("Freezing quality branch for Phase 2 fine-tuning...")
        for param in self.model.quality_spatial_encoder.parameters():
            param.requires_grad = False
        for param in self.model.quality_temporal_encoder.parameters():
            param.requires_grad = False
        for param in self.model.quality_head.parameters():
            param.requires_grad = False
        print("Quality branch frozen")

    def _create_optimizer_phase2(self, config):
        """Create optimizer for Phase 2 (quality branch frozen, FiLM + backbone trainable)."""
        film_params = list(self.model.quality_aware_backbone.film_spatial.parameters()) + \
                     list(self.model.quality_aware_backbone.film_temporal.parameters())
        backbone_params = list(self.model.quality_aware_backbone.physformer.parameters())

        return optim.AdamW([
            {'params': film_params, 'lr': config.TRAIN.LR},
            {'params': backbone_params, 'lr': config.TRAIN.LR * 0.1},
        ], weight_decay=1e-4)

    def train(self, data_loader):
        """Training Loop for QAFC-PhysFormer."""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        mean_training_losses = []
        mean_valid_losses = []

        for epoch in range(self.max_epoch_num):
            print(f"==== QAFC-PhysFormer Epoch: {epoch} (Training Phase {self.training_phase}) ====")
            self.model.train()

            tbar = tqdm(data_loader["train"], ncols=80)
            epoch_losses = []
            epoch_rppg_losses = []
            epoch_ranking_losses = []

            for idx, batch in enumerate(tbar):
                # QAFCPhysFormerLoader returns: video_high, video_low, bvp_label, crf_info
                video_high = batch[0].float().to(self.device)
                video_low = batch[1].float().to(self.device)
                bvp_label = batch[2].float().to(self.device)
                # crf_info = batch[3]  # Not needed for training, used for debugging

                # Forward pass with high quality video
                pred_rppg, quality_score_high, _, _, _ = self.model(video_high, gra_sharp=2.0)

                # Forward pass with low quality video (for ranking loss)
                _, quality_score_low, _, _, _ = self.model(video_low, gra_sharp=2.0)

                # Normalize rPPG prediction and ground truth
                pred_rppg_norm = (pred_rppg - pred_rppg.mean(dim=-1, keepdim=True)) / \
                                pred_rppg.std(dim=-1, keepdim=True)
                bvp_label_norm = (bvp_label - bvp_label.mean(dim=-1, keepdim=True)) / \
                                bvp_label.std(dim=-1, keepdim=True)

                # Calculate rPPG loss (NegPearson)
                loss_rppg = self.criterion_Pearson(pred_rppg_norm, bvp_label_norm)

                # Calculate quality ranking loss
                loss_ranking = self.criterion_Ranking(quality_score_high, quality_score_low)

                # Combined loss (simple weighted sum)
                total_loss = loss_rppg + 0.1 * loss_ranking

                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                epoch_losses.append(total_loss.item())
                epoch_rppg_losses.append(loss_rppg.item())
                epoch_ranking_losses.append(loss_ranking.item())

                tbar.set_description(f"Loss: {total_loss.item():.4f} (rPPG: {loss_rppg.item():.4f}, Rank: {loss_ranking.item():.4f})")

            epoch_avg_loss = np.mean(epoch_losses)
            mean_training_losses.append(epoch_avg_loss)
            print(f"Epoch {epoch} Avg Loss: {epoch_avg_loss:.4f} (rPPG: {np.mean(epoch_rppg_losses):.4f}, Ranking: {np.mean(epoch_ranking_losses):.4f})")

            self.save_model(epoch)

            # Validation
            self.current_epoch = epoch
            valid_loss = self.valid(data_loader)
            if valid_loss is not None:
                mean_valid_losses.append(valid_loss)

        # Print best epoch summary
        if not self.config.TEST.USE_LAST_EPOCH:
            print("Best trained epoch: {}, min_val_loss: {}".format(
                self.best_epoch, self.min_valid_loss))

        # Plot and save loss history
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, [], self.config)

    def valid(self, data_loader):
        """Validation Loop"""
        if data_loader["valid"] is None:
            print("No data for valid")
            return None

        print("==== QAFC-PhysFormer Validation ====")
        self.model.eval()
        valid_loss = []

        with torch.no_grad():
            tbar = tqdm(data_loader["valid"], ncols=80)
            for idx, batch in enumerate(tbar):
                video_high = batch[0].float().to(self.device)
                video_low = batch[1].float().to(self.device)
                bvp_label = batch[2].float().to(self.device)

                # Forward pass
                pred_rppg, quality_score_high, _, _, _ = self.model(video_high, gra_sharp=2.0)
                _, quality_score_low, _, _, _ = self.model(video_low, gra_sharp=2.0)

                # Normalize
                pred_rppg_norm = (pred_rppg - pred_rppg.mean(dim=-1, keepdim=True)) / \
                                pred_rppg.std(dim=-1, keepdim=True)
                bvp_label_norm = (bvp_label - bvp_label.mean(dim=-1, keepdim=True)) / \
                                bvp_label.std(dim=-1, keepdim=True)

                # Calculate loss
                loss_rppg = self.criterion_Pearson(pred_rppg_norm, bvp_label_norm)
                loss_ranking = self.criterion_Ranking(quality_score_high, quality_score_low)
                loss = loss_rppg + loss_ranking  # Simple sum for validation

                valid_loss.append(loss.item())

                tbar.set_description(f"Val Loss: {loss.item():.4f}")

        avg_val_loss = np.mean(valid_loss)
        print(f"Validation Average Loss: {avg_val_loss:.4f}")

        if self.min_valid_loss is None:
            self.min_valid_loss = avg_val_loss
            self.best_epoch = self.current_epoch if hasattr(self, 'current_epoch') else 0
            print("Update best model! Best epoch: {}".format(self.best_epoch))
        elif avg_val_loss < self.min_valid_loss:
            self.min_valid_loss = avg_val_loss
            self.best_epoch = self.current_epoch if hasattr(self, 'current_epoch') else 0
            print("Update best model! Best epoch: {}".format(self.best_epoch))

        return avg_val_loss

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(self.model_dir, f"{self.model_file_name}_Epoch{index}.pth")
        torch.save(self.model.state_dict(), model_path)
        print(f"Saved Model: {model_path}")

    def test(self, data_loader):
        """Tests the model."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print('')
        print("=== QAFC-PhysFormer Testing ===")

        predictions = dict()
        labels = dict()

        # Load checkpoint for testing
        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
            print(self.config.INFERENCE.MODEL_PATH)
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()

        print("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                video = batch[0].float().to(self.device)
                # Handle different batch formats
                if len(batch) > 2:
                    bvp_label = batch[2].float().to(self.device)
                    subj_index = batch[1][0] if len(batch) > 1 else idx
                    sort_index = int(batch[3][0]) if len(batch) > 3 else 0
                else:
                    bvp_label = batch[1].float().to(self.device)
                    subj_index = idx
                    sort_index = 0

                # Inference
                pred_rppg, _, _, _, _ = self.model(video, gra_sharp=2.0)

                # Normalize for metric calculation
                pred_rppg_norm = (pred_rppg - pred_rppg.mean(dim=-1, keepdim=True)) / \
                                pred_rppg.std(dim=-1, keepdim=True)

                batch_size = video.shape[0]
                for i in range(batch_size):
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()

                    predictions[subj_index][sort_index] = pred_rppg[i]
                    labels[subj_index][sort_index] = bvp_label[i]

        print('')
        calculate_metrics(predictions, labels, self.config)
        if self.config.TEST.OUTPUT_SAVE_DIR:
            self.save_test_outputs(predictions, labels, self.config)
