import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from tqdm import tqdm
from neural_methods.trainer.BaseTrainer import BaseTrainer

class STVENTrainer(BaseTrainer):
    def __init__(self, config, data_loader):
        """
        STVEN Trainer for video enhancement pre-training.
        
        Args:
            config: Configuration object.
            data_loader: Generic data loader dictionary {"train": ..., "valid": ..., "test": ...}.
        """
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.config = config

        # Initialize STVEN model
        from neural_methods.model.STVEN import STVEN
        stven_config = config.MODEL.STVEN
        self.model = STVEN(
            in_channels=stven_config.get('in_channels', 3),
            out_channels=stven_config.get('out_channels', 3),
            base_channels=stven_config.get('base_channels', 64),
            num_st_blocks=stven_config.get('num_st_blocks', 6),
            frame_length=stven_config.get('frame_length', 160),
            use_bitrate_labels=stven_config.get('use_bitrate_labels', False),
            num_bitrate_levels=stven_config.get('num_bitrate_levels', 1)
        ).to(self.device)

        # Optimization & Loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.TRAIN.LR if hasattr(config.TRAIN, 'LR') else 1e-4)
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def train_step(self, compressed_vid, uncompressed_vid, bitrate_label):
        """
        Performs a single training step with reconstruction and cycle consistency losses.
        
        Args:
            compressed_vid: Video with compression [B, C, T, H, W]
            uncompressed_vid: Original video [B, C, T, H, W]
            bitrate_label: One-hot encoded bitrate label [B, num_classes]
            
        Returns:
            total_loss, rec_loss, cyc_loss
        """
        compressed_vid = compressed_vid.to(self.device).float()
        uncompressed_vid = uncompressed_vid.to(self.device).float()
        bitrate_label = bitrate_label.to(self.device).float()
        
        B = compressed_vid.shape[0]
        
        # 1. Create "Target 0" label (Uncompressed) for enhancement
        # Following pseudo: Assuming label 0 is [1, 0, 0, ...]
        target_uncompressed_label = torch.zeros_like(bitrate_label)
        target_uncompressed_label[:, 0] = 1.0

        # --- PHASE 1: Enhancement (Forward Pass) ---
        # Map: Compressed (k) -> Uncompressed (0)
        enhanced_vid = self.model(compressed_vid, target_uncompressed_label)

        # --- PHASE 2: Cycle Consistency (Backward Pass) ---
        # Map: Enhanced (0) -> Re-compressed (k)
        reconstructed_compressed_vid = self.model(enhanced_vid, bitrate_label)

        # --- PHASE 3: Loss Calculation ---
        
        # A. Reconstruction Loss
        loss_rec_l1 = self.l1_loss(enhanced_vid, uncompressed_vid)
        loss_rec_mse = self.mse_loss(enhanced_vid, uncompressed_vid)
        loss_rec = loss_rec_mse + loss_rec_l1

        # B. Cycle Loss
        loss_cyc = self.l1_loss(reconstructed_compressed_vid, compressed_vid)

        # Total Loss
        total_loss = loss_rec + loss_cyc

        # --- PHASE 4: Optimization ---
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), loss_rec.item(), loss_cyc.item()

    def train(self, data_loader):
        """
        Main training loop for STVEN.
        """
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        for epoch in range(self.max_epoch_num):
            print(f"==== STVEN Training Epoch: {epoch} ====")
            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80)
            
            epoch_total_loss = []
            epoch_rec_loss = []
            epoch_cyc_loss = []

            for idx, batch in enumerate(tbar):
                # Expecting batch format: [compressed_vid, uncompressed_vid, bitrate_label]
                # Note: Adjust following data loader implementation
                compressed_vid, uncompressed_vid, bitrate_label = batch[0], batch[1], batch[2]
                
                loss_total, loss_rec, loss_cyc = self.train_step(compressed_vid, uncompressed_vid, bitrate_label)
                
                epoch_total_loss.append(loss_total)
                epoch_rec_loss.append(loss_rec)
                epoch_cyc_loss.append(loss_cyc)
                
                tbar.set_description(f"Loss: {loss_total:.4f} (Rec: {loss_rec:.4f}, Cyc: {loss_cyc:.4f})")

            print(f"Epoch {epoch} Average Loss: {np.mean(epoch_total_loss):.4f}")
            self.valid(data_loader)
            self.save_model(epoch)

    def valid(self, data_loader):
        """
        Validation loop for STVEN.
        """
        if data_loader["valid"] is None:
            print("No data for valid, skipping...")
            return None

        print("==== STVEN Validation ====")
        self.model.eval()
        valid_loss = []
        
        with torch.no_grad():
            tbar = tqdm(data_loader["valid"], ncols=80)
            for idx, batch in enumerate(tbar):
                compressed_vid, uncompressed_vid, bitrate_label = batch[0], batch[1], batch[2]
                compressed_vid = compressed_vid.to(self.device).float()
                uncompressed_vid = uncompressed_vid.to(self.device).float()
                bitrate_label = bitrate_label.to(self.device).float()
                
                # 1. Create "Target 0" label (Uncompressed)
                target_uncompressed_label = torch.zeros_like(bitrate_label)
                target_uncompressed_label[:, 0] = 1.0

                # --- PHASE 1: Enhancement ---
                enhanced_vid = self.model(compressed_vid, target_uncompressed_label)

                # --- PHASE 2: Cycle Consistency ---
                reconstructed_compressed_vid = self.model(enhanced_vid, bitrate_label)

                # --- PHASE 3: Loss Calculation ---
                loss_rec_l1 = self.l1_loss(enhanced_vid, uncompressed_vid)
                loss_rec_mse = self.mse_loss(enhanced_vid, uncompressed_vid)
                loss_rec = loss_rec_mse + loss_rec_l1
                
                loss_cyc = self.l1_loss(reconstructed_compressed_vid, compressed_vid)
                
                total_loss = loss_rec + loss_cyc
                valid_loss.append(total_loss.item())
                
                tbar.set_description(f"Val Loss: {total_loss.item():.4f}")

        avg_val_loss = np.mean(valid_loss)
        print(f"Validation Average Loss: {avg_val_loss:.4f}")
        return avg_val_loss

    def test(self, data_loader):
        """
        Test loop for STVEN.
        """
        if data_loader["test"] is None:
            print("No data for test, skipping...")
            return None

        print("==== STVEN Testing ====")
        self.model.eval()
        test_loss = []
        
        with torch.no_grad():
            tbar = tqdm(data_loader["test"], ncols=80)
            for idx, batch in enumerate(tbar):
                compressed_vid, uncompressed_vid, bitrate_label = batch[0], batch[1], batch[2]
                compressed_vid = compressed_vid.to(self.device).float()
                uncompressed_vid = uncompressed_vid.to(self.device).float()
                bitrate_label = bitrate_label.to(self.device).float()
                
                target_uncompressed_label = torch.zeros_like(bitrate_label)
                target_uncompressed_label[:, 0] = 1.0

                enhanced_vid = self.model(compressed_vid, target_uncompressed_label)
                reconstructed_compressed_vid = self.model(enhanced_vid, bitrate_label)

                loss_rec_l1 = self.l1_loss(enhanced_vid, uncompressed_vid)
                loss_rec_mse = self.mse_loss(enhanced_vid, uncompressed_vid)
                loss_rec = loss_rec_mse + loss_rec_l1
                
                loss_cyc = self.l1_loss(reconstructed_compressed_vid, compressed_vid)
                
                total_loss = loss_rec + loss_cyc
                test_loss.append(total_loss.item())
                tbar.set_description(f"Test Loss: {total_loss.item():.4f}")

        print(f"Test Average Loss: {np.mean(test_loss):.4f}")

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_STVEN_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved STVEN Model Path: ', model_path)
