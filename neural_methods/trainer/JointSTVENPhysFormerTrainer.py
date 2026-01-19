
import os
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from neural_methods.trainer.BaseTrainer import BaseTrainer
from neural_methods.model.STVEN import PhysFormerWithSTVEN
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.loss.PhysFormerLossComputer import TorchLossComputer
import math

class JointSTVENPhysFormerTrainer(BaseTrainer):
    """
    Trainer for Joint Training of STVEN (Frontend) and PhysFormer (Backend).
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
        
        # Initialize PhysFormerWithSTVEN
        self.model = PhysFormerWithSTVEN(
            stven_config=config.MODEL.STVEN,
            physformer_config=config.MODEL.PHYSFORMER
        ).to(self.device)

        # Load Pretrained Weights
        self._load_pretrained_weights(config)

        # Freeze PhysFormer
        for param in self.model.physformer.parameters():
            param.requires_grad = False
        
        # Verify freezing
        for name, param in self.model.physformer.named_parameters():
             if param.requires_grad:
                 print(f"Warning: PhysFormer parameter {name} is not frozen!")

        # Optimizer (Only for STVEN)
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=config.TRAIN.LR
        )
        
        # Loss Functions (for rPPG)
        self.criterion_Pearson = Neg_Pearson()
        
        # Logging
        self.min_valid_loss = None
        self.best_epoch = 0

    def _load_pretrained_weights(self, config):
        """Loads pretrained weights for STVEN and PhysFormer."""
        # Load STVEN
        if config.MODEL.STVEN.PRETRAINED_PATH:
            print(f"Loading STVEN weights from {config.MODEL.STVEN.PRETRAINED_PATH}")
            stven_state = torch.load(config.MODEL.STVEN.PRETRAINED_PATH, map_location=self.device)
            # Handle potential DataParallel wrapping or different key names
            if 'state_dict' in stven_state:
                stven_state = stven_state['state_dict']
            self.model.stven.load_state_dict(stven_state, strict=False)
        else:
            print("Warning: No pretrained path provided for STVEN. Using random init.")

        # Load PhysFormer
        if config.MODEL.PHYSFORMER.PRETRAINED_PATH:
            print(f"Loading PhysFormer weights from {config.MODEL.PHYSFORMER.PRETRAINED_PATH}")
            phys_state = torch.load(config.MODEL.PHYSFORMER.PRETRAINED_PATH, map_location=self.device)
             # Handle potential DataParallel wrapping or different key names
            if 'state_dict' in phys_state:
                phys_state = phys_state['state_dict']
            # Remove module. prefix if present (from DataParallel)
            new_phys_state = {}
            for k, v in phys_state.items():
                if k.startswith('module.'):
                    new_phys_state[k[7:]] = v
                else:
                    new_phys_state[k] = v
            self.model.physformer.load_state_dict(new_phys_state, strict=False)
        else:
            print("Warning: No pretrained path provided for PhysFormer. This is critical for backend!")

    def train(self, data_loader):
        """Training Loop"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        for epoch in range(self.max_epoch_num):
            print(f"==== Joint Training Epoch: {epoch} ====")
            self.model.train() # STVEN in train mode
            self.model.physformer.eval() # PhysFormer always frozen/eval
            
            tbar = tqdm(data_loader["train"], ncols=80)
            loss_rPPG_avg = []
            
            for idx, batch in enumerate(tbar):
                # STVENLoader returns: compressed_data, uncompressed_data, bitrate_label, bvp_label
                compressed_vid = batch[0].float().to(self.device)
                uncompressed_vid = batch[1].float().to(self.device)
                # bitrate_label = batch[2] # Ignored
                # bvp_label = batch[3] # Ignored for training (using Teacher)
                
                # 1. Teacher Forward (Uncompressed -> PhysFormer)
                # PhysFormer expects (video, gra_sharp). returns (rPPG, attention_scores...)
                # We need to access physformer directly or via wrapper? 
                # Wrapper forward() goes stven->physformer.
                # We need just physformer.
                with torch.no_grad():
                     teacher_rPPG, _, _, _ = self.model.physformer(uncompressed_vid, 2.0)
                     # Normalize Teacher rPPG
                     teacher_rPPG = (teacher_rPPG - torch.mean(teacher_rPPG, axis=-1, keepdim=True)) / torch.std(teacher_rPPG, axis=-1, keepdim=True)

                # 2. Student Forward (Compressed -> STVEN -> PhysFormer)
                # Hardcode label 0 (High Quality) for global blind enhancement
                num_classes = self.model.stven.num_bitrate_levels
                student_label = torch.zeros(compressed_vid.shape[0], num_classes).to(self.device)
                student_label[:, 0] = 1.0 
                
                # PhysFormerWithSTVEN forward does: stven(x, label) -> physformer(x)
                student_rPPG, _, _, _ = self.model(compressed_vid, student_label, 2.0)
                
                # Normalize Student rPPG
                student_rPPG = (student_rPPG - torch.mean(student_rPPG, axis=-1, keepdim=True)) / torch.std(student_rPPG, axis=-1, keepdim=True)
                
                # 3. Loss (Pearson between Student and Teacher)
                loss_pearson = self.criterion_Pearson(student_rPPG, teacher_rPPG)
                
                total_loss = loss_pearson
                
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                loss_rPPG_avg.append(total_loss.item())
                tbar.set_description(f"Loss: {total_loss.item():.4f}")

            print(f"Epoch {epoch} Avg Loss: {np.mean(loss_rPPG_avg):.4f}")
            self.save_model(epoch)
            self.valid(data_loader)

    def valid(self, data_loader):
        """Validation Loop"""
        if data_loader["valid"] is None:
            print("No data for valid")
            return

        print("==== Joint Validation ====")
        self.model.eval()
        valid_loss = []

        with torch.no_grad():
            tbar = tqdm(data_loader["valid"], ncols=80)
            for idx, batch in enumerate(tbar):
                compressed_vid = batch[0].float().to(self.device)
                uncompressed_vid = batch[1].float().to(self.device)
                # bvp_label = batch[3] # Ignored for validation (using Teacher)
                
                # Teacher
                teacher_rPPG, _, _, _ = self.model.physformer(uncompressed_vid, 2.0)
                teacher_rPPG = (teacher_rPPG - torch.mean(teacher_rPPG, axis=-1, keepdim=True)) / torch.std(teacher_rPPG, axis=-1, keepdim=True)

                # Student
                num_classes = self.model.stven.num_bitrate_levels
                student_label = torch.zeros(compressed_vid.shape[0], num_classes).to(self.device)
                student_label[:, 0] = 1.0

                student_rPPG, _, _, _ = self.model(compressed_vid, student_label, 2.0)
                student_rPPG = (student_rPPG - torch.mean(student_rPPG, axis=-1, keepdim=True)) / torch.std(student_rPPG, axis=-1, keepdim=True)

                loss = self.criterion_Pearson(student_rPPG, teacher_rPPG)
                valid_loss.append(loss.item())
                
                tbar.set_description(f"Val Loss: {loss.item():.4f}")

        avg_val_loss = np.mean(valid_loss)
        print(f"Validation Average Loss: {avg_val_loss:.4f}")
        
        if self.min_valid_loss is None or avg_val_loss < self.min_valid_loss:
            self.min_valid_loss = avg_val_loss
            self.best_epoch = -1 
            print("New best validation loss!")
            
    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        # We only want to save STVEN weights usually, or the whole thing?
        # User said "tune STVEN", so likely wants the tuned STVEN.
        # But saving the whole wrapper is safer for resuming.
        model_path = os.path.join(self.model_dir, f"{self.model_file_name}_Joint_Epoch{index}.pth")
        torch.save(self.model.state_dict(), model_path)
        print(f"Saved Joint Model: {model_path}")

    def test(self, data_loader):
        """test"""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        
        print("==== Joint Testing ====")
        self.model.eval()
        predictions = dict()
        labels = dict()
        
        # Metrics
        loss_mean = 0.0
        
        # We use standard metrics from evaluation package or implement here?
        # Main.py might handle it if we return results? No, main calls trainer.test()
        # Usually BaseTrainer.test saves results.
        
        with torch.no_grad():
             tbar = tqdm(data_loader["test"], ncols=80)
             for idx, batch in enumerate(tbar):
                 compressed_vid = batch[0].float().to(self.device)
                 # uncompressed_vid = batch[1] # Not needed for test unless reference
                 # bitrate_label = batch[2] 
                 bvp_label = batch[3].float().to(self.device)
                 
                 # Prepare student label (High Quality)
                 num_classes = self.model.stven.num_bitrate_levels
                 student_label = torch.zeros(compressed_vid.shape[0], num_classes).to(self.device)
                 student_label[:, 0] = 1.0
                 
                 # Inference
                 # student_rPPG, score1, score2, score3 = self.model(compressed_vid, student_label, 2.0)
                 # Wait, forward returns (rPPG, ...)
                 student_rPPG, _, _, _ = self.model(compressed_vid, student_label, 2.0)
                 
                 # Normalize for metric calculation? 
                 # Usually rPPG evaluation does bandpass filtering and calculating HR, 
                 # or if calculating correlation on signal, we normalize.
                 # Let's normalize both to 0 mean 1 std.
                 
                 for i in range(len(bvp_label)):
                     pred = (student_rPPG[i] - torch.mean(student_rPPG[i])) / torch.std(student_rPPG[i])
                     gt = (bvp_label[i] - torch.mean(bvp_label[i])) / torch.std(bvp_label[i])
                     
                     if idx not in predictions:
                         predictions[idx] = []
                         labels[idx] = []
                     predictions[idx].append(pred.cpu().numpy())
                     labels[idx].append(gt.cpu().numpy())
                     
        # Calculate Metrics (Pearson, MAE, RMSE) - usually done post-processing or here?
        # For simplicity, let's print overall Pearson
        # This part depends on how strict the user wants the "Test" output to be. 
        # For checks, I'll save the outputs to a .npz or something for the user to analyze, 
        # or implement basic HR estimation? 
        # Given "Refining Joint Training", I will stick to saving predictions and printing Loss.
        # But wait, the user wants to see "metrics". 
        
        # I will implement basic Pearson calculation on the fly
        all_preds = []
        all_labels = []
        for i in predictions:
            all_preds.extend(predictions[i])
            all_labels.extend(labels[i])
            
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Pearson
        corrs = []
        for p, l in zip(all_preds, all_labels):
            corr = np.corrcoef(p, l)[0, 1]
            corrs.append(corr)
        
        print(f"Test Pearson Correlation: {np.mean(corrs):.4f}")
        
        # Save results
        save_dir = self.config.TEST.OUTPUT_SAVE_DIR
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(os.path.join(save_dir, "test_predictions.npy"), all_preds)
        np.save(os.path.join(save_dir, "test_labels.npy"), all_labels)
        print(f"Saved test results to {save_dir}")

