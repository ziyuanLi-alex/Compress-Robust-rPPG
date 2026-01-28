import os
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from neural_methods.trainer.BaseTrainer import BaseTrainer
from neural_methods.model.STVEN import PhysFormerWithSTVEN
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.loss.PhysFormerLossComputer import TorchLossComputer
from evaluation.metrics import calculate_metrics
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
        
        # Logging (initialize before _load_pretrained_weights)
        self.min_valid_loss = None
        self.best_epoch = 0
        
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

    def _load_pretrained_weights(self, config):
        """Loads pretrained weights for STVEN and PhysFormer (for training initialization)."""
        # This is called during __init__ for training mode
        # For testing, the joint checkpoint is loaded in test() method
        
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

        mean_training_losses = []
        mean_valid_losses = []
        
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
                bvp_label = batch[3].float().to(self.device) # Ground Truth PPG

                # 1. Prepare Target (Ground Truth PPG)
                # Standardize Label: (L - mean) / std
                teacher_rPPG = (bvp_label - torch.mean(bvp_label, axis=-1, keepdim=True)) / torch.std(bvp_label, axis=-1, keepdim=True)
                
                # 2. Student Forward (Compressed -> STVEN -> PhysFormer)
                # Hardcode label 0 (High Quality) for global blind enhancement
                num_classes = self.model.stven.num_bitrate_levels
                student_label = torch.zeros(compressed_vid.shape[0], num_classes).to(self.device)
                student_label[:, 0] = 1.0 
                
                # PhysFormerWithSTVEN forward does: stven(x, label) -> physformer(x)
                student_rPPG, _, _, _ = self.model(compressed_vid, student_label, 2.0)
                
                # Normalize Student rPPG
                student_rPPG = (student_rPPG - torch.mean(student_rPPG, axis=-1, keepdim=True)) / torch.std(student_rPPG, axis=-1, keepdim=True)
                
                # 3. Loss (Negative Pearson between Student and Ground Truth)
                loss_pearson = self.criterion_Pearson(student_rPPG, teacher_rPPG)
                
                total_loss = loss_pearson
                
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                loss_rPPG_avg.append(total_loss.item())
                tbar.set_description(f"Loss: {total_loss.item():.4f}")

            epoch_avg_loss = np.mean(loss_rPPG_avg)
            mean_training_losses.append(epoch_avg_loss)
            print(f"Epoch {epoch} Avg Loss: {epoch_avg_loss:.4f}")
            
            self.save_model(epoch)
            
            # Validation
            self.current_epoch = epoch  # For best_epoch tracking in valid()
            valid_loss = self.valid(data_loader)
            if valid_loss is not None:
                mean_valid_losses.append(valid_loss)
        
        # Print best epoch summary
        if not self.config.TEST.USE_LAST_EPOCH:
            print("best trained epoch: {}, min_val_loss: {}".format(
                self.best_epoch, self.min_valid_loss))
        
        # Plot and save loss history
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, [], self.config)

    def valid(self, data_loader):
        """Validation Loop"""
        if data_loader["valid"] is None:
            print("No data for valid")
            return None

        print("==== Joint Validation ====")
        self.model.eval()
        valid_loss = []

        with torch.no_grad():
            tbar = tqdm(data_loader["valid"], ncols=80)
            for idx, batch in enumerate(tbar):
                compressed_vid = batch[0].float().to(self.device)
                uncompressed_vid = batch[1].float().to(self.device)
                bvp_label = batch[3].float().to(self.device) # Ground Truth PPG
                
                # Teacher (Ground Truth)
                teacher_rPPG = (bvp_label - torch.mean(bvp_label, axis=-1, keepdim=True)) / torch.std(bvp_label, axis=-1, keepdim=True)

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
        # We only want to save STVEN weights usually, or the whole thing?
        # User said "tune STVEN", so likely wants the tuned STVEN.
        # But saving the whole wrapper is safer for resuming.
        model_path = os.path.join(self.model_dir, f"{self.model_file_name}_Joint_Epoch{index}.pth")
        torch.save(self.model.state_dict(), model_path)
        print(f"Saved Joint Model: {model_path}")

    def test(self, data_loader):
        """ Runs the model on test sets."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        
        print('')
        print("===Testing===")

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
                compressed_vid = batch[0].float().to(self.device)
                if self.config.TEST.DATA.DATASET == 'UBFC-rPPG-h264':
                    bvp_label = batch[1].float().to(self.device)
                    subj_index = batch[2][0]
                    sort_index = int(batch[3][0])
                else:
                    bvp_label = batch[3].float().to(self.device)
                    # Handle indices for STVENLoader if attempting to run calculate_metrics
                    # STVENLoader doesn't return metadata needed for detailed metrics dict keys
                    # So we might default to simple indexing if not h264
                    subj_index = idx 
                    sort_index = 0

                # Prepare student label (High Quality)
                num_classes = self.model.stven.num_bitrate_levels
                student_label = torch.zeros(compressed_vid.shape[0], num_classes).to(self.device)
                student_label[:, 0] = 1.0
                
                # Inference
                student_rPPG, _, _, _ = self.model(compressed_vid, student_label, 2.0)
                
                # Normalize for metric calculation
                student_rPPG = (student_rPPG - torch.mean(student_rPPG, axis=-1, keepdim=True)) / torch.std(student_rPPG, axis=-1, keepdim=True)
                # Note: bvp_label used in calculate_metrics assumes it's just raw values to be passed

                batch_size = compressed_vid.shape[0]
                for i in range(batch_size):
                    # For UBFC-rPPG-h264 with batch size 4 (default), metadata is usually tuple/list
                    if self.config.TEST.DATA.DATASET == 'UBFC-rPPG-h264':
                         subj_index = batch[2][i]
                         sort_index = int(batch[3][i])

                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    
                    predictions[subj_index][sort_index] = student_rPPG[i]
                    labels[subj_index][sort_index] = bvp_label[i]

        print('')
        calculate_metrics(predictions, labels, self.config)
        if self.config.TEST.OUTPUT_SAVE_DIR: # saving test outputs
            self.save_test_outputs(predictions, labels, self.config)
