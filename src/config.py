import os
import json

class Config():
    project_name = 'hms'
    seed = 42
    debug = False

    # Paths
    root = f'/media/latlab/MR/projects/kaggle-{project_name}'
    data_dir = os.path.join(root, 'data')
    results_dir = os.path.join(root, 'results')
    train_eeg_dir = os.path.join(data_dir, 'train_eegs')
    train_spectrogram_dir = os.path.join(data_dir, 'train_spectrograms')
    models_dir = os.path.join(results_dir, 'models')

    # Table vars
    stratif_vars = ['expert_consensus', 'rater_group']
    grouping_vars = ['patient_id']

    # Wandb
    use_wandb = True
    wandb_key = '1b0401db7513303bdea77fb070097f9d2850cf3b'
    tags = ['torch', 'cv', 'v2']
    notes = ''

    # Training vars
    cv_fold = 5
    train_full_model = False
    one_fold = False    # Train for only one fold
    dataloader_num_workers = 8
    pretrained = True
    train_type = 'normal'
    deterministic = False # torch.backends.cudnn.deterministic
    benchmark = False     # torch.backends.cudnn.benchmark
    pin_memory = True

    # Data specific
    TARGETS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
    num_classes = len(TARGETS)
    ch_list = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2', 'EKG']
    ch_pairs = [('Fp1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'), ('Fp2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'), ('Fp1', 'F3'), 
                ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'), ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'), ('Fz', 'Cz'), ('Cz', 'Pz')]
    
        