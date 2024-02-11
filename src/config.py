import os

class Config():
    project_name = 'hms'
    seed = 42

    # Paths
    root = f'/media/latlab/MR/projects/kaggle-{project_name}'
    data_dir = os.path.join(root, 'data')
    results_dir = os.path.join(root, 'results')
    train_eeg_dir = os.path.join(data_dir, 'train_eegs')
    train_spectrogram_dir = os.path.join(data_dir, 'train_spectrograms')

    # Table vars
    stratif_vars = ['expert_consensus']
    grouping_vars = ['patient_id']

    # Wandb
    use_wandb = True
    wandb_key = '1b0401db7513303bdea77fb070097f9d2850cf3b'
    tags = ['torch', 'cv', 'best_epoch']
    notes = ''

    # Training vars
    cv_fold = 5
    train_full_model = False
    one_fold = False    # Train for only one fold

    # Data specific
    TARGETS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']