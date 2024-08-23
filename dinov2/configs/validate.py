from omegaconf import DictConfig

def dataset_validation(config: DictConfig) -> None:
    """Validate the dataset configuration.
    Check that datasets can be found and are formatted correctly.
    Check for no issues with num_slices.
    """
    raise NotImplementedError("Dataset validation is not implemented yet.")

def validate_config(config: DictConfig) -> None:
    errors = []
    
    optim_config = config.get("optim", {})
    
    if optim_config.get("warmup_epochs", 0) > optim_config.get("epochs", 0):
        errors.append(f"Validation error: warmup_epochs ({optim_config['warmup_epochs']}) cannot be greater than epochs ({optim_config['epochs']}).")

    teacher_config = config.get("teacher", {})
    if teacher_config.get("warmup_teacher_temp_epochs", 0) > optim_config.get("epochs", 0):
        errors.append(f"Validation error: warmup_teacher_temp_epochs ({teacher_config['warmup_teacher_temp_epochs']}) cannot be greater than epochs ({optim_config['epochs']}).")
    

    dino_config = config.get("dino", {})
    if dino_config.get("head_n_prototypes", 1) < 1:
        errors.append(f"Validation error: dino.head_n_prototypes ({dino_config['head_n_prototypes']}) must be at least 1.")
    
    ibot_config = config.get("ibot", {})
    if ibot_config.get("head_n_prototypes", 1) < 1:
        errors.append(f"Validation error: ibot.head_n_prototypes ({ibot_config['head_n_prototypes']}) must be at least 1.")

    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
    else:
        print("Configuration is valid.")
