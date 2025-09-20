from dataclasses import dataclass, field


@dataclass
class EnvironmentConfig:
    env_name: str = "HumanoidStandup-v5"
    max_episodes: int = 1000
    max_steps_per_episode: int = 1000


@dataclass
class HyperparametersConfig:
    lr_actor: float = 1e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    tau: float = 5e-3
    batch_size: int = 64
    buffer_size: int = 100_000
    hidden_size: int = 256


@dataclass
class TrainingConfig:
    update_every: int = 4
    noise_scale: float = 0.1
    noise_decay: float = 0.995
    min_noise: float = 0.01


@dataclass
class LoggingConfig:
    log_interval: int = 10
    save_interval: int = 100
    model_save_path: str = "models/"


@dataclass
class Config:
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    hyperparameter: HyperparametersConfig = field(default_factory=HyperparametersConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


config = Config()
