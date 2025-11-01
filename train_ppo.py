"""
PPO Training Script for Hexapod Balance Control
Trains a PPO agent using Stable-Baselines3 with curriculum learning.
"""

import os
import argparse
from datetime import datetime
import numpy as np

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn

from hexapod_env import HexapodBalanceEnv, make_hexapod_env


class CurriculumCallback(EvalCallback):
    """
    Callback to switch from balance-only to stepping phase based on performance.
    """
    
    def __init__(
        self,
        eval_env,
        stepping_threshold: float = 5.0,
        stepping_episodes: int = 20,
        **kwargs
    ):
        super().__init__(eval_env, **kwargs)
        self.stepping_threshold = stepping_threshold
        self.stepping_episodes = stepping_episodes
        self.stepping_enabled = False
        self.reward_history = []
    
    def _on_step(self) -> bool:
        # Call parent evaluation
        continue_training = super()._on_step()
        
        # Check if we should enable stepping
        if not self.stepping_enabled and len(self.evaluations_results) > 0:
            recent_rewards = self.evaluations_results[-self.stepping_episodes:]
            if len(recent_rewards) >= self.stepping_episodes:
                avg_reward = np.mean(recent_rewards)
                
                if avg_reward >= self.stepping_threshold:
                    print(f"\n{'='*60}")
                    print(f"CURRICULUM SWITCH: Enabling stepping phase!")
                    print(f"Average reward: {avg_reward:.2f} >= {self.stepping_threshold:.2f}")
                    print(f"{'='*60}\n")
                    
                    # Enable stepping in all environments
                    for env_idx in range(self.training_env.num_envs):
                        self.training_env.env_method("set_enable_stepping", True, indices=env_idx)
                    
                    self.stepping_enabled = True
        
        return continue_training


def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.
    
    Args:
        initial_value: Initial learning rate
    
    Returns:
        Schedule function
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def create_training_env(n_envs: int = 8, **env_kwargs):
    """
    Create vectorized training environments.
    
    Args:
        n_envs: Number of parallel environments
        **env_kwargs: Environment configuration
    
    Returns:
        Vectorized environment
    """
    env_fns = [make_hexapod_env(i, seed=42, **env_kwargs) for i in range(n_envs)]
    env = SubprocVecEnv(env_fns)
    env = VecMonitor(env)
    return env


def train_ppo(
    output_dir: str = "./runs",
    total_timesteps: int = 2_000_000,
    n_envs: int = 8,
    n_steps: int = 1024,
    batch_size: int = 1024,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.005,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    policy_kwargs: dict = None,
    curriculum: bool = True,
    curriculum_threshold: float = 5.0,
    eval_freq: int = 10_000,
    checkpoint_freq: int = 50_000,
    seed: int = 42,
):
    """
    Train PPO agent on hexapod balance task.
    
    Args:
        output_dir: Directory for logs and models
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
        n_steps: Steps per environment per update
        batch_size: Minibatch size
        learning_rate: Learning rate (can use schedule)
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_range: PPO clip range
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Max gradient norm for clipping
        policy_kwargs: Policy network configuration
        curriculum: Enable curriculum learning
        curriculum_threshold: Reward threshold to enable stepping
        eval_freq: Evaluation frequency (steps)
        checkpoint_freq: Checkpoint save frequency (steps)
        seed: Random seed
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"ppo_hexapod_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"{'='*60}")
    print(f"Training PPO for Hexapod Balance Control")
    print(f"{'='*60}")
    print(f"Output directory: {run_dir}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    print(f"Curriculum learning: {curriculum}")
    print(f"{'='*60}\n")
    
    # Environment configuration
    env_kwargs = {
        "max_episode_steps": 500,  # 10s at 50Hz
        "fall_threshold_deg": 35.0,
        "action_delta_clamp": 0.05,
        "enable_stepping": False,  # Start with balance only
        "domain_randomization": True,
        "action_smoothing_alpha": 0.3,
    }
    
    # Create training and evaluation environments
    train_env = create_training_env(n_envs=n_envs, **env_kwargs)
    eval_env = create_training_env(n_envs=4, **env_kwargs)
    
    # Policy network configuration
    if policy_kwargs is None:
        policy_kwargs = {
            "net_arch": [128, 128],  # Two hidden layers
            "activation_fn": nn.Tanh,
        }
    
    # Create PPO model
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=linear_schedule(learning_rate),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log=os.path.join(run_dir, "tensorboard"),
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        device="auto",
    )
    
    # Configure logger
    logger = configure(run_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)
    
    # Callbacks
    callbacks = []
    
    # Evaluation callback
    if curriculum:
        eval_callback = CurriculumCallback(
            eval_env=eval_env,
            best_model_save_path=os.path.join(run_dir, "best_model"),
            log_path=os.path.join(run_dir, "eval"),
            eval_freq=eval_freq // n_envs,  # Adjust for parallel envs
            n_eval_episodes=10,
            deterministic=True,
            render=False,
            stepping_threshold=curriculum_threshold,
            stepping_episodes=20,
        )
    else:
        eval_callback = EvalCallback(
            eval_env=eval_env,
            best_model_save_path=os.path.join(run_dir, "best_model"),
            log_path=os.path.join(run_dir, "eval"),
            eval_freq=eval_freq // n_envs,
            n_eval_episodes=10,
            deterministic=True,
            render=False,
        )
    callbacks.append(eval_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq // n_envs,
        save_path=os.path.join(run_dir, "checkpoints"),
        name_prefix="ppo_hexapod",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    
    callback_list = CallbackList(callbacks)
    
    # Train the model
    print("Starting training...\n")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        log_interval=10,
        progress_bar=True,
    )
    
    # Save final model
    final_model_path = os.path.join(run_dir, "final_model")
    model.save(final_model_path)
    print(f"\nTraining complete! Final model saved to: {final_model_path}.zip")
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    return model, run_dir


def evaluate_model(model_path: str, n_episodes: int = 10, render: bool = True):
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to saved model (.zip file)
        n_episodes: Number of episodes to evaluate
        render: Whether to render during evaluation
    """
    print(f"Evaluating model: {model_path}")
    
    # Load model
    model = PPO.load(model_path)
    
    # Create environment
    env = HexapodBalanceEnv(
        max_episode_steps=500,
        enable_stepping=True,  # Enable stepping for evaluation
    )
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            done = terminated or truncated
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step)
        
        print(f"Episode {episode + 1}/{n_episodes}: "
              f"Reward = {episode_reward:.2f}, "
              f"Length = {step}, "
              f"Final Roll = {info['roll_deg']:.1f}°, "
              f"Final Pitch = {info['pitch_deg']:.1f}°")
    
    print(f"\nEvaluation Results:")
    print(f"  Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Mean length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train PPO for hexapod balance control")
    parser.add_argument("--output-dir", type=str, default="./runs", help="Output directory")
    parser.add_argument("--timesteps", type=int, default=2_000_000, help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--no-curriculum", action="store_true", help="Disable curriculum learning")
    parser.add_argument("--curriculum-threshold", type=float, default=5.0, help="Reward threshold for stepping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval", type=str, default=None, help="Evaluate a trained model")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of evaluation episodes")
    
    args = parser.parse_args()
    
    if args.eval:
        # Evaluation mode
        evaluate_model(args.eval, n_episodes=args.eval_episodes, render=True)
    else:
        # Training mode
        model, run_dir = train_ppo(
            output_dir=args.output_dir,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            learning_rate=args.learning_rate,
            curriculum=not args.no_curriculum,
            curriculum_threshold=args.curriculum_threshold,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
