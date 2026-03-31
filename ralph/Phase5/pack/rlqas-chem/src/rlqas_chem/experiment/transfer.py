"""Cross-geometry policy transfer for quantum circuit search."""
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict, Any, Optional


def train_multi_geometry(
    molecule: str,
    bond_lengths_train: List[float],
    agent_type: str = 'ppo',
    operator_pool: str = 'fop',
    n_episodes_per_geometry: int = 100,
) -> Any:
    """Train an RL agent across multiple bond length geometries.

    The SAME agent is trained across all geometries round-robin,
    alternating geometries with n_episodes_per_geometry each.

    Args:
        molecule: Molecular formula (e.g. "H2")
        bond_lengths_train: List of bond lengths to train on
        agent_type: RL agent type ("ppo")
        operator_pool: Operator pool type ("fop", "qop")
        n_episodes_per_geometry: Episodes per geometry per round

    Returns:
        Trained agent object
    """
    from rlqas_chem.molecule import process_molecule
    from rlqas_chem.search.ucc.environment import UCCSearchEnv
    from rlqas_chem.search.ucc.controller import UCCPPOAgent

    # Create envs for each geometry
    envs = []
    for bl in bond_lengths_train:
        mol = process_molecule(molecule, bl, 'UCC')
        env = UCCSearchEnv(mol, {
            "run_classical_opt": True,
            "complexity_penalty": 0.0,
        })
        envs.append(env)

    if not envs:
        raise ValueError("No training geometries provided")

    # Initialize agent from first env (sets obs/action dimensions)
    agent_config = {
        "use_gpu": False,
        "seed": 42,
        "verbose": 0,
        "n_steps": 128,
        "batch_size": 32,
        "n_epochs": 4,
        "ent_coef": 0.01,
    }

    if agent_type.lower() == 'ppo':
        agent = UCCPPOAgent(config=agent_config, env=envs[0])
        max_steps = 20  # typical max circuit depth
        # Training loop: alternate geometries
        steps_per_geometry = n_episodes_per_geometry * max_steps
        for env in envs:
            # SB3 PPO can be retrained on a new env by updating the model's env
            try:
                from stable_baselines3.common.vec_env import DummyVecEnv
                vec_env = DummyVecEnv([lambda e=env: e])
                agent.model.set_env(vec_env)
                agent.model.learn(total_timesteps=steps_per_geometry, reset_num_timesteps=False)
            except Exception as e:
                print(f"  Warning: training on env failed: {e}")
    else:
        raise ValueError(f"Unsupported agent_type for transfer: {agent_type}")

    return agent


def evaluate_transfer(
    agent: Any,
    molecule: str,
    bond_lengths_test: List[float],
    operator_pool: str = 'fop',
    n_episodes_finetune: int = 50,
) -> Dict[str, Dict[str, float]]:
    """Evaluate zero-shot, fine-tune, and scratch performance on test geometries.

    Args:
        agent: Trained agent from train_multi_geometry()
        molecule: Molecular formula
        bond_lengths_test: List of test bond lengths
        operator_pool: Operator pool type
        n_episodes_finetune: Episodes for fine-tuning and scratch training

    Returns:
        Dict mapping bond_length -> {zero_shot_error, finetune_error, scratch_error}
        (errors in Hartree)
    """
    import copy
    from rlqas_chem.molecule import process_molecule
    from rlqas_chem.search.ucc.environment import UCCSearchEnv
    from rlqas_chem.search.ucc.controller import UCCPPOAgent
    from stable_baselines3.common.vec_env import DummyVecEnv

    max_steps = 20
    results = {}

    for bl in bond_lengths_test:
        mol = process_molecule(molecule, bl, 'UCC')
        fci_energy = mol.fci_energy
        env = UCCSearchEnv(mol, {
            "run_classical_opt": True,
            "complexity_penalty": 0.0,
        })
        vec_env = DummyVecEnv([lambda e=env: e])

        # Zero-shot: run agent for 1 episode without gradient updates
        zero_shot_energy = float('inf')
        try:
            obs, _ = env.reset()
            done = False
            while not done:
                obs_batch = obs[np.newaxis, :]
                action, _ = agent.model.predict(obs_batch, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated
                zero_shot_energy = info.get("energy", zero_shot_energy)
            # Also check global best
            env_best = getattr(env, 'global_best_energy', zero_shot_energy)
            if env_best < zero_shot_energy:
                zero_shot_energy = env_best
        except Exception as e:
            print(f"  zero_shot error at bl={bl}: {e}")

        zero_shot_err = abs(zero_shot_energy - fci_energy) if zero_shot_energy != float('inf') else None

        # Fine-tune: continue training the transferred agent
        finetune_energy = float('inf')
        try:
            agent_copy = copy.deepcopy(agent)
            finetune_env = UCCSearchEnv(mol, {
                "run_classical_opt": True,
                "complexity_penalty": 0.0,
            })
            vec_finetune = DummyVecEnv([lambda e=finetune_env: e])
            agent_copy.model.set_env(vec_finetune)
            agent_copy.model.learn(
                total_timesteps=n_episodes_finetune * max_steps,
                reset_num_timesteps=False,
            )
            finetune_energy = getattr(finetune_env, 'global_best_energy', float('inf'))
        except Exception as e:
            print(f"  finetune error at bl={bl}: {e}")

        finetune_err = abs(finetune_energy - fci_energy) if finetune_energy != float('inf') else None

        # Scratch: train new agent from scratch
        scratch_energy = float('inf')
        try:
            scratch_env = UCCSearchEnv(mol, {
                "run_classical_opt": True,
                "complexity_penalty": 0.0,
            })
            scratch_agent_config = {
                "use_gpu": False, "seed": 0, "verbose": 0,
                "n_steps": 128, "batch_size": 32, "n_epochs": 4, "ent_coef": 0.01,
            }
            scratch_agent = UCCPPOAgent(config=scratch_agent_config, env=scratch_env)
            vec_scratch = DummyVecEnv([lambda e=scratch_env: e])
            scratch_agent.model.set_env(vec_scratch)
            scratch_agent.model.learn(total_timesteps=n_episodes_finetune * max_steps)
            scratch_energy = getattr(scratch_env, 'global_best_energy', float('inf'))
        except Exception as e:
            print(f"  scratch error at bl={bl}: {e}")

        scratch_err = abs(scratch_energy - fci_energy) if scratch_energy != float('inf') else None

        results[bl] = {
            "zero_shot_error": float(zero_shot_err) if zero_shot_err is not None else None,
            "finetune_error": float(finetune_err) if finetune_err is not None else None,
            "scratch_error": float(scratch_err) if scratch_err is not None else None,
            "fci_energy": float(fci_energy),
            "zero_shot_energy": float(zero_shot_energy) if zero_shot_energy != float('inf') else None,
            "finetune_energy": float(finetune_energy) if finetune_energy != float('inf') else None,
            "scratch_energy": float(scratch_energy) if scratch_energy != float('inf') else None,
        }

        print(f"  bl={bl}: zero_shot_err={zero_shot_err:.4f} Ha, "
              f"finetune_err={finetune_err:.4f} Ha, "
              f"scratch_err={scratch_err:.4f} Ha")

    return results
