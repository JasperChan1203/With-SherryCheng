"""Hyperparameter optimization using Optuna for rlqas-chem."""
from __future__ import annotations
from typing import Optional, Tuple, Dict, Any


def optimize_hyperparams(
    molecule: str,
    bond_length: float,
    agent_type: str = 'ppo',
    operator_pool: str = 'fop',
    n_trials: int = 50,
    n_episodes_per_trial: int = 150,
    active_space: Optional[Tuple[int, int]] = None,
    alpha: float = 1.0,
) -> Dict[str, Any]:
    """Find best RL hyperparameters using Optuna TPE sampler.

    Args:
        molecule: Molecular formula, e.g. "H2", "LiH"
        bond_length: Bond length in Angstroms
        agent_type: RL agent type ("ppo", "a2c", "dqn", "grpo")
        operator_pool: Operator pool type ("fop", "qop")
        n_trials: Number of Optuna trials
        n_episodes_per_trial: Episodes per trial
        active_space: Optional (n_electrons, n_orbitals) tuple
        alpha: Pareto alpha parameter (1.0 = energy only)

    Returns:
        dict with keys: best_params, best_energy, study
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError(
            "optuna is required for hyperparameter optimization. "
            "Install it with: pip install optuna>=3.0"
        )

    import rlqas_chem

    def objective(trial: 'optuna.Trial') -> float:
        # Common hyperparameters for all agents
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        gamma = trial.suggest_float("gamma", 0.9, 0.999)
        max_ops = trial.suggest_int("max_operators", 5, 20)

        # Agent-specific hyperparameters
        agent_config = {}
        if agent_type in ("ppo", "a2c"):
            ent_coef = trial.suggest_float("ent_coef", 1e-4, 0.1, log=True)
            n_steps = trial.suggest_categorical("n_steps", [64, 128, 256])
            agent_config = {
                "learning_rate": lr,
                "gamma": gamma,
                "ent_coef": ent_coef,
                "n_steps": n_steps,
            }
        elif agent_type == "dqn":
            epsilon_start = trial.suggest_float("epsilon_start", 0.5, 1.0)
            epsilon_end = trial.suggest_float("epsilon_end", 0.01, 0.1)
            agent_config = {
                "learning_rate": lr,
                "gamma": gamma,
                "epsilon_start": epsilon_start,
                "epsilon_end": epsilon_end,
            }
        elif agent_type == "grpo":
            group_size = trial.suggest_categorical("group_size", [2, 4, 8])
            clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
            agent_config = {
                "lr": lr,
                "gamma": gamma,
                "group_size": group_size,
                "clip_range": clip_range,
            }
        else:
            agent_config = {"learning_rate": lr, "gamma": gamma}

        try:
            result = rlqas_chem.search(
                molecule=molecule,
                bond_length=bond_length,
                ansatz_type="UCC",
                agent_type=agent_type,
                n_episodes=n_episodes_per_trial,
                active_space=active_space,
                operator_pool=operator_pool,
                alpha=alpha,
                max_operators=max_ops,
                config=agent_config,
            )
            return result.get("best_energy", float("inf"))
        except Exception:
            return float("inf")

    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_trial = study.best_trial
    best_params = best_trial.params
    best_energy = best_trial.value

    return {
        "best_params": best_params,
        "best_energy": best_energy,
        "study": study,
    }
