# RLQAS Phase 1 - PPO RL Agent Knowledge Base

## Implementation Patterns

### RL Agent Design Patterns
- **Abstract Base Class**: RLAgent provides generic Gym-compatible interface
- **Wrapper Pattern**: PPOAgent wraps Stable-Baselines3 implementation
- **Configuration Management**: Centralized config with validation and defaults
- **UCC Compatibility**: Non-abstract helper methods for quantum chemistry integration

### Stable-Baselines3 Integration Patterns
- **Model Wrapping**: SB3 model encapsulated in custom agent class
- **Device Management**: Automatic GPU detection with CPU fallback
- **Save/Load**: Use SB3's native .zip format for model persistence
- **Policy Configuration**: MlpPolicy as default, configurable for different architectures

### Testing Patterns
- **Reproducibility**: Fixed random seeds for all stochastic components
- **Two-Phase Testing**: CartPole for basic RL, UCC-compatible env for integration
- **Coverage Requirement**: >80% test coverage for module
- **Integration Testing**: Import Task 001 modules for UCC compatibility validation

## Technology Insights

### Stable-Baselines3
- **Version**: >=2.0.0 required
- **PPO Implementation**: Reliable, well-tested PPO algorithm
- **Policy Types**: MlpPolicy, CnnPolicy, MultiInputPolicy available
- **Training Interface**: learn() method with total_timesteps parameter
- **Model Saving**: save() creates .zip file with model and metadata

### PyTorch Device Management
```python
import torch

def get_device(use_gpu=True):
    """Automatic device detection."""
    if use_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"
```

### Random Seed Management
```python
import numpy as np
import torch
import random

def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

## UCC Compatibility Design

### State Formatting
- **Input**: Energy (float) + circuit parameters (np.ndarray)
- **Output**: Concatenated state vector for agent
- **Rationale**: Agent needs both energy and parameter information for decision making

### Action Parsing
- **Input**: Action index from agent (int)
- **Output**: Excitation operator information (Dict)
- **Rationale**: Map agent actions to quantum circuit operations

### Interface Philosophy
- **Generic First**: RLAgent interface works with any Gym environment
- **Specialized Helpers**: UCC-specific methods as non-abstract helpers
- **Extensibility**: Easy to add other domain-specific helpers

## Configuration Management

### Default Parameters
From RLQAS specification section 3.3:
- learning_rate: 3e-4
- gamma: 0.99
- gae_lambda: 0.95
- clip_range: 0.2
- ent_coef: 0.01
- vf_coef: 0.5
- max_grad_norm: 0.5
- n_steps: 2048
- batch_size: 64
- n_epochs: 10

### Validation Rules
- learning_rate: positive float
- gamma: float in (0, 1]
- n_steps: positive integer
- batch_size: positive integer <= n_steps
- seed: integer

## Testing Strategies

### CartPole-v1 Validation
- **Purpose**: Verify basic RL functionality
- **Success Criteria**: Reward > 100 within reasonable training
- **Seed Management**: Fixed seed for reproducibility
- **Training Steps**: ~10,000 timesteps sufficient for basic validation

### UCC Compatibility Testing
- **Purpose**: Verify integration with quantum chemistry workflow
- **Data Source**: Task 001 molecule processor (H2 molecule)
- **Test Environment**: Simple UCC-compatible environment
- **Validation**: Helper methods work correctly with real data

### Save/Load Testing
- **Format**: SB3 .zip format
- **Procedure**: Train → Save → Load → Compare performance
- **Validation**: Loaded model performs identically

## Common Pitfalls and Solutions

### 1. SB3 Model Initialization
**Problem**: SB3 model requires environment at initialization
**Solution**: Initialize model lazily or pass dummy environment for configuration

### 2. Random Seed Conflicts
**Problem**: Multiple libraries with different seed setting methods
**Solution**: Use comprehensive set_seed() function covering all libraries

### 3. GPU Memory Issues
**Problem**: GPU out of memory during training
**Solution**: Automatic fallback to CPU, configurable memory limits

### 4. Dependency Version Conflicts
**Problem**: SB3, PyTorch, Gym version incompatibilities
**Solution**: Use tested version combinations, document requirements clearly

## Integration Notes

### With Task 001 (Molecule Processing)
- Import path: `sys.path.append("../001")`
- Used for: UCC-compatible test environments
- Key classes: `MoleculeData`, `process_molecule()`

### For Task 004 (UCC Search)
- Interface: RLAgent abstract methods
- Helpers: UCC state/action formatting methods
- Configuration: Consistent hyperparameter defaults

## Performance Considerations

### Training Speed
- **CartPole**: Should train in minutes on CPU
- **GPU Acceleration**: 2-5x speedup if available
- **Memory**: Moderate (~2-4GB for typical usage)

### Model Size
- **SB3 Models**: Typically 1-10MB depending on policy architecture
- **Checkpoints**: Include optimizer state, larger than final model

## Documentation Standards

### Code Documentation
- All public methods: Google-style docstrings
- Type hints: For all function signatures
- Examples: In docstrings for complex methods

### User Documentation
- Configuration examples
- Training examples with CartPole
- UCC integration guide
- Troubleshooting common issues

## Implementation Experience

### SB3 Integration Insights
- **Policy mapping**: SB3 uses policy class names (ActorCriticPolicy, ActorCriticCnnPolicy, MultiInputActorCriticPolicy) internally but expects policy type strings ("MlpPolicy", "CnnPolicy", "MultiInputPolicy") for initialization. Need to map between them when loading saved models.
- **Gymnasium transition**: SB3 now uses Gymnasium internally and automatically wraps OpenAI Gym environments via shimmy. Users should install shimmy>=2.0 for compatibility.
- **GPU warnings**: SB3 warns that MLP policies are primarily intended for CPU; GPU utilization may be poor. Automatic device detection handles this gracefully.
- **Environment persistence**: SB3 does not save the environment with the model; `model.env` may be `None` after loading. Users need to set environment separately if needed.

### Configuration Validation Patterns
- **Type and value validation**: Use separate rules for each parameter with descriptive error messages.
- **Cross-parameter validation**: Additional checks for relationships (e.g., `batch_size <= n_steps`).
- **Unknown parameter rejection**: Strict validation prevents typos in configuration keys.

### Testing Strategies
- **Mock vs real dependencies**: Use real SB3/Gym for integration tests but mock heavy dependencies for unit tests.
- **Slow test marking**: Use `@pytest.mark.slow` for tests with expensive computations (e.g., molecule processing).
- **Coverage targets**: Achieve >80% coverage across all modules; focus on critical paths rather than trivial getters/setters.

### UCC Compatibility Design
- **Helper method defaults**: Simple concatenation for state formatting and minimal action parsing provide sensible defaults that can be overridden by subclasses.
- **Real molecule data integration**: Successful import and use of Task 001 modules validates the interface design.