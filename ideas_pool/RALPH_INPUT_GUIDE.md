# How to Prepare Input for Ralph (RLQAS Implementation)

## Overview
This guide explains how to structure input for Ralph (AI development assistant) to effectively implement the RLQAS project. Proper input structuring ensures clear understanding, efficient implementation, and successful completion of tasks.

## Core Input Components

### 1. **Task Context Package**
Before starting any task, provide Ralph with:

**A. Project Overview**
```markdown
## RLQAS Project Context
- **Objective**: Modular Reinforcement Learning Quantum Architecture Search
- **Phase**: 1 (of 4) - UCC Search Core Functionality
- **Key Technology Stack**:
  - Quantum: Tencirchem-ng 2024.10, OpenFermion
  - RL: PyTorch, Gym
  - Language: Python 3.8+
- **Reference Documents**:
  1. RLQAS_Ralph_20260205_EN.md (Full specification)
  2. RLQAS_Phase1_Tasks.md (Task breakdown)
  3. This input guide
```

**B. Environment Configuration**
```yaml
# environment.yml or requirements.txt MUST be provided
name: rlqas
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.8
  - numpy>=1.21
  - scipy>=1.7
  - pandas>=1.3
  - tencirchem-ng>=2024.10
  - openfermion>=1.5
  - torch>=1.9
  - gym>=0.21
  - pytest>=7.0
  - black>=22.0
```

### 2. **Task-Specific Input Format**

#### Option A: Markdown Task Card (Recommended)
```markdown
# RLQAS Task Implementation Request

## Task Identification
- **Task ID**: RLQAS_Phase1_001
- **Task Title**: Molecule Processing Module
- **Priority**: P0
- **Dependencies**: None
- **Estimated Time**: 4-6 hours

## Reference Materials
1. **Primary Spec**: Section 3.1 of RLQAS_Ralph_20260205_EN.md
2. **Detailed Tasks**: RLQAS_Phase1_Tasks.md#rlqas_phase1_001
3. **Related Code**: None (first task)

## Implementation Requirements

### Functional Requirements
1. Implement `process_molecule()` function as specified
2. Create `MoleculeData` dataclass with all fields
3. Integrate with Tencirchem-ng 2024.10
4. Support H₂, LiH, BeH₂ test molecules

### Technical Requirements
- **File Structure**:
  ```
  src/modules/molecule_processor.py
  tests/test_molecule_processor.py
  ```
- **Interfaces to Implement**:
  ```python
  @dataclass
  class MoleculeData:
      hamiltonian: QubitOperator
      n_qubits: int
      reference_state: np.ndarray
      fci_energy: float
      molecular_info: Dict

  def process_molecule(
      molecule: str,
      bond_length: float,
      ansatz_type: str,
      active_space: Optional[Tuple[int, int]] = None,
      basis_set: str = "sto-3g",
      transform: str = "parity"
  ) -> MoleculeData
  ```

### Testing Requirements
- **Test Cases**:
  1. H₂ molecule processing (2 qubits)
  2. LiH molecule processing (4 qubits)
  3. Error handling for invalid inputs
- **Coverage**: >80% code coverage
- **Test File**: `tests/test_molecule_processor.py`

### Acceptance Criteria
- [ ] `process_molecule()` correctly processes H₂ molecule
- [ ] `process_molecule()` correctly processes LiH molecule
- [ ] `MoleculeData` contains all required fields
- [ ] Integration with Tencirchem-ng works without errors
- [ ] All unit tests pass with >80% coverage
- [ ] Code follows PEP8 standards

## Deliverables
1. Complete implementation in `src/modules/molecule_processor.py`
2. Comprehensive test suite in `tests/test_molecule_processor.py`
3. Documentation and docstrings
4. Verification that module works with subsequent tasks

## Questions to Clarify (if any)
1. Any uncertainties about Tencirchem-ng integration?
2. Specific format for `QubitOperator`?
3. Error handling strategy for unsupported molecules?
```

#### Option B: JSON Structured Format
```json
{
  "task": {
    "id": "RLQAS_Phase1_001",
    "title": "Molecule Processing Module",
    "priority": "P0",
    "dependencies": [],
    "estimated_hours": 6
  },
  "references": {
    "primary_spec": "RLQAS_Ralph_20260205_EN.md#3.1",
    "detailed_task": "RLQAS_Phase1_Tasks.md#rlqas_phase1_001",
    "related_files": []
  },
  "requirements": {
    "functional": [
      "Implement process_molecule() function",
      "Create MoleculeData dataclass",
      "Integrate with Tencirchem-ng 2024.10",
      "Support H₂, LiH, BeH₂ test molecules"
    ],
    "technical": {
      "file_structure": {
        "src": "modules/molecule_processor.py",
        "tests": "test_molecule_processor.py"
      },
      "interfaces": [
        "MoleculeData dataclass",
        "process_molecule() function"
      ]
    }
  },
  "testing": {
    "test_cases": [
      "H₂ molecule processing",
      "LiH molecule processing",
      "Error handling for invalid inputs"
    ],
    "coverage_threshold": 80,
    "test_file": "tests/test_molecule_processor.py"
  },
  "acceptance_criteria": [
    "process_molecule() correctly processes H₂ molecule",
    "process_molecule() correctly processes LiH molecule",
    "MoleculeData contains all required fields",
    "Integration with Tencirchem-ng works without errors",
    "All unit tests pass with >80% coverage",
    "Code follows PEP8 standards"
  ],
  "deliverables": [
    "Complete implementation in src/modules/molecule_processor.py",
    "Comprehensive test suite in tests/test_molecule_processor.py",
    "Documentation and docstrings",
    "Verification of module functionality"
  ]
}
```

### 3. **Essential Context for Every Task**

Regardless of format, always include:

**A. Project Architecture Context**
```
RLQAS System Flow:
User Input → MoleculeProcessor → QuantumSimulator → RL Agent → Circuit Builder → Evaluator

Current Task Position: [Highlight where this task fits]
```

**B. Key Design Decisions**
- Any architectural patterns to follow (Factory, Strategy, etc.)
- Error handling approach (exceptions, logging, etc.)
- Configuration management strategy

**C. External Dependencies**
- Specific library versions (Tencirchem-ng 2024.10 NOT 2024.9)
- API constraints or limitations
- Hardware/software requirements

### 4. **What NOT to Include**

Avoid:
- Vague requirements ("make it work well")
- Contradictory specifications
- Implementation details that should be left to Ralph's discretion
- Time estimates that may pressure incorrect implementation

### 5. **Progressive Disclosure Strategy**

For complex tasks, provide information in layers:

**Layer 1: Core Requirements** (always include)
- What the module MUST do
- Input/output specifications
- Acceptance criteria

**Layer 2: Implementation Guidance** (include when helpful)
- Suggested architecture patterns
- Reference implementations or similar code
- Performance considerations

**Layer 3: Advanced Details** (include only if critical)
- Optimization requirements
- Edge cases to handle
- Integration specifics

### 6. **Validation Requirements**

Clearly specify how Ralph should validate the implementation:

```markdown
## Validation Instructions

### Self-Test Requirements
1. Run `pytest tests/test_molecule_processor.py -v`
2. Check coverage: `pytest --cov=src/modules/molecule_processor.py`
3. Run PEP8 check: `black --check src/modules/molecule_processor.py`

### Integration Test Requirements
1. Verify module works with sample code:
   ```python
   from src.modules.molecule_processor import process_molecule
   data = process_molecule("H2", 0.74, "UCC")
   assert data.n_qubits == 2
   ```

### Success Reporting
- Provide summary of test results
- Highlight any issues found
- Confirm all acceptance criteria are met
```

### 7. **Communication Protocol**

Define how Ralph should communicate:

**During Implementation**:
- Ask clarifying questions if requirements are unclear
- Report progress at logical milestones
- Flag potential issues early

**Upon Completion**:
- Provide implementation summary
- Include test results and coverage
- Note any deviations from requirements and reasons
- Suggest improvements for future tasks

**For Blockers**:
- Clearly describe the blocking issue
- Provide context and attempted solutions
- Suggest possible alternatives

## Example: Complete Ralph Input for RLQAS_Phase1_001

```markdown
# Ralph: Implement RLQAS_Phase1_001

## Context Package
**Project**: RLQAS Phase 1 - UCC Search Core
**Current Task**: Molecule Processing Module (Foundation)
**Environment**: Python 3.8+, Tencirchem-ng 2024.10, OpenFermion 1.5+
**Reference**: See RLQAS_Ralph_20260205_EN.md section 3.1

## Task: RLQAS_Phase1_001
**Goal**: Create molecule processing module that converts molecular info to quantum computation inputs.

## Requirements
1. Implement `process_molecule()` function with signature:
   ```python
   def process_molecule(
       molecule: str,
       bond_length: float,
       ansatz_type: str,
       active_space: Optional[Tuple[int, int]] = None,
       basis_set: str = "sto-3g",
       transform: str = "parity"
   ) -> MoleculeData
   ```

2. Create `MoleculeData` dataclass with fields:
   - `hamiltonian: QubitOperator`
   - `n_qubits: int`
   - `reference_state: np.ndarray`
   - `fci_energy: float`
   - `molecular_info: Dict`

3. Integrate with Tencirchem-ng 2024.10 for:
   - Molecular integrals
   - Hamiltonian generation
   - FCI energy calculation

## Testing
- Create `tests/test_molecule_processor.py`
- Test cases: H₂ (2 qubits), LiH (4 qubits)
- Achieve >80% code coverage
- Include error case tests

## Acceptance Criteria
- [ ] Function processes H₂ correctly (2 qubits output)
- [ ] Function processes LiH correctly (4 qubits output)
- [ ] All tests pass
- [ ] Code is PEP8 compliant
- [ ] Module can be imported without errors

## Files to Create/Modify
- `src/modules/molecule_processor.py`
- `tests/test_molecule_processor.py`

## Questions to Ask If Unsure
1. How to handle Tencirchem-ng API changes?
2. Specific format for QubitOperator?
3. Error handling approach?

## Deliverables
1. Working molecule processing module
2. Complete test suite
3. Documentation

Please proceed with implementation and report progress.
```

## Best Practices Summary

1. **Be Specific**: Exact function signatures, data structures, test requirements
2. **Provide Context**: Where this task fits in the larger system
3. **Include Examples**: Sample inputs/outputs, configuration examples
4. **Define Success Clearly**: Measurable acceptance criteria
5. **Anticipate Questions**: Address common uncertainties proactively
6. **Enable Validation**: Clear testing requirements and verification steps
7. **Respect Dependencies**: Acknowledge task dependencies and constraints

## Adapting for Different Ralph Versions

If Ralph has specific input format preferences:
- **Text-based Ralph**: Use Markdown format with clear sections
- **API-based Ralph**: Use JSON with structured schema
- **GUI-based Ralph**: Prepare task cards with bullet points

The key is structured, complete, and unambiguous requirements with clear validation criteria.
