# Multi-Agent Network Security System - Usage Guide

## Directory Structure

```
project/
├── adapters/
│   ├── observer/          # Observer LoRA weights
│   ├── responder/         # Responder LoRA weights
│   └── consultant/        # Consultant LoRA weights
├── inference_orchestrator.py
├── test_agent_system.py
├── test_scenarios.json
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install torch transformers peft accelerate
```

## Quick Start

### 1. Run Full Orchestration (Observer → Responder → Consultant)

```bash
python inference_orchestrator.py
```

This will:
- Load all three specialized agents
- Process 5 sample network events
- Show the complete workflow for each event
- Save results to `analysis_results.json`

### 2. Run Comprehensive Testing

```bash
python test_agent_system.py
```

This will:
- Test each agent independently with 10 scenarios
- Evaluate Observer classification accuracy
- Generate detailed test report in `test_report.json`

## Example Output

### Orchestration Output

```
================================================================================
PROCESSING EVENT: High entropy DNS query xk9p2mzqw7h5vbnc4t8s...
================================================================================

Stage 1: OBSERVER ANALYSIS
--------------------------------------------------------------------------------
label: high_risk
reason: Raise alert: suspected DNS tunneling; collect DNS logs for confirmation.


Stage 2: RESPONDER RECOMMENDATION
--------------------------------------------------------------------------------
Classify as potential exfiltration. Recommend: collect DNS logs, block domain 
if confirmed, isolate host pending approval.


Stage 3: CONSULTANT STRATEGIC ANALYSIS
--------------------------------------------------------------------------------
Hypothesis: multi-stage compromise leading to DNS-based exfiltration. Next: IR 
escalation, isolate host, hunt related DNS IOCs, review auth history.

================================================================================
```

## Test Scenarios Included

### High-Risk Events
1. **DNS Tunneling** - High entropy DNS queries
2. **C2 Communication** - Suspicious outbound connections to port 4444
3. **Malicious Process Execution** - PowerShell/WScript execution
4. **Data Exfiltration** - Large data transfers to external IPs

### Medium-Risk Events
5. **Brute Force Attacks** - Multiple failed login attempts
6. **Service Degradation** - Application error spikes

### Low-Risk Events
7. **Normal Authentication** - Successful user logins
8. **Normal Network Traffic** - Standard HTTPS connections

## Custom Event Testing

Create your own events:

```python
from inference_orchestrator import SecurityOrchestrator

# Initialize
orchestrator = SecurityOrchestrator(
    "meta-llama/Llama-3.2-3B",
    "adapters"
)

# Custom event
custom_event = {
    "timestamp": "2026-01-25T15:00:00Z",
    "type": "netflow",
    "details": "Unusual outbound connection from WS-156 to 192.168.1.100:8080 bytes_out=5000000"
}

# Process
result = orchestrator.process_event(custom_event)
```

## Understanding Agent Roles

### Observer Agent
- **Input**: Raw network event
- **Output**: Classification (benign/anomalous/high_risk/staged_attack)
- **Focus**: Pattern recognition and threat detection

### Responder Agent
- **Input**: Event + Observer classification
- **Output**: Tactical response recommendations
- **Focus**: Immediate containment and mitigation

### Consultant Agent
- **Input**: Event + Observer classification
- **Output**: Strategic analysis and long-term recommendations
- **Focus**: Root cause analysis and preventive measures

## Performance Metrics

Based on training results:

| Agent | Training Loss | Dataset Size | Epochs | Training Time |
|-------|--------------|--------------|--------|---------------|
| Observer | 0.898 | 329 examples | 4 | 192s |
| Responder | 1.746 | 128 examples | 4 | 77s |
| Consultant | 1.951 | 43 examples | 10 | 71s |

## Classification Categories

### Observer Classifications

1. **benign**: Normal network activity, no action needed
2. **anomalous**: Unusual but not immediately dangerous, requires monitoring
3. **high_risk**: Likely malicious activity, requires immediate action
4. **staged_attack**: Multi-stage attack detected, requires full incident response

## Output Files

### analysis_results.json
Contains full orchestration results with all three agent responses for each event.

### test_report.json
Contains comprehensive test results including:
- Observer classification accuracy
- All agent responses for each scenario
- Metadata and statistics

## Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size or use CPU offloading
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    max_memory={0: "10GB", "cpu": "30GB"}
)
```

### Adapter Not Found
Ensure your directory structure matches:
```
adapters/
  observer/
    adapter_config.json
    adapter_model.safetensors
  responder/
    ...
  consultant/
    ...
```

## API Reference

### SecurityOrchestrator

```python
orchestrator = SecurityOrchestrator(base_model_name, adapters_dir)

# Process single event
result = orchestrator.process_event(event_dict)

# Process multiple events
results = orchestrator.process_events_batch(events_list)
```

### Event Format

```python
event = {
    "timestamp": "ISO 8601 format",
    "type": "dns|netflow|auth|endpoint|app",
    "details": "Human-readable event description"
}
```

## Best Practices

1. **Always review high-impact actions** marked with `[requires_approval]`
2. **Monitor false positives** and retrain agents with new examples
3. **Correlate alerts** across multiple events for better context
4. **Keep training data updated** with recent attack patterns
5. **Use consultant recommendations** for long-term security posture improvement

## Citation

If using this system in research, please cite:

```bibtex
@software{multi_agent_security_2026,
  title={Multi-Agent Network Security Monitoring System},
  author={Your Name},
  year={2026},
  description={LoRA-based specialized agents for network security monitoring}
}
```
