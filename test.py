"""
Comprehensive Testing Script for Multi-Agent Security System
Tests all three agents with various scenarios and generates detailed reports
"""

import os
import json
from datetime import datetime
from typing import Dict, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


class AgentTester:
    """Test individual agents with specific scenarios"""
    
    def __init__(self, base_model_name: str, adapter_path: str, agent_name: str):
        self.agent_name = agent_name
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()
        
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip()


def test_observer(tester: AgentTester, scenarios: List[Dict]) -> List[Dict]:
    """Test Observer agent"""
    print("\n" + "="*80)
    print("TESTING OBSERVER AGENT")
    print("="*80)
    
    results = []
    
    for i, scenario in enumerate(scenarios, 1):
        event = scenario['event']
        
        prompt = f"""Role: observer
Event time: {event['timestamp']}
Event type: {event['type']}
Event details: {event['details']}

Task: Classify the event's vitality into exactly ONE of: [benign, anomalous, high_risk, staged_attack].
Provide a single-line label and a brief (1-2 sentence) justification.
Format:
label: <one of the four>
reason: <brief justification>"""
        
        print(f"\nTest {i}/{len(scenarios)}: {scenario['name']}")
        print("-" * 80)
        print(f"Event: {event['details']}")
        
        response = tester.generate(prompt)
        print(f"\nObserver Response:\n{response}")
        
        # Extract classification
        classification = "unknown"
        for line in response.split('\n'):
            if line.lower().startswith('label:'):
                classification = line.split(':', 1)[1].strip()
                break
        
        # Check against expected
        expected = scenario.get('expected_observer_classification', '')
        match = classification.lower() == expected.lower()
        
        print(f"\nClassification: {classification}")
        print(f"Expected: {expected}")
        print(f"Match: {'✓' if match else '✗'}")
        
        results.append({
            "scenario_id": scenario['scenario_id'],
            "scenario_name": scenario['name'],
            "event": event,
            "response": response,
            "classification": classification,
            "expected": expected,
            "match": match
        })
    
    return results


def test_responder(tester: AgentTester, scenarios: List[Dict]) -> List[Dict]:
    """Test Responder agent"""
    print("\n" + "="*80)
    print("TESTING RESPONDER AGENT")
    print("="*80)
    
    results = []
    
    for i, scenario in enumerate(scenarios, 1):
        event = scenario['event']
        classification = scenario.get('expected_observer_classification', 'anomalous')
        
        prompt = f"""Role: responder
Observer classification: {classification}
Event time: {event['timestamp']}
Event type: alert
Event details: Alert received: {event['details']}

Task: Based on the observer's classification and the event details, propose a concise, safe, and actionable response.
If the action is high-impact (network isolation, credential reset, blocking IP), include "ACTION:" with a one-line command suggestion and require human approval tag "[requires_approval]".
Also include a one-line "RATIONALE:" explaining why."""
        
        print(f"\nTest {i}/{len(scenarios)}: {scenario['name']}")
        print("-" * 80)
        print(f"Classification: {classification}")
        print(f"Event: {event['details']}")
        
        response = tester.generate(prompt)
        print(f"\nResponder Response:\n{response}")
        
        results.append({
            "scenario_id": scenario['scenario_id'],
            "scenario_name": scenario['name'],
            "event": event,
            "classification": classification,
            "response": response
        })
    
    return results


def test_consultant(tester: AgentTester, scenarios: List[Dict]) -> List[Dict]:
    """Test Consultant agent"""
    print("\n" + "="*80)
    print("TESTING CONSULTANT AGENT")
    print("="*80)
    
    results = []
    
    for i, scenario in enumerate(scenarios, 1):
        event = scenario['event']
        classification = scenario.get('expected_observer_classification', 'anomalous')
        
        prompt = f"""Role: consultant
Event time: {event['timestamp']}
Event type: correlation
Event details: Correlation analysis for {event['details']}
Observer classification (if available): {classification}

Task: Produce a short analytical report (3-6 sentences) summarizing the event, impact assessment, recommended next steps, and suggested monitoring queries or checks."""
        
        print(f"\nTest {i}/{len(scenarios)}: {scenario['name']}")
        print("-" * 80)
        print(f"Classification: {classification}")
        print(f"Event: {event['details']}")
        
        response = tester.generate(prompt)
        print(f"\nConsultant Response:\n{response}")
        
        results.append({
            "scenario_id": scenario['scenario_id'],
            "scenario_name": scenario['name'],
            "event": event,
            "classification": classification,
            "response": response
        })
    
    return results


def generate_test_report(observer_results, responder_results, consultant_results, output_file="test_report.json"):
    """Generate comprehensive test report"""
    
    # Calculate observer accuracy
    observer_matches = sum(1 for r in observer_results if r['match'])
    observer_accuracy = (observer_matches / len(observer_results) * 100) if observer_results else 0
    
    report = {
        "test_metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_scenarios": len(observer_results),
            "observer_accuracy": f"{observer_accuracy:.1f}%"
        },
        "observer_results": {
            "total_tests": len(observer_results),
            "correct_classifications": observer_matches,
            "accuracy": observer_accuracy,
            "details": observer_results
        },
        "responder_results": {
            "total_tests": len(responder_results),
            "details": responder_results
        },
        "consultant_results": {
            "total_tests": len(consultant_results),
            "details": consultant_results
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Test report saved to {output_file}")
    return report


def print_summary(report: Dict):
    """Print test summary"""
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    print(f"\nObserver Agent:")
    print(f"  Total Tests: {report['observer_results']['total_tests']}")
    print(f"  Correct Classifications: {report['observer_results']['correct_classifications']}")
    print(f"  Accuracy: {report['observer_results']['accuracy']:.1f}%")
    
    print(f"\nResponder Agent:")
    print(f"  Total Tests: {report['responder_results']['total_tests']}")
    
    print(f"\nConsultant Agent:")
    print(f"  Total Tests: {report['consultant_results']['total_tests']}")
    
    print("\n" + "="*80)


def main():
    """Main test execution"""
    # Configuration
    BASE_MODEL = "meta-llama/Llama-3.2-3B"
    ADAPTERS_DIR = "adapters"
    TEST_SCENARIOS_FILE = "test_scenarios.json"
    
    # Load test scenarios
    print("Loading test scenarios...")
    with open(TEST_SCENARIOS_FILE, 'r') as f:
        data = json.load(f)
        scenarios = data['test_scenarios']
    
    print(f"Loaded {len(scenarios)} test scenarios\n")
    
    # Initialize testers
    print("Initializing agents...")
    
    observer_tester = AgentTester(
        BASE_MODEL,
        os.path.join(ADAPTERS_DIR, "observer"),
        "observer"
    )
    print("✓ Observer agent loaded")
    
    responder_tester = AgentTester(
        BASE_MODEL,
        os.path.join(ADAPTERS_DIR, "responder"),
        "responder"
    )
    print("✓ Responder agent loaded")
    
    consultant_tester = AgentTester(
        BASE_MODEL,
        os.path.join(ADAPTERS_DIR, "consultant"),
        "consultant"
    )
    print("✓ Consultant agent loaded")
    
    # Run tests
    observer_results = test_observer(observer_tester, scenarios)
    responder_results = test_responder(responder_tester, scenarios)
    consultant_results = test_consultant(consultant_tester, scenarios)
    
    # Generate report
    report = generate_test_report(observer_results, responder_results, consultant_results)
    
    # Print summary
    print_summary(report)


if __name__ == "__main__":
    main()
