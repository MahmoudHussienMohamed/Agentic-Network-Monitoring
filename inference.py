"""
Multi-Agent Network Security Monitoring System
Orchestrates Observer -> Responder -> Consultant workflow
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


class SecurityAgent:
    """Base class for specialized security agents"""
    
    def __init__(self, base_model_name: str, adapter_path: str, agent_name: str):
        self.agent_name = agent_name
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()
        
    def generate_response(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate response from the agent"""
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
        
        # Decode only the generated tokens (exclude prompt)
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return generated_text.strip()


class ObserverAgent(SecurityAgent):
    """Monitors network events and classifies threats"""
    
    def analyze_event(self, event: Dict) -> Tuple[str, str]:
        """Analyze network event and return classification"""
        prompt = f"""Role: observer
Event time: {event['timestamp']}
Event type: {event['type']}
Event details: {event['details']}

Task: Classify the event's vitality into exactly ONE of: [benign, anomalous, high_risk, staged_attack].
Provide a single-line label and a brief (1-2 sentence) justification.
Format:
label: <one of the four>
reason: <brief justification>"""

        response = self.generate_response(prompt)
        return response


class ResponderAgent(SecurityAgent):
    """Provides tactical response recommendations"""
    
    def recommend_action(self, event: Dict, observer_classification: str) -> str:
        """Generate response recommendation based on observer's classification"""
        prompt = f"""Role: responder
Observer classification: {observer_classification}
Event time: {event['timestamp']}
Event type: alert
Event details: Alert received: {event['details']}

Task: Based on the observer's classification and the event details, propose a concise, safe, and actionable response.
If the action is high-impact (network isolation, credential reset, blocking IP), include "ACTION:" with a one-line command suggestion and require human approval tag "[requires_approval]".
Also include a one-line "RATIONALE:" explaining why."""

        response = self.generate_response(prompt)
        return response


class ConsultantAgent(SecurityAgent):
    """Provides strategic analysis and recommendations"""
    
    def analyze_correlation(self, event: Dict, observer_classification: str) -> str:
        """Provide strategic analysis"""
        prompt = f"""Role: consultant
Event time: {event['timestamp']}
Event type: correlation
Event details: Correlation analysis for {event['details']}
Observer classification (if available): {observer_classification}

Task: Produce a short analytical report (3-6 sentences) summarizing the event, impact assessment, recommended next steps, and suggested monitoring queries or checks."""

        response = self.generate_response(prompt)
        return response


class SecurityOrchestrator:
    """Orchestrates the multi-agent workflow"""
    
    def __init__(self, base_model_name: str, adapters_dir: str):
        print("Loading agents...")
        self.observer = ObserverAgent(
            base_model_name,
            os.path.join(adapters_dir, "observer"),
            "observer"
        )
        print("✓ Observer agent loaded")
        
        self.responder = ResponderAgent(
            base_model_name,
            os.path.join(adapters_dir, "responder"),
            "responder"
        )
        print("✓ Responder agent loaded")
        
        self.consultant = ConsultantAgent(
            base_model_name,
            os.path.join(adapters_dir, "consultant"),
            "consultant"
        )
        print("✓ Consultant agent loaded")
        print("All agents ready!\n")
    
    def process_event(self, event: Dict) -> Dict:
        """Process security event through all three agents"""
        print(f"\n{'='*80}")
        print(f"PROCESSING EVENT: {event['details'][:60]}...")
        print(f"{'='*80}\n")
        
        # Stage 1: Observer Classification
        print("Stage 1: OBSERVER ANALYSIS")
        print("-" * 80)
        observer_response = self.observer.analyze_event(event)
        print(observer_response)
        
        # Extract classification label
        classification = "benign"
        for line in observer_response.split('\n'):
            if line.lower().startswith('label:'):
                classification = line.split(':', 1)[1].strip()
                break
        
        # Stage 2: Responder Recommendation
        print("\n\nStage 2: RESPONDER RECOMMENDATION")
        print("-" * 80)
        responder_response = self.responder.recommend_action(event, classification)
        print(responder_response)
        
        # Stage 3: Consultant Strategic Analysis
        print("\n\nStage 3: CONSULTANT STRATEGIC ANALYSIS")
        print("-" * 80)
        consultant_response = self.consultant.analyze_correlation(event, classification)
        print(consultant_response)
        
        # Compile results
        result = {
            "event": event,
            "timestamp": datetime.now().isoformat(),
            "observer": {
                "classification": classification,
                "response": observer_response
            },
            "responder": {
                "response": responder_response
            },
            "consultant": {
                "response": consultant_response
            }
        }
        
        print(f"\n{'='*80}\n")
        return result
    
    def process_events_batch(self, events: List[Dict]) -> List[Dict]:
        """Process multiple events"""
        results = []
        for i, event in enumerate(events, 1):
            print(f"\n\n### EVENT {i}/{len(events)} ###")
            result = self.process_event(event)
            results.append(result)
        
        return results


def create_test_events() -> List[Dict]:
    """Create sample test events based on training data patterns"""
    return [
        {
            "timestamp": "2026-01-25T14:30:45Z",
            "type": "dns",
            "details": "High entropy DNS query xk9p2mzqw7h5vbnc4t8s.cdn.verify.ru from WS-142"
        },
        {
            "timestamp": "2026-01-25T14:31:12Z",
            "type": "netflow",
            "details": "Unusual outbound connection from WS-089 to 10.45.123.67:4444 bytes_out=1523847"
        },
        {
            "timestamp": "2026-01-25T14:31:45Z",
            "type": "auth",
            "details": "Multiple failed logins (23) for user admin on SRV-12 from 10.88.192.45"
        },
        {
            "timestamp": "2026-01-25T14:32:00Z",
            "type": "endpoint",
            "details": "Suspicious process powershell executed on WS-205 by user jdoe"
        },
        {
            "timestamp": "2026-01-25T14:32:30Z",
            "type": "app",
            "details": "Service api_gateway error 500 spike on SRV-45"
        }
    ]


def save_results(results: List[Dict], output_file: str = "analysis_results.json"):
    """Save analysis results to JSON file"""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")


def main():
    """Main execution function"""
    # Configuration
    BASE_MODEL = "meta-llama/Llama-3.2-3B"
    ADAPTERS_DIR = "adapters"  # Directory containing observer/, responder/, consultant/
    
    # Initialize orchestrator
    print("Initializing Security Orchestration System...")
    print(f"Base Model: {BASE_MODEL}")
    print(f"Adapters Directory: {ADAPTERS_DIR}\n")
    
    orchestrator = SecurityOrchestrator(BASE_MODEL, ADAPTERS_DIR)
    
    # Create test events
    test_events = create_test_events()
    
    # Process events
    print("\n" + "="*80)
    print("STARTING MULTI-AGENT SECURITY ANALYSIS")
    print("="*80)
    
    results = orchestrator.process_events_batch(test_events)
    
    # Save results
    save_results(results)
    
    # Summary
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    
    classifications = {}
    for result in results:
        cls = result['observer']['classification']
        classifications[cls] = classifications.get(cls, 0) + 1
    
    print("\nClassification Distribution:")
    for cls, count in sorted(classifications.items()):
        print(f"  {cls}: {count}")
    
    print(f"\nTotal events processed: {len(results)}")
    print("="*80)


if __name__ == "__main__":
    main()
