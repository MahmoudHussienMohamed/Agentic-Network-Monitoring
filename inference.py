"""
Multi-Agent Network Security Monitoring System
Orchestrates Observer -> Responder -> Consultant workflow
FIXED: Proper stopping criteria and output parsing
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
        print(f"Loading {agent_name} agent...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()
        print(f"✓ {agent_name} loaded")
        
    def generate_response(self, prompt: str, max_new_tokens: int = 150) -> str:
        """Generate response with proper stopping criteria"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=10,
                temperature=0.3,  # Lower temperature for more focused output
                top_p=0.85,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,  # Penalize repetition
                no_repeat_ngram_size=3,  # Prevent 3-gram repetition
                # early_stopping=True
            )
        
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        generated_text = self._clean_response(generated_text)
        
        return generated_text.strip()
    
    def _clean_response(self, text: str) -> str:
        """Clean up response by removing repetitions and incomplete sentences"""
        stop_phrases = [
            "Task:",
            "Role:",
            "Event time:",
            "Based on the observer",
            "Provide a single-line",
            "\n\n\n"
        ]
        
        for phrase in stop_phrases:
            if phrase in text:
                text = text.split(phrase)[0]
        
        # Take only first few complete sentences (max 5 lines)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if len(lines) > 5:
            lines = lines[:5]
        
        return '\n'.join(lines)


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

        response = self.generate_response(prompt, max_new_tokens=100)
        return response


class ResponderAgent(SecurityAgent):
    """Provides tactical response recommendations"""
    
    def recommend_action(self, event: Dict, observer_classification: str) -> str:
        """Generate response recommendation"""
        prompt = f"""Role: responder
Observer classification: {observer_classification}
Event time: {event['timestamp']}
Event type: alert
Event details: Alert received: {event['details']}

Task: Based on the observer's classification and the event details, propose a concise, safe, and actionable response.
If the action is high-impact (network isolation, credential reset, blocking IP), include "ACTION:" with a one-line command suggestion and require human approval tag "[requires_approval]".
Also include a one-line "RATIONALE:" explaining why."""

        response = self.generate_response(prompt, max_new_tokens=120)
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

        response = self.generate_response(prompt, max_new_tokens=150)
        return response


class SecurityOrchestrator:
    """Orchestrates the multi-agent workflow"""
    
    def __init__(self, base_model_name: str, adapters_dir: str):
        print("="*80)
        print("INITIALIZING MULTI-AGENT SECURITY SYSTEM")
        print("="*80)
        
        self.observer = ObserverAgent(
            base_model_name,
            os.path.join(adapters_dir, "observer"),
            "Observer"
        )
        
        self.responder = ResponderAgent(
            base_model_name,
            os.path.join(adapters_dir, "responder"),
            "Responder"
        )
        
        self.consultant = ConsultantAgent(
            base_model_name,
            os.path.join(adapters_dir, "consultant"),
            "Consultant"
        )
        
        print("="*80)
        print("ALL AGENTS READY")
        print("="*80 + "\n")
    
    def process_event(self, event: Dict) -> Dict:
        """Process security event through all three agents"""
        print(f"\n{'='*80}")
        print(f"EVENT: {event['details'][:70]}...")
        print(f"{'='*80}\n")
        
        # Stage 1: Observer Classification
        print("│ STAGE 1: OBSERVER ANALYSIS")
        print("├" + "─"*78)
        observer_response = self.observer.analyze_event(event)
        print(f"│ {observer_response.replace(chr(10), chr(10) + '│ ')}")
        
        # Extract classification
        classification = self._extract_classification(observer_response)
        print(f"│\n│ → Classification: {classification.upper()}")
        
        # Stage 2: Responder Recommendation
        print(f"\n│ STAGE 2: RESPONDER RECOMMENDATION")
        print("├" + "─"*78)
        responder_response = self.responder.recommend_action(event, classification)
        print(f"│ {responder_response.replace(chr(10), chr(10) + '│ ')}")
        
        # Stage 3: Consultant Strategic Analysis
        print(f"\n│ STAGE 3: CONSULTANT STRATEGIC ANALYSIS")
        print("├" + "─"*78)
        consultant_response = self.consultant.analyze_correlation(event, classification)
        print(f"│ {consultant_response.replace(chr(10), chr(10) + '│ ')}")
        
        print(f"\n{'='*80}\n")
        
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
        
        return result
    
    def _extract_classification(self, response: str) -> str:
        """Extract classification label from observer response"""
        for line in response.split('\n'):
            line_lower = line.lower().strip()
            if line_lower.startswith('label:'):
                label = line.split(':', 1)[1].strip()
                # Clean any extra text after the label
                for word in ['benign', 'anomalous', 'high_risk', 'staged_attack']:
                    if word in label.lower():
                        return word
        return "anomalous"  # Default if not found
    
    def process_events_batch(self, events: List[Dict]) -> List[Dict]:
        """Process multiple events"""
        results = []
        total = len(events)
        
        for i, event in enumerate(events, 1):
            print(f"\n{'#'*80}")
            print(f"# EVENT {i}/{total}")
            print(f"{'#'*80}")
            
            result = self.process_event(event)
            results.append(result)
        
        return results


def create_test_events() -> List[Dict]:
    """Create sample test events"""
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
    ADAPTERS_DIR = "adapters"
    
    # Initialize orchestrator (this loads all models - takes time)
    orchestrator = SecurityOrchestrator(BASE_MODEL, ADAPTERS_DIR)
    
    # Create test events
    test_events = create_test_events()
    
    # Process events
    print("\n" + "="*80)
    print("STARTING MULTI-AGENT SECURITY ANALYSIS")
    print(f"Processing {len(test_events)} events...")
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
        print(f"  • {cls}: {count}")
    
    print(f"\nTotal events processed: {len(results)}")
    print("="*80)


if __name__ == "__main__":
    main()