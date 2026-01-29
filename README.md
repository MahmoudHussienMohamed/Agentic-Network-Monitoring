# Agentic Network Monitoring

## 🚀 Project Overview
This project implements a **Multi-Agent Network Security Monitoring System** using a fine-tuned **Llama 3.2-3B** foundation model. It is designed to autonomously monitor private networks, detect security threats, and provide actionable response strategies.

The system mimics a real-world Security Operations Center (SOC) through a hierarchical architecture of three specialized agents:
1.  **Observer Agent**: Monitors network traffic and detects anomalies.
2.  **Responder Agent**: Provides immediate tactical mitigation and response plans.
3.  **Consultant Agent**: Offers strategic long-term security analysis and policy recommendations.

Youtube demo [here](https://youtu.be/TWuxiYFSvaY).

---

## 💡 Why Fine-Tuning? (vs. Raw LLMs)
While large foundational models (like GPT-4 or Llama 3 70B) are capable generalists, they often lack the specific nuance required for specialized network security tasks. This project leverages **Parameter-Efficient Fine-Tuning (LoRA)** to create highly specialized agents.

### Key Advantages:
*   **Domain Expertise**: Each agent is fine-tuned on a curated dataset of security logs, incident response playbooks, and strategic assessments, resulting in higher accuracy for specific tasks (e.g., distinguishing between legitimate admin activity and brute-force attacks).
*   **Operational Efficiency**: By using a smaller 3B parameter model with LoRA adapters, the entire system can run locally on consumer-grade hardware (e.g., NVIDIA RTX 3060/4060 class), enabling **real-time inference**.
*   **Data Privacy**: Unlike cloud-based LLMs that require sending sensitive network logs to external APIs, this local system allows for complete data sovereignty—critical for private and secure network environments.
*   **Structured & Consistent Output**: specialized training ensures agents produce output in expected JSON or structured text formats, making them easier to integrate into automated pipelines.

---

## 🏗️ System Architecture

### 1. Observer Agent (The "Eyes")
*   **Role**: Monitoring & Detection.
*   **Input**: Raw network logs (DNS queries, NetFlow data, Auth logs).
*   **Function**: Analyzes high-entropy domains, unusual outbound connections, and login failures.
*   **Output**: Classification (`Normal` vs `Anomalous`) and initial triage notes.

### 2. Responder Agent (The "Hands")
*   **Role**: Tactical Response.
*   **Input**: Anomalous events flagged by the Observer.
*   **Function**: Generates immediate containment steps (e.g., "Block IP," "Isolate Host," "Reset Credentials").
*   **Output**: Actionable playbooks and remediation commands.

### 3. Consultant Agent (The "Brain")
*   **Role**: Strategic Analysis.
*   **Input**: Event context and Responder actions.
*   **Function**: Correlates events over time to detect APTs (Advanced Persistent Threats) or coordinated attacks.
*   **Output**: Strategic reports, policy updates, and long-term security recommendations.

---

## 📂 Project Structure
```text
Agentic-Network-Monitoring/
├── adapters/              # Saved LoRA adapters for each agent
├── data/                  # Training datasets (.jsonl)
├── train.py               # Main training script (PEFT/LoRA)
├── inference.py           # Multi-agent simulation script
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
├── Paper.md               # Research paper & methodology details
└── analysis_results.json  # Output from the inference pipeline
```

---


## 📊 Training Results & Performance
The system uses **LoRA (Low-Rank Adaptation)** to fine-tune only ~0.37% of the total parameters (12M of 3.2B), ensuring efficiency without catastrophic forgetting.

### Training Metrics (4 Epochs)
| Agent | Initial Loss | Final Loss | Improvement | Focus Area |
| :--- | :--- | :--- | :--- | :--- |
| **Observer** | 3.65 | **0.39** | ~89% | Anomaly Detection |
| **Responder** | 3.76 | **0.55** | ~85% | Incident Response |
| **Consultant** | 3.91 | **0.66** | ~83% | Strategic Reasoning |

*Hardware Used: NVIDIA RTX 5060 Ti (16GB VRAM) with FP16 precision.*

---

## 🛠️ Prerequisites & Setup

### Requirements
*   **OS**: Windows 10/11 or Linux.
*   **Python**: Version 3.13+ recommended.
*   **Hardware**: CUDA-capable GPU with at least 8GB VRAM (for inference) / 16GB (for training).

### Installation
1.  **Clone the repository**:
    ```powershell
    git clone https://github.com/MahmoudHussienMohamed/Agentic-Network-Monitoring.git
    cd Agentic-Network-Monitoring
    ```

2.  **Create a virtual environment**:
    ```powershell
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```powershell
    pip install -r requirements.txt
    # Ensure PyTorch with CUDA support is installed:
    # pip install torch --index-url https://download.pytorch.org/whl/cu121
    ```

---

## 💻 Usage

### 1. Training the Agents
To reproduce the training results, run the `train.py` script for each agent. This will generate the LoRA adapters in the `adapters/` directory.

```powershell
# Train Observer
python train.py observer data/observer.jsonl adapters/observer

# Train Responder
python train.py responder data/responder.jsonl adapters/responder

# Train Consultant
python train.py consultant data/consultant.jsonl adapters/consultant
```

### 2. Running Inference
To run the full multi-agent simulation on sample network events:

```powershell
python inference.py
```
This will load all three agents, process the events in the pipeline, and output the analysis to `analysis_results.json`.

---

## 📄 Example Output
A sample of the multi-agent processing flow:

```json
{
  "event": "High entropy DNS query xk9p2mzqw7h5vbnc4t8s.cdn.verify.ru",
  "observer": {
    "classification": "ANOMALOUS",
    "analysis": "Suspicious DNS tunneling pattern detected."
  },
  "responder": {
    "action": "Block domain at DNS firewall",
    "triage": "Isolate host pending investigation."
  },
  "consultant": {
    "strategy": "Review DNS logging policy. Hunt for C2 beacons across fleet."
  }
}
```

---
*University Project for Pre-Master Degree (Advanced Cryptocurrency & Security)*
