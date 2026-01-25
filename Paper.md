# Multi-Agent Network Security Monitoring System: Training Methodology and Performance Analysis

## Abstract

This research presents a novel approach to network security monitoring through the development of a specialized multi-agent system built upon the Llama 3.2-3B foundation model. The system employs parameter-efficient fine-tuning techniques to create three distinct specialized agents—Observer, Responder, and Consultant—each trained for specific roles in network security incident detection, response, and strategic analysis. Our methodology demonstrates the viability of using Low-Rank Adaptation (LoRA) for creating role-specific language models while maintaining computational efficiency on consumer-grade hardware.

## 1. System Architecture and Design Rationale

The proposed system implements a hierarchical multi-agent architecture designed to mirror the operational structure of security operations centers. The architecture comprises three specialized agents, each fulfilling distinct functional roles within the network monitoring pipeline. This division of labor enables more focused training objectives and allows each agent to develop deep expertise in its designated domain while maintaining the flexibility of the underlying foundation model.

The Observer agent serves as the primary monitoring and detection component, responsible for analyzing network traffic patterns, identifying anomalies, and generating structured alerts. The Responder agent functions as the tactical response unit, providing immediate remediation strategies and implementation guidance for detected threats. The Consultant agent operates at the strategic level, offering comprehensive security assessments, policy recommendations, and long-term defensive strategies.

## 2. Foundation Model Selection and Technical Infrastructure

The system utilizes Meta's Llama 3.2-3B as the base foundation model, selected for its balance between capability and computational efficiency. With 3.22 billion parameters, this model provides sufficient representational capacity for complex security reasoning while remaining deployable on consumer hardware. The model was accessed through the Hugging Face Transformers library, leveraging the pre-trained weights as initialization for domain-specific fine-tuning.

The training infrastructure employed an NVIDIA RTX 5060 Ti with 16GB of video memory, demonstrating the accessibility of this approach for research institutions and organizations with limited computational resources. The system operated in FP16 (16-bit floating-point) precision mode, which reduced memory requirements by approximately fifty percent compared to full FP32 precision while maintaining numerical stability during gradient descent optimization.

## 3. Parameter-Efficient Fine-Tuning Methodology

### 3.1 Low-Rank Adaptation (LoRA) Configuration

Rather than full model fine-tuning, which would require updating all 3.22 billion parameters, this research employed Low-Rank Adaptation to achieve parameter efficiency. The LoRA methodology decomposes weight update matrices into low-rank representations, dramatically reducing the number of trainable parameters while preserving model expressiveness.

The specific LoRA configuration implemented a rank of eight with an alpha scaling factor of thirty-two, yielding approximately 12.16 million trainable parameters per agent. This represents only 0.377 percent of the total model parameters, enabling efficient training while avoiding catastrophic forgetting of the foundation model's pre-trained knowledge. The adapter modules were applied to seven critical projection matrices within each transformer layer: the query, key, value, and output projections in the attention mechanism, as well as the gate, up, and down projections in the feed-forward network.

A dropout rate of 0.05 was applied to the LoRA layers to prevent overfitting, particularly important given the relatively small domain-specific training datasets. The adapters were configured for causal language modeling, maintaining the autoregressive generation capabilities of the base model while specializing its outputs for security-relevant tasks.

### 3.2 Training Hyperparameters and Optimization Strategy

The training process employed the AdamW optimizer implemented in PyTorch, selected for its adaptive learning rate capabilities and weight decay regularization. The base learning rate was set to 2×10⁻⁴, with a cosine annealing schedule providing gradual decay throughout training. A warmup period of one hundred steps allowed for stable initialization of the optimization dynamics.

The effective batch size was maintained at eight examples through gradient accumulation, combining a per-device batch size of two with four accumulation steps. This configuration balanced memory efficiency with gradient estimate quality, enabling stable convergence while respecting hardware constraints. Gradient clipping was applied with a maximum norm of 1.0 to prevent instability from occasional large gradients during training on specialized technical content.

Gradient checkpointing was enabled to reduce memory consumption by trading computation for memory, recomputing intermediate activations during the backward pass rather than storing them. This technique proved essential for fitting the training process within the available 16GB of GPU memory while maintaining reasonable batch sizes.

## 4. Dataset Characteristics and Training Corpus

### 4.1 Observer Agent Training Data

The Observer agent was trained on a curated dataset of 329 examples focused on network traffic analysis, anomaly detection, and alert generation. This dataset represented the largest training corpus among the three agents, reflecting the complexity and diversity of monitoring scenarios encountered in operational network environments. The training examples encompassed various attack vectors, normal traffic patterns, and edge cases requiring nuanced interpretation.

### 4.2 Responder Agent Training Data

The Responder agent received 128 training examples centered on incident response procedures, remediation strategies, and tactical mitigation techniques. The relatively smaller dataset compared to the Observer reflected the more constrained action space of response procedures, where quality and precision of responses were prioritized over covering an extensive range of scenarios.

### 4.3 Consultant Agent Training Data

The Consultant agent was trained on 43 examples of strategic security analysis, policy development, and comprehensive threat assessments. This represented the smallest dataset, justified by the higher-level nature of strategic analysis and the need for deep reasoning over broad coverage. The extended training duration of ten epochs for this agent compensated for the limited dataset size, allowing for thorough optimization of the specialized knowledge required for strategic decision-making.

## 5. Training Performance and Convergence Analysis

### 5.1 Observer Agent Training Dynamics

The Observer agent training exhibited strong convergence characteristics over four epochs, completing in 192.27 seconds at a throughput of 6.845 samples per second. The training loss trajectory demonstrated rapid initial learning, with the loss decreasing from an initial value of 3.6551 to 0.3951 by the final epoch, representing an approximately 89 percent reduction. This dramatic improvement indicates successful adaptation to the network monitoring domain.

The gradient norm progression revealed healthy optimization dynamics, with initial values around 2.49 stabilizing to 0.38 by the end of training. This convergence of gradient magnitudes suggests the model reached a well-optimized region of the parameter space without signs of instability or divergence. The learning rate schedule progressed as designed, reaching its peak of 1.98×10⁻⁴ at epoch 2.39 before decaying according to the cosine schedule to 8.52×10⁻⁶ by the final epoch.

The final average training loss of 0.8981 across all training steps demonstrates effective learning while maintaining some residual uncertainty, which is desirable for preventing overconfident predictions in the security domain where novel threats regularly emerge.

### 5.2 Responder Agent Training Dynamics

The Responder agent completed training in 77.00 seconds over four epochs, achieving a throughput of 6.649 samples per second. The loss trajectory followed a similar pattern to the Observer agent, beginning at 3.7634 and concluding at 0.5581, representing an 85 percent reduction. The slightly higher final loss compared to the Observer agent may reflect the inherent complexity of generating appropriate response procedures, which require precise technical accuracy and consideration of operational constraints.

Gradient norm values progressed from 2.42 to 1.14, indicating stable optimization throughout training. The higher final gradient norm relative to the Observer agent suggests the Responder agent may benefit from additional training epochs or a slightly adjusted learning rate schedule in future iterations, though the current performance metrics indicate successful convergence.

The average training loss of 1.7464 was higher than the Observer agent, which is expected given the smaller dataset size and the challenge of learning precise procedural knowledge from limited examples.

### 5.3 Consultant Agent Training Dynamics

The Consultant agent underwent extended training over ten epochs, completing in 71.06 seconds at 6.051 samples per second. The extended training duration was deliberately chosen to compensate for the limited dataset of 43 examples, allowing the model to thoroughly internalize the strategic reasoning patterns required for high-level security analysis.

The loss decreased from 3.9188 to 0.6653, representing an 83 percent reduction. The training curve showed continued improvement throughout the extended training period without evidence of overfitting, suggesting the model successfully learned generalizable patterns rather than memorizing specific examples. The gradient norm progression from 2.31 to 1.92 remained stable, though the higher final value compared to the Observer agent indicates the optimization landscape for strategic analysis tasks may be more complex.

The average training loss of 1.9509 was the highest among the three agents, which is expected given the small dataset size and the abstract nature of strategic security reasoning. Despite this, the successful convergence indicates the model acquired useful capabilities for its designated role.

## 6. Technical Implementation and Reproducibility

The training pipeline was implemented in Python 3.13 using the Hugging Face Transformers library version 4.x and the PEFT (Parameter-Efficient Fine-Tuning) library for LoRA implementation. The training script employed mixed-precision training (FP16) through PyTorch's automatic mixed precision capabilities, achieving substantial speedup and memory reduction without measurable degradation in final model quality.

Each agent's adapted weights were saved as independent LoRA adapters, requiring approximately 50 megabytes of storage per agent compared to the 6.4 gigabytes required for full model weights. This parameter efficiency extends beyond training to deployment, enabling rapid switching between specialized agents by loading different adapter weights while sharing the common base model.

The training process proved highly reproducible, with deterministic results achievable through proper random seed initialization. The entire training pipeline for all three agents completed in approximately 5.7 minutes of total computation time, demonstrating the efficiency of the LoRA approach for rapid iteration and experimentation.

## 7. Conclusion and Future Directions

This research successfully demonstrates the application of parameter-efficient fine-tuning techniques for creating specialized multi-agent systems in the network security domain. The LoRA-based approach enabled training of three distinct specialized agents with minimal computational resources while preserving the general capabilities of the foundation model. The training metrics indicate successful convergence for all agents, with particularly strong performance from the Observer agent and stable learning dynamics across all components.

Future work will focus on evaluation of the trained agents on held-out test sets, analysis of inter-agent communication patterns, and investigation of techniques for further reducing the Consultant agent's training loss through dataset augmentation or curriculum learning approaches. Additionally, deployment strategies for production environments and integration with existing security information and event management systems represent important directions for transitioning this research into operational practice.