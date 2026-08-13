# Speculative Engine

This module handles the logic and benchmarking for the Multi-Token Prediction (MTP) speculative decoding loop. 

## Key Responsibilities
- **Token Acceptance ($\tau$)**: Measuring how often the main backbone model accepts the tokens proposed by the draft head.
- **Draft Mechanics**: Ensuring that draft heads behave correctly and align their probability distributions with the backbone model.
