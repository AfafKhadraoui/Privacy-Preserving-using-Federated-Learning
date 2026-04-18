# ENSIA P007 - Privacy-Preserving Face Recognition using Federated Learning

This repository implements a full biometric recognition pipeline where raw face photos stay on-device and model training is performed with Federated Learning (FL). Differential Privacy (DP) is added to reduce leakage from model updates, and attacks are run to demonstrate the privacy gain.

## Project Identity

- School: Ecole Nationale Superieure d'Intelligence Artificielle (ENSIA)
- Program: AI and Computer Science Programme
- Academic year: 2025-2026
- Project code: P007
- Main objective: Build and demonstrate privacy-preserving face recognition with measurable security evidence

## Core Privacy Claim

Raw face photos never leave the client device. During FL, the server receives only model weight updates, not images, not names, and not embeddings. After training, recognition runs on-device.

## Team Task Assignment

| Member | Responsibility |
|---|---|
| Abderrahim | Step 1 - Face Detection and Preprocessing |
| To be decided | Step 2 - Data Augmentation |
| Kosai | Step 3 - Feature Extraction and Model |
| Afaf | Step 4 - Federated Learning |
| Amel | Step 5 - Privacy and Security |
| Mehdi | Step 6 - Flutter Demo Application |
| To be decided | Step 7 - Attack Implementation and Showcase |

## Full Pipeline Overview

1. Step 1 - Face Detection and Preprocessing
- Input: Raw photo
- Output: aligned and normalized 160x160 face tensor
- Tooling: MTCNN from facenet-pytorch

2. Step 2 - Data Augmentation
- Input: one aligned 160x160 face crop
- Output: 20 augmented variants for training
- Note: augmentation is training-only, never applied during inference

3. Step 3 - Feature Extraction
- Input: normalized face tensor
- Output: 512-dimensional L2-normalized embedding
- Model: InceptionResNetV1

4. Step 4 - Federated Learning
- Clients train locally and send weight updates
- Server aggregates with FedAvg
- No raw biometric data is transmitted

5. Step 5 - Privacy and Security
- Differential Privacy (Opacus): clipping + Gaussian noise + epsilon accounting
- Secure Aggregation (Flower SecAgg): server sees aggregate only

6. Step 6 - Flutter Demo App
- Enrollment, Recognition, FL Dashboard, Security Demo screens

7. Step 7 - Attack Showcase
- Model inversion on no-DP model vs DP model
- Membership inference comparison (member vs non-member distance gap)

## Two-Model Strategy (Mandatory)

The experiment must always keep two FL models:

- Version A (No DP): higher accuracy baseline, vulnerable to inversion
- Version B (With DP): lower leakage, expected privacy-accuracy tradeoff

Only one variable changes between A and B: DP activation (USE_DP flag). Architecture, data, rounds, and training process remain otherwise identical.

## What You Should Do Next (Practical Execution Plan)

1. Environment and dependencies
- Create and activate Python environment
- Install PyTorch first, then remaining dependencies
- Verify imports: torch, flwr, opacus, facenet_pytorch

2. Data preparation
- Place 1-3 photos per person in data/raw/<person_name>/
- Run preprocessing to produce data/cropped/

3. Federated setup
- Partition cropped data into data/clients/client_XX/
- Confirm each client has data before training

4. Baseline and FL training
- Train centralized baseline model
- Train FL model without DP (Version A)
- Train FL model with DP (Version B)

5. Security evaluation
- Run model inversion on both versions
- Run membership inference on both versions
- Save metrics and generated plots in results/

6. Presentation outputs
- Build attack comparison visual (original vs no-DP reconstruction vs DP output)
- Build privacy-accuracy tradeoff chart (epsilon vs accuracy)

7. Flutter integration
- Export final model(s) to TFLite
- Integrate in app/assets/
- Demonstrate Enrollment, Recognition, Training Dashboard, and Security Demo screens

## Execution Order

Run in this order:

1. src/preprocessing/prepare_dataset.py
2. src/federated/partition.py
3. experiments/train_centralized.py
4. experiments/train_fl_no_dp.py
5. experiments/train_fl_with_dp.py
6. experiments/run_attacks.py
7. experiments/plot_results.py
8. src/model/export_tflite.py
9. app/flutter run

## Deliverables Checklist

- Models:
	- results/models/model_centralized.pth
	- results/models/model_fl_no_dp.pth
	- results/models/model_fl_with_dp.pth
- Metrics:
	- results/metrics/centralized_results.json
	- results/metrics/fl_no_dp_results.json
	- results/metrics/fl_with_dp_results.json
	- results/metrics/membership_inference.json
- Plots:
	- results/plots/accuracy_per_round.png
	- results/plots/privacy_accuracy_tradeoff.png
	- results/plots/attack_no_dp.png
	- results/plots/attack_with_dp.png
	- results/plots/attack_comparison.png

## Technical References

- Detailed and complete specification: PROJECT_SPEC.md
- This README: quick project map, team responsibilities, and execution guide

