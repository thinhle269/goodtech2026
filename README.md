# TAFed-MSID

TAFed-MSID is a research codebase for federated intrusion detection in heterogeneous IoT/IIoT environments.

The project studies a multi-stage intrusion-detection setting in which learning is distributed across multiple clients. It includes conventional federated-learning baselines, robust aggregation methods, and a trust-aware aggregation approach. The experiments are designed to examine model behavior under non-IID data distributions and adversarial client updates.

The repository contains modules for data preparation, client partitioning, model definition, federated training, aggregation, experiment execution, evaluation, and result visualization. Configuration files are provided for adapting the experiments to a local environment.

The current implementation is intended for use with Edge-IIoTset and CICIoT2023. Dataset files are not included in this repository.

Users are expected to inspect the source code, prepare the required environment and datasets, configure the experiment settings, and reproduce the training and evaluation independently.
