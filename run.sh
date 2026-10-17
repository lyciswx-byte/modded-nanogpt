# Record #92 adapted for one NVIDIA H100 (see SINGLE_H100.md).
torchrun --standalone --nproc_per_node=1 train_gpt.py
