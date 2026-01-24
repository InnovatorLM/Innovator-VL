#!/usr/bin/env bash
set -e

# Innovator-VL RL Training Example
# Please configure the actual paths before use

echo "Innovator-VL RL Training Example"
echo "Config: RL/configs/innovator-vl-8b-gspo.yaml"
echo ""
echo "Please ensure you have configured:"
echo "1. Model path (actor.path)"
echo "2. Dataset path (dataset.path)"
echo "3. Experiment output path (cluster.fileroot)"
echo ""
echo "Example command:"
echo "python3 -m areal.launcher.local \\"
echo "  /path/to/Innovator-VL/RL/trains/grpo.py \\"
echo "  --config /path/to/Innovator-VL/RL/configs/innovator-vl-8b-gspo.yaml"
echo ""

# Uncomment and configure the correct paths before actual execution
# python3 -m areal.launcher.local \\
#   /path/to/Innovator-VL/RL/trains/grpo.py \\
#   --config /path/to/Innovator-VL/RL/configs/innovator-vl-8b-gspo.yaml
