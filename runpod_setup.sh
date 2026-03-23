#!/bin/bash
# runpod_setup.sh — Manual RunPod setup for SPPF audio training on LibriSpeech
#
# Usage on RunPod:
#   chmod +x runpod_setup.sh
#   ./runpod_setup.sh

set -e

WORKSPACE=/workspace/sppf_audio_demo
DATA_DIR=$WORKSPACE/data
OUTPUT_DIR=$WORKSPACE/outputs

# ─────────────────────────────────────────────
# 1. Install dependencies
# ─────────────────────────────────────────────
echo "==> Installing dependencies..."
pip install torchaudio tqdm pillow --quiet

# ─────────────────────────────────────────────
# 2. Clone repo
# ─────────────────────────────────────────────
echo "==> Cloning repo..."
if [ ! -d "$WORKSPACE/.git" ]; then
    git clone https://github.com/gcox-flomenta/sppf-audio-demo.git $WORKSPACE
fi
cd $WORKSPACE

# ─────────────────────────────────────────────
# 3. Train
#    LibriSpeech train-clean-100 (~6GB) is auto-downloaded by torchaudio.
# ─────────────────────────────────────────────
echo "==> Starting training..."
mkdir -p $DATA_DIR $OUTPUT_DIR

python train_audio_sppf.py \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --latent_dim 64 \
  --num_epochs 50 \
  --batch_size 64 \
  --lr 3e-4

echo ""
echo "==> Training complete."
echo "    Best model: $OUTPUT_DIR/ckpt_best.pt"
echo "    Latest:     $OUTPUT_DIR/ckpt_latest.pt"
echo ""
echo "Download ckpt_best.pt and drop it into C:\\code\\sppf_audio_demo\\"
