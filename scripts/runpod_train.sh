#!/bin/bash
# scripts/runpod_train.sh
# Automated training script for RunPod pods — SPPF Audio Autoencoder.
# All credentials come from environment variables — nothing hardcoded.
#
# Required env vars (set via GitHub Actions / RunPod API):
#   GH_TOKEN          GitHub PAT (for uploading checkpoints to Releases)
#   GH_REPO           e.g. gcox-flomenta/sppf-audio-demo
#   LATENT_DIM        default: 64
#   NUM_EPOCHS        default: 50
#   BATCH_SIZE        default: 64
#   LR                default: 3e-4
#
# Optional S3 env vars (if set, S3 is used for checkpoints):
#   AWS_ACCESS_KEY_ID
#   AWS_SECRET_ACCESS_KEY
#   AWS_S3_BUCKET     e.g. my-sppf-training
#   AWS_REGION        default: us-east-1

set -e
trap 'log "ERROR at line $LINENO: $BASH_COMMAND"' ERR

WORKSPACE=/workspace
REPO_DIR=$WORKSPACE/repo
DATA_DIR=$WORKSPACE/data
OUTPUT_DIR=$WORKSPACE/outputs

LATENT_DIM=${LATENT_DIM:-64}
NUM_EPOCHS=${NUM_EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-64}
LR=${LR:-3e-4}

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== runpod_train.sh starting (SPPF Audio) ==="
log "REPO_DIR=$REPO_DIR"
log "DATA_DIR=$DATA_DIR"
log "OUTPUT_DIR=$OUTPUT_DIR"
log "LATENT_DIM=$LATENT_DIM NUM_EPOCHS=$NUM_EPOCHS BATCH_SIZE=$BATCH_SIZE LR=$LR"
log "Disk usage: $(df -h /workspace | tail -1)"

# ─────────────────────────────────────────────
# 1. Install dependencies
# ─────────────────────────────────────────────
log "--- Step 1: Installing dependencies ---"
pip install -q torchaudio tqdm pillow awscli soundfile  # torch already in RunPod image — never reinstall
apt-get update -q && apt-get install -y libsndfile1 --quiet  # torchaudio needs libsndfile for .flac (LibriSpeech)
log "pip done"
apt-get update -q
apt-get install -y curl --quiet
log "apt done"

# Configure AWS if credentials are present
USE_S3=false
if [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$AWS_S3_BUCKET" ]; then
    aws configure set aws_access_key_id "$AWS_ACCESS_KEY_ID"
    aws configure set aws_secret_access_key "$AWS_SECRET_ACCESS_KEY"
    aws configure set default.region "${AWS_REGION:-us-east-1}"
    aws configure set default.output json
    aws s3 ls "s3://${AWS_S3_BUCKET}/" > /dev/null 2>&1 && USE_S3=true || log "Warning: S3 bucket not accessible — falling back to GitHub Releases"
fi
log "USE_S3=$USE_S3  BUCKET=${AWS_S3_BUCKET:-none}"

# Install GitHub CLI
if ! command -v gh &>/dev/null; then
    log "Installing GitHub CLI..."
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] \
        https://cli.github.com/packages stable main" \
        | tee /etc/apt/sources.list.d/github-cli.list > /dev/null
    apt-get update -q && apt-get install -y gh --quiet
fi
log "gh version: $(gh --version | head -1)"

# ─────────────────────────────────────────────
# 2. Enter repo (already cloned by bootstrap)
# ─────────────────────────────────────────────
log "--- Step 2: Entering repo ---"
log "Contents of /workspace: $(ls /workspace)"
cd $REPO_DIR
log "PWD: $(pwd)"

# ─────────────────────────────────────────────
# 3. Dataset — LibriSpeech (auto-downloaded by torchaudio)
# ─────────────────────────────────────────────
log "--- Step 3: Dataset ---"
log "LibriSpeech will be auto-downloaded by torchaudio to $DATA_DIR (~6GB)"
mkdir -p $DATA_DIR

# ─────────────────────────────────────────────
# 4. Create GitHub release upfront
# ─────────────────────────────────────────────
log "--- Step 4: GitHub release ---"
echo "$GH_TOKEN" | gh auth login --with-token 2>/dev/null || true
log "gh auth status: $(gh auth status 2>&1 | head -2 || echo 'auth check failed')"

RELEASE_TAG="training-$(date '+%Y%m%d-%H%M')-dim${LATENT_DIM}"
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown-gpu")
log "GPU: $GPU_NAME"
log "Release tag: $RELEASE_TAG"

gh release create "$RELEASE_TAG" \
    --repo "$GH_REPO" \
    --title "Training in progress — $RELEASE_TAG" \
    --notes "GPU: $GPU_NAME | latent_dim=$LATENT_DIM | epochs=$NUM_EPOCHS | batch=$BATCH_SIZE | Status: RUNNING" \
    && log "Release created OK" \
    || log "Warning: could not create GitHub release (will retry at end)"

# ─────────────────────────────────────────────
# 5. Resume checkpoint — download if available
# ─────────────────────────────────────────────
log "--- Step 5: Check for resume checkpoint ---"
mkdir -p $OUTPUT_DIR

# The training script resumes from ckpt_latest.pt in the output dir.
# We store resume.pt externally (S3 / GitHub Releases) and copy it to ckpt_latest.pt.
if [ ! -f "$OUTPUT_DIR/ckpt_latest.pt" ]; then
    RESUME_DOWNLOADED=false

    # Try S3 first
    if [ "$USE_S3" = "true" ]; then
        log "Checking S3 for resume.pt..."
        aws s3 cp "s3://${AWS_S3_BUCKET}/checkpoints/resume.pt" "$OUTPUT_DIR/resume.pt" 2>/dev/null \
            && RESUME_DOWNLOADED=true \
            || log "No resume.pt in S3"
    fi

    # Try GitHub Releases if S3 didn't have it
    if [ "$RESUME_DOWNLOADED" = "false" ]; then
        log "Checking 'checkpoints' release for resume.pt..."
        gh release download "checkpoints" --repo "$GH_REPO" --pattern "resume.pt" --dir "$OUTPUT_DIR" 2>/dev/null \
            && RESUME_DOWNLOADED=true \
            || log "No resume.pt in 'checkpoints' release"
    fi

    if [ "$RESUME_DOWNLOADED" = "true" ] && [ -f "$OUTPUT_DIR/resume.pt" ]; then
        RESUME_EPOCH=$(python3 -c "import torch; c=torch.load('$OUTPUT_DIR/resume.pt', map_location='cpu', weights_only=False); print(c.get('epoch',0))" 2>/dev/null || echo "0")
        if [ "$RESUME_EPOCH" -ge "$NUM_EPOCHS" ]; then
            log "resume.pt is at epoch $RESUME_EPOCH which >= num_epochs $NUM_EPOCHS — deleting and starting fresh"
            rm "$OUTPUT_DIR/resume.pt"
        else
            log "Found resume.pt at epoch $RESUME_EPOCH — copying to ckpt_latest.pt for auto-resume"
            cp "$OUTPUT_DIR/resume.pt" "$OUTPUT_DIR/ckpt_latest.pt"
        fi
    else
        log "No resume checkpoint found — starting fresh"
    fi
else
    log "ckpt_latest.pt already exists in output dir — will auto-resume"
fi

# ─────────────────────────────────────────────
# 6. Train in background + upload checkpoints every 20 min
# ─────────────────────────────────────────────
log "--- Step 6: Starting training ---"
log "Training script: $(ls $REPO_DIR/train_audio_sppf.py 2>/dev/null && echo 'found' || echo 'NOT FOUND')"

python train_audio_sppf.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --latent_dim $LATENT_DIM \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR &
TRAIN_PID=$!
log "Training PID: $TRAIN_PID"

# Upload ckpt_latest.pt every 20 minutes while training runs
LAST_UPLOAD_TIME=0
while kill -0 $TRAIN_PID 2>/dev/null; do
    sleep 1200  # 20 minutes

    if [ -f "$OUTPUT_DIR/ckpt_latest.pt" ]; then
        CURRENT_EPOCH=$(python3 -c "import torch; c=torch.load('$OUTPUT_DIR/ckpt_latest.pt', map_location='cpu', weights_only=False); print(c.get('epoch',0))" 2>/dev/null || echo "?")
        log "Mid-training upload: epoch $CURRENT_EPOCH"

        # Copy ckpt_latest.pt as resume.pt for external storage
        cp "$OUTPUT_DIR/ckpt_latest.pt" "$OUTPUT_DIR/resume.pt"

        # Upload to GitHub Releases
        gh release upload "$RELEASE_TAG" "$OUTPUT_DIR/ckpt_latest.pt" "$OUTPUT_DIR/resume.pt" \
            --repo "$GH_REPO" --clobber \
            && log "Mid-training GH upload succeeded" \
            || log "Upload failed — will retry next cycle"

        # Push resume.pt to S3
        if [ "$USE_S3" = "true" ]; then
            aws s3 cp "$OUTPUT_DIR/resume.pt" "s3://${AWS_S3_BUCKET}/checkpoints/resume.pt" \
                && log "resume.pt synced to S3" \
                || log "Warning: failed to sync resume.pt to S3"
            aws s3 cp "$OUTPUT_DIR/ckpt_latest.pt" "s3://${AWS_S3_BUCKET}/checkpoints/ckpt_latest.pt" \
                && log "ckpt_latest.pt synced to S3" || true
        else
            # Fallback: keep GitHub Releases checkpoints release in sync
            gh release view "checkpoints" --repo "$GH_REPO" >/dev/null 2>&1 \
                || gh release create "checkpoints" --repo "$GH_REPO" \
                    --title "Latest resume checkpoint" \
                    --notes "Always contains the most recent resume.pt — overwritten each upload" \
                    --prerelease || true
            gh release upload "checkpoints" "$OUTPUT_DIR/resume.pt" --repo "$GH_REPO" --clobber 2>/dev/null \
                && log "resume.pt synced to 'checkpoints' release" \
                || log "Warning: failed to sync resume.pt to 'checkpoints'"
        fi
    fi
done

wait $TRAIN_PID
TRAIN_EXIT=$?
log "Training process exited with code $TRAIN_EXIT"

# ─────────────────────────────────────────────
# 7. Upload final checkpoint
# ─────────────────────────────────────────────
log "--- Step 7: Uploading final assets ---"
log "Output dir contents: $(ls $OUTPUT_DIR 2>/dev/null || echo 'empty')"

# Create resume.pt from final ckpt_latest.pt
if [ -f "$OUTPUT_DIR/ckpt_latest.pt" ]; then
    cp "$OUTPUT_DIR/ckpt_latest.pt" "$OUTPUT_DIR/resume.pt"
fi

# Ensure release exists
gh release view "$RELEASE_TAG" --repo "$GH_REPO" >/dev/null 2>&1 \
    || gh release create "$RELEASE_TAG" \
        --repo "$GH_REPO" \
        --title "Complete — $RELEASE_TAG" \
        --notes "GPU: $GPU_NAME | latent_dim=$LATENT_DIM | epochs=$NUM_EPOCHS | batch=$BATCH_SIZE | Status: COMPLETE" \
    || true

# Upload best + latest checkpoints
UPLOAD_FILES=""
[ -f "$OUTPUT_DIR/ckpt_best.pt" ]   && UPLOAD_FILES="$UPLOAD_FILES $OUTPUT_DIR/ckpt_best.pt"
[ -f "$OUTPUT_DIR/ckpt_latest.pt" ] && UPLOAD_FILES="$UPLOAD_FILES $OUTPUT_DIR/ckpt_latest.pt"
[ -f "$OUTPUT_DIR/resume.pt" ]      && UPLOAD_FILES="$UPLOAD_FILES $OUTPUT_DIR/resume.pt"

if [ -n "$UPLOAD_FILES" ]; then
    gh release upload "$RELEASE_TAG" $UPLOAD_FILES \
        --repo "$GH_REPO" --clobber \
        && log "Final GH upload succeeded" \
        || log "Warning: final GH upload failed"
fi

# Final S3 sync
if [ "$USE_S3" = "true" ]; then
    log "Syncing final checkpoints to S3..."
    [ -f "$OUTPUT_DIR/ckpt_best.pt" ]   && aws s3 cp "$OUTPUT_DIR/ckpt_best.pt"   "s3://${AWS_S3_BUCKET}/checkpoints/ckpt_best.pt"   || true
    [ -f "$OUTPUT_DIR/ckpt_latest.pt" ] && aws s3 cp "$OUTPUT_DIR/ckpt_latest.pt" "s3://${AWS_S3_BUCKET}/checkpoints/ckpt_latest.pt" || true
    [ -f "$OUTPUT_DIR/resume.pt" ]      && aws s3 cp "$OUTPUT_DIR/resume.pt"      "s3://${AWS_S3_BUCKET}/checkpoints/resume.pt"      || true
    log "S3 sync complete: s3://${AWS_S3_BUCKET}/checkpoints/"
fi

gh release edit "$RELEASE_TAG" \
    --repo "$GH_REPO" \
    --title "Complete — $RELEASE_TAG" \
    --notes "GPU: $GPU_NAME | latent_dim=$LATENT_DIM | epochs=$NUM_EPOCHS | batch=$BATCH_SIZE | Status: COMPLETE" \
    || true

log "Done: https://github.com/$GH_REPO/releases/tag/$RELEASE_TAG"

# ─────────────────────────────────────────────
# 8. Self-terminate pod (stop billing)
# ─────────────────────────────────────────────
log "--- Step 8: Terminating pod ---"
curl -s "https://api.runpod.io/graphql" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -d "{\"query\": \"mutation { podTerminate(input: {podId: \\\"${RUNPOD_POD_ID}\\\"}) }\"}"

log "All done."
