"""
launch_pod.py — Called by GitHub Actions to spin up a RunPod training pod.
Reads all config from environment variables set by the workflow.

Audio model is ~591K params (vs 100M+ for video) so trains much faster.
RTX 4090 is overkill but keeps things simple; expect fast epoch times.
"""
import os, json, requests, sys, subprocess

api_key    = os.environ["RUNPOD_API_KEY"]
gh_token   = os.environ.get("GH_TOKEN", "")
if not gh_token:
    # Fall back to gh CLI auth token
    try:
        gh_token = subprocess.check_output(["gh", "auth", "token"], text=True).strip()
        print(f"  GH_TOKEN from gh CLI: {gh_token[:10]}...")
    except Exception:
        print("ERROR: No GH_TOKEN env var and 'gh auth token' failed. Set GH_TOKEN or run 'gh auth login'.")
        sys.exit(1)
gh_repo    = os.environ["GH_REPO"]
latent_dim = os.environ.get("LATENT_DIM", "64")
num_epochs = os.environ.get("NUM_EPOCHS", "50")
batch_size = os.environ.get("BATCH_SIZE", "64")
lr         = os.environ.get("LR", "3e-4")
gpu_pref   = os.environ.get("GPU_TYPE", "NVIDIA GeForce RTX 4090")
# S3 — optional; if not set, falls back to GitHub Releases
s3_key     = os.environ.get("AWS_ACCESS_KEY_ID", "")
s3_secret  = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
s3_bucket  = os.environ.get("AWS_S3_BUCKET", "")
s3_region  = os.environ.get("AWS_REGION", "us-east-1")

# Try preferred GPU first, then fall back to others with SECURE availability
GPU_FALLBACK_ORDER = [
    gpu_pref,
    "NVIDIA GeForce RTX 4090",
    "NVIDIA L4",
    "NVIDIA A10",
    "NVIDIA RTX A5000",
    "A40",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
]
# Deduplicate while preserving order
seen = set()
GPU_FALLBACK_ORDER = [g for g in GPU_FALLBACK_ORDER if not (g in seen or seen.add(g))]

# Inline credentials directly in dockerArgs.
# GH_TOKEN is substituted here by Python before the request is sent.
repo_url = f"https://x-access-token:{gh_token}@github.com/{gh_repo}.git"
# Only wipe the repo (get latest code) — data/outputs persist on the network volume
docker_cmd = (
    f"bash -c '"
    f"rm -rf /workspace/repo && "
    f"git clone {repo_url} /workspace/repo && "
    f"cd /workspace/repo && "
    f"bash scripts/runpod_train.sh "
    f"2>&1 | tee /workspace/training.log; "
    f"sleep 300"
    f"'"
)

headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

QUERY = """
mutation($input: PodFindAndDeployOnDemandInput!) {
  podFindAndDeployOnDemand(input: $input) {
    id
    machineId
    costPerHr
  }
}
"""

pod = None
used_gpu = None
for gpu_type in GPU_FALLBACK_ORDER:
    variables = {
        "input": {
            "name": "sppf-audio-training",
            "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
            "gpuTypeId": gpu_type,
            "cloudType": "SECURE",  # NEVER use COMMUNITY
            "gpuCount": 1,
            "containerDiskInGb": 50,  # Audio needs less disk than video
            "startSsh": True,
            "startJupyter": False,
            "dockerArgs": docker_cmd,
            "env": [
                {"key": "GH_TOKEN",                "value": gh_token},
                {"key": "GH_REPO",                 "value": gh_repo},
                {"key": "LATENT_DIM",              "value": latent_dim},
                {"key": "NUM_EPOCHS",              "value": num_epochs},
                {"key": "BATCH_SIZE",              "value": batch_size},
                {"key": "LR",                      "value": lr},
                {"key": "RUNPOD_API_KEY",          "value": api_key},
                {"key": "AWS_ACCESS_KEY_ID",       "value": s3_key},
                {"key": "AWS_SECRET_ACCESS_KEY",   "value": s3_secret},
                {"key": "AWS_S3_BUCKET",           "value": s3_bucket},
                {"key": "AWS_REGION",              "value": s3_region},
            ],
        }
    }
    resp = requests.post(
        "https://api.runpod.io/graphql",
        headers=headers,
        json={"query": QUERY, "variables": variables},
    )
    data = resp.json()

    if "errors" in data:
        msg = data["errors"][0].get("message", "")
        print(f"  {gpu_type}: unavailable ({msg[:60]})")
        continue

    pod = data["data"]["podFindAndDeployOnDemand"]
    used_gpu = gpu_type
    break

if not pod:
    print("No GPUs available. Tried:", GPU_FALLBACK_ORDER)
    sys.exit(1)

print(f"Pod launched!")
print(f"  Pod ID:  {pod['id']}")
print(f"  GPU:     {used_gpu}")
print(f"  Cost:    ${pod['costPerHr']:.3f}/hr")
print(f"  Config:  latent_dim={latent_dim} epochs={num_epochs} batch={batch_size} lr={lr}")
print(f"  Results: https://github.com/{gh_repo}/releases")
