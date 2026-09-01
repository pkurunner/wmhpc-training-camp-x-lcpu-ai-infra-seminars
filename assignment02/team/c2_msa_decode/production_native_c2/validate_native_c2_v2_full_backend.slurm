#!/usr/bin/env bash
#SBATCH --job-name=c2-native-v2-full
#SBATCH --partition=gpu
#SBATCH --account=default_acct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/lcpu/85117379/c2-native-v2-full-backend-artifacts-20260829/slurm-%j.log

set -Eeuo pipefail
: "${C2_EXPECTED_LIB_SHA:?submit with the reviewed v2 AOT SHA-256}"
export C2_EXPECTED_SOURCE_SHA=6379cb0c71aa79610dd807b40982a6653637d926c1e3dab78155f1f3a5a4f6ce
export C2_FULL_ARTIFACT_DIR=/home/lcpu/85117379/c2-native-v2-full-backend-artifacts-20260829
exec bash /home/lcpu/85117379/validate_native_c2_full_backend_20260829.sh
