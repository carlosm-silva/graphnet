#!/bin/bash

# Script to submit all alpha experiments in parallel
# This will submit jobs for alpha = 0.026, 0.040, and 0.060

echo "Submitting alpha experiments in parallel..."

# Submit job for alpha = 0.026
echo "Submitting job for alpha = 0.026..."
JOB_026=$(sbatch l40s_mixed_alpha_0_026.sbatch | awk '{print $4}')
echo "Job submitted with ID: $JOB_026"

# Submit job for alpha = 0.040
echo "Submitting job for alpha = 0.040..."
JOB_040=$(sbatch l40s_mixed_alpha_0_040.sbatch | awk '{print $4}')
echo "Job submitted with ID: $JOB_040"

# Submit job for alpha = 0.060
echo "Submitting job for alpha = 0.060..."
JOB_060=$(sbatch l40s_mixed_alpha_0_060.sbatch | awk '{print $4}')
echo "Job submitted with ID: $JOB_060"

echo ""
echo "All jobs submitted successfully!"
echo "Job IDs:"
echo "  Alpha 0.026: $JOB_026"
echo "  Alpha 0.040: $JOB_040"
echo "  Alpha 0.060: $JOB_060"
echo ""
echo "You can monitor the jobs using:"
echo "  squeue -u $USER"
echo ""
echo "Check job status with:"
echo "  scontrol show job $JOB_026"
echo "  scontrol show job $JOB_040"
echo "  scontrol show job $JOB_060" 