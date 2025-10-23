#!/bin/bash

# Assign input file paths from environment variables
# These will be the S3 URIs passed from the AWS Batch job
INPUT_FILE_1=$1
INPUT_FILE_2=$2
OUTPUT_FILE=$3

# Log the inputs for debugging
echo "Starting concatenation job"
echo "Input 1: $INPUT_FILE_1"
echo "Input 2: $INPUT_FILE_2"
echo "Output: $OUTPUT_FILE"

# Create a temporary working directory
mkdir /tmp/working
cd /tmp/working

# Download the source files from S3 to the container's local storage
# This data transfer is free because it's within the same AWS region [cite: 35]
aws s3 cp $INPUT_FILE_1 part1.mp4
aws s3 cp $INPUT_FILE_2 part2.mp4

# --- Robust FFmpeg Concatenation using MPEG-TS Intermediate ---
# This three-step method is recommended for reliability with GoPro files [cite: 47, 53]

# 1. Convert each MP4 segment to a temporary MPEG-TS file
ffmpeg -i part1.mp4 -c copy -bsf:v h264_mp4toannexb -f mpegts part1.ts
ffmpeg -i part2.mp4 -c copy -bsf:v h264_mp4toannexb -f mpegts part2.ts

# 2. Concatenate the intermediate TS files into a single stream
ffmpeg -i "concat:part1.ts|part2.ts" -c copy -bsf:a aac_adtstoasc final_concat.mp4

# 3. Upload the final concatenated file back to S3
# Data transfer into S3 (ingress) is also free [cite: 22]
aws s3 cp final_concat.mp4 $OUTPUT_FILE

echo "Job finished successfully. Output uploaded to $OUTPUT_FILE"