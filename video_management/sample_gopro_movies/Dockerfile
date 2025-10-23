# Use a base image that includes FFmpeg and AWS CLI
FROM jrottenberg/ffmpeg:4.1-alpine

# Install Python and the AWS CLI
RUN apk --no-cache add python3 py3-pip && \
    pip3 install --upgrade pip && \
    pip3 install awscli

# Copy the concatenation script into the container
COPY concat.sh /usr/local/bin/concat.sh

# Make the script executable
RUN chmod +x /usr/local/bin/concat.sh

# Set the script as the entry point
ENTRYPOINT ["/usr/local/bin/concat.sh"]