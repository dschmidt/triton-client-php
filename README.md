# Triton PHP Client

A PHP client for streaming inference with NVIDIA Triton Inference Server using gRPC.

## Prerequisites

- PHP 8.0 or higher
- gRPC PHP extension (`grpc.so`)
- Composer

## Installation

1. Install dependencies:
```bash
composer install
```

2. Generate protobuf classes:
```bash
composer run-script generate-proto
```

3. Ensure gRPC extension is loaded:
```bash
php -m | grep grpc
```

## Usage

### Basic Usage

```bash
# Use default prompt
php client.php

# Custom prompt
php client.php "Hello, how are you?"

# Custom prompt with max tokens
php client.php "Write a story" 256
```

### Environment Variables

- `TRITON_URL` - Triton server URL (default: `localhost:8001`)
- `MODEL_NAME` - Model name (default: `mistral-streaming`)

### Examples

```bash
# Connect to remote server
TRITON_URL=triton-server.example.com:8001 php client.php "Your prompt"

# Use different model
MODEL_NAME=llama-chat php client.php "What is AI?"

# Custom server and model
TRITON_URL=192.168.1.100:8001 MODEL_NAME=gpt-model php client.php "Explain quantum physics" 512
```

## Project Structure

```
├── client.php              # Main client application
├── composer.json           # Dependencies and scripts
├── composer.lock           # Lock file
├── .gitignore              # Git ignore rules
├── README.md               # This file
├── grpc_service.proto      # Triton gRPC service definitions
├── model_config.proto      # Triton model configuration definitions
├── vendor/                 # Composer dependencies (ignored)
└── generated/              # Generated protobuf classes (ignored)
    ├── Inference/          # Inference service classes
    └── GPBMetadata/        # Protobuf metadata classes
```

## Development

### Updating Proto Files

To update to the latest Triton proto definitions:

```bash
composer run-script update-protos
composer run-script generate-proto
```

### Regenerating Protobuf Classes

If you modify the .proto files, regenerate the PHP classes:

```bash
composer run-script generate-proto
```

### Proto File Management

- **Proto files are checked into git** - They represent the API contract
- **Generated classes are ignored** - They're regenerated automatically
- **Updates are automated** - Use `update-protos` script to get latest from Triton
- **Build process** - Classes are generated automatically on `composer install`

## Error Handling

The client handles common gRPC errors gracefully:

- Connection failures
- Server errors in response stream
- Malformed responses
- Timeout issues

Error messages are prefixed with `[ERROR]` for easy identification.

## Requirements

- A running Triton Inference Server with a streaming model
- Model must support the expected input/output format:
  - Input: `text_input` (BYTES), `max_tokens` (INT32)
  - Output: `text_output` (BYTES), `is_final` (BOOL)

## License

This project is part of the IABG deployment infrastructure.# triton-client-php
