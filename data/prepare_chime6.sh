#!/bin/bash

# CHiME-6 Dataset Preparation Script
# Wrapper script for prepare_chime6_dataset.py

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to display usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -i, --input PATH     Path to CHiME-5 dataset (required)"
    echo "  -o, --output PATH    Output directory for CHiME-6 dataset (required)"
    echo "  -v, --verbose        Enable verbose output"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 -i ~/datasets/CHiME5 -o ~/datasets/CHiME6_prepared -v"
    echo ""
}

# Parse command line arguments
CHIME5_PATH=""
OUTPUT_DIR=""
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            CHIME5_PATH="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check required arguments
if [[ -z "$CHIME5_PATH" ]]; then
    print_error "CHiME-5 input path is required"
    show_usage
    exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    print_error "Output directory is required"
    show_usage
    exit 1
fi

# Print banner
echo "="*60
echo "CHiME-6 Dataset Preparation"
echo "Based on Kaldi s5_track1 recipe stage 0"
echo "="*60
echo ""

# Check dependencies
print_info "Checking dependencies..."

# Check Python
if ! command_exists python3; then
    print_error "Python 3 is required but not installed"
    exit 1
fi

# Check SoX
if ! command_exists sox; then
    print_error "SoX is required but not installed"
    print_info "Install SoX:"
    print_info "  macOS: brew install sox"
    print_info "  Ubuntu: sudo apt-get install sox"
    print_info "  CentOS: sudo yum install sox"
    exit 1
fi

# Check SoX version
SOX_VERSION=$(sox --version 2>&1 | head -n1)
print_info "Found SoX: $SOX_VERSION"

# Check if CHiME-5 path exists
if [[ ! -d "$CHIME5_PATH" ]]; then
    print_error "CHiME-5 path does not exist: $CHIME5_PATH"
    exit 1
fi

# Check CHiME-5 structure
print_info "Validating CHiME-5 dataset structure..."
if [[ ! -d "$CHIME5_PATH/audio" ]]; then
    print_error "CHiME-5 audio directory not found: $CHIME5_PATH/audio"
    exit 1
fi

if [[ ! -d "$CHIME5_PATH/transcriptions" ]]; then
    print_error "CHiME-5 transcriptions directory not found: $CHIME5_PATH/transcriptions"
    exit 1
fi

print_success "CHiME-5 dataset structure validated"

# Create output directory if it doesn't exist
if [[ ! -d "$OUTPUT_DIR" ]]; then
    print_info "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_SCRIPT="$SCRIPT_DIR/prepare_chime6_dataset.py"

# Check if Python script exists
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    print_error "Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Build Python command
PYTHON_CMD="python3 $PYTHON_SCRIPT --chime5_path '$CHIME5_PATH' --output_dir '$OUTPUT_DIR'"

if [[ "$VERBOSE" == true ]]; then
    PYTHON_CMD="$PYTHON_CMD --verbose"
fi

# Display configuration
echo ""
print_info "Configuration:"
print_info "  CHiME-5 Path: $CHIME5_PATH"
print_info "  Output Dir:   $OUTPUT_DIR"
print_info "  Verbose:      $VERBOSE"
echo ""

# Ask for confirmation
read -p "Proceed with CHiME-6 dataset preparation? [y/N] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "Operation cancelled by user"
    exit 0
fi

echo ""
print_info "Starting CHiME-6 dataset preparation..."
print_info "This may take a while depending on dataset size..."
echo ""

# Run the Python script
if eval "$PYTHON_CMD"; then
    echo ""
    print_success "CHiME-6 dataset preparation completed successfully!"
    print_info "Dataset location: $OUTPUT_DIR/CHiME6"
    print_info "Audio files: $OUTPUT_DIR/CHiME6/audio"
    print_info "Transcriptions: $OUTPUT_DIR/CHiME6/transcriptions"
    echo ""
    print_info "Next steps:"
    print_info "1. Validate the prepared dataset"
    print_info "2. Use with Whisper for speech recognition evaluation"
    print_info "3. See CHiME6_PREPARATION.md for usage examples"
else
    echo ""
    print_error "CHiME-6 dataset preparation failed!"
    print_info "Check the error messages above for details"
    exit 1
fi