#!/usr/bin/env bash
# Script to generate Cosign key pair for image signing

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if cosign is installed
if ! command -v cosign &> /dev/null; then
    print_error "cosign is not installed. Please install it first."
    print_info ""
    print_info "Installation options:"
    print_info "  macOS:   brew install cosign"
    print_info "  Linux:   https://docs.sigstore.dev/cosign/installation/"
    print_info "  Go:      go install github.com/sigstore/cosign/v2/cmd/cosign@latest"
    exit 1
fi

print_info "Cosign version: $(cosign version 2>&1 | head -1)"
print_info ""

# Set output directory
OUTPUT_DIR="${1:-ci/keys}"
mkdir -p "$OUTPUT_DIR"

print_step "Generating Cosign key pair..."
print_info "Output directory: $OUTPUT_DIR"
print_info ""

# Check if keys already exist
if [ -f "$OUTPUT_DIR/cosign.key" ] || [ -f "$OUTPUT_DIR/cosign.pub" ]; then
    print_warning "Cosign keys already exist in $OUTPUT_DIR"
    read -p "Do you want to overwrite them? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Aborted. Existing keys preserved."
        exit 0
    fi
    rm -f "$OUTPUT_DIR/cosign.key" "$OUTPUT_DIR/cosign.pub"
fi

# Generate key pair
print_step "Running: cosign generate-key-pair"
print_info "You will be prompted to enter a password to encrypt the private key."
print_info "Remember this password - you'll need it in ci/credentials.yml"
print_info ""

cd "$OUTPUT_DIR"
cosign generate-key-pair
cd - > /dev/null

print_info ""
print_info "✅ Cosign keys generated successfully!"
print_info ""
print_info "Files created:"
print_info "  Private key: $OUTPUT_DIR/cosign.key"
print_info "  Public key:  $OUTPUT_DIR/cosign.pub"
print_info ""

# Display the keys
print_step "Private Key (cosign.key):"
echo "----------------------------------------"
cat "$OUTPUT_DIR/cosign.key"
echo "----------------------------------------"
print_info ""

print_step "Public Key (cosign.pub):"
echo "----------------------------------------"
cat "$OUTPUT_DIR/cosign.pub"
echo "----------------------------------------"
print_info ""

# Instructions
print_step "Next Steps:"
print_info ""
print_info "1. Copy the keys to your credentials file:"
print_info "   vim ci/credentials.yml"
print_info ""
print_info "2. Add the following to ci/credentials.yml:"
print_info ""
echo "   cosign_private_key: |"
cat "$OUTPUT_DIR/cosign.key" | sed 's/^/     /'
echo ""
echo "   cosign_public_key: |"
cat "$OUTPUT_DIR/cosign.pub" | sed 's/^/     /'
echo ""
echo "   cosign_password: YOUR_PASSWORD_HERE"
print_info ""

print_warning "SECURITY NOTES:"
print_info "  - Keep cosign.key secure and never commit it to version control"
print_info "  - The password is required to use the private key"
print_info "  - Store the password securely (password manager, vault, etc.)"
print_info "  - The public key can be shared for signature verification"
print_info ""

print_step "Optional: Add keys to .gitignore"
print_info "  echo 'ci/keys/' >> .gitignore"
print_info ""

print_info "✅ Setup complete!"
