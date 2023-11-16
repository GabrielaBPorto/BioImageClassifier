cd "$(dirname "$0")"/..
RAW_ORGANIZED_DIR="data/raw/organized/imagens_ihq_er/"
if [ -d "$RAW_ORGANIZED_DIR" ]; then
    rm -rf "$RAW_ORGANIZED_DIR"
    echo "Organized directory cleaned up."
else
    echo "Organized directory does not exist."
fi