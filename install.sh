#!/usr/bin/env bash
# Install spark-llm on an NVIDIA DGX Spark. Run with sudo from this repository.
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
    echo "Run this installer with sudo: sudo ./install.sh" >&2
    exit 1
fi

SRC="${SPARK_LLM_SRC:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
OPERATOR_USER="${SPARK_LLM_USER:-${SUDO_USER:-}}"
MODELS_DIR="${SPARK_LLM_MODELS_DIR:-/srv/models}"

if [[ -z "$OPERATOR_USER" || "$OPERATOR_USER" == "root" ]]; then
    echo "Set SPARK_LLM_USER to the non-root account that will operate spark-llm." >&2
    exit 1
fi
if [[ ! "$OPERATOR_USER" =~ ^[a-z_][a-z0-9_-]*\$?$ ]] || ! id "$OPERATOR_USER" >/dev/null 2>&1; then
    echo "Invalid or missing operator account: $OPERATOR_USER" >&2
    exit 1
fi
OPERATOR_GROUP="$(id -gn "$OPERATOR_USER")"

for command_name in python3 docker systemctl nvidia-smi visudo; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Missing required command: $command_name" >&2
        exit 1
    fi
done
if ! python3 -c "import rich" >/dev/null 2>&1; then
    echo "Missing Python dependency 'rich'. Install it with:" >&2
    echo "  sudo apt-get install python3-rich" >&2
    exit 1
fi
if [[ ! -f "$SRC/spark_llm.py" || ! -f "$SRC/etc/config.toml" ]]; then
    echo "Incomplete source tree: $SRC" >&2
    exit 1
fi

echo "== Install application =="
install -D -m 0755 "$SRC/spark_llm.py" /usr/local/lib/spark-llm/spark_llm.py
install -D -m 0644 "$SRC/test_console_paths.py" /usr/local/lib/spark-llm/test_console_paths.py
install -D -m 0644 "$SRC/image_server.py" /usr/local/lib/spark-llm/image_server.py
install -D -m 0644 "$SRC/image_server_lib.py" /usr/local/lib/spark-llm/image_server_lib.py
install -D -m 0644 "$SRC/Dockerfile.diffusers" /usr/local/lib/spark-llm/Dockerfile.diffusers

printf '%s\n' \
    '#!/usr/bin/env bash' \
    'exec python3 /usr/local/lib/spark-llm/spark_llm.py "$@"' \
    > /usr/local/bin/spark-llm
chmod 0755 /usr/local/bin/spark-llm

echo "== Optional llama.cpp engine =="
LLAMACPP_SOURCE="${SPARK_LLM_LLAMACPP_BIN:-/home/$OPERATOR_USER/llama.cpp/build-static/bin/llama-server}"
if [[ -x "$LLAMACPP_SOURCE" ]]; then
    install -D -m 0755 "$LLAMACPP_SOURCE" /usr/local/lib/spark-llm/llama-server
else
    echo "llama-server not found at $LLAMACPP_SOURCE; GGUF declarations will remain blocked."
fi

echo "== Install safe default configuration =="
install -d -o "$OPERATOR_USER" -g "$OPERATOR_GROUP" -m 0755 "$MODELS_DIR"
install -d -o "$OPERATOR_USER" -g "$OPERATOR_GROUP" -m 0755 \
    /etc/spark-llm /etc/spark-llm/models.d /etc/spark-llm/logs

if [[ ! -f /etc/spark-llm/config.toml ]]; then
    install -o "$OPERATOR_USER" -g "$OPERATOR_GROUP" -m 0644 \
        "$SRC/etc/config.toml" /etc/spark-llm/config.toml
    sed -i "s|^models_dir = .*|models_dir = \"$MODELS_DIR\"|" /etc/spark-llm/config.toml
else
    echo "Keeping existing /etc/spark-llm/config.toml"
fi

for declaration in "$SRC"/etc/models.d/*.toml; do
    target="/etc/spark-llm/models.d/$(basename "$declaration")"
    if [[ ! -e "$target" ]]; then
        install -o "$OPERATOR_USER" -g "$OPERATOR_GROUP" -m 0644 "$declaration" "$target"
    elif ! cmp -s "$declaration" "$target"; then
        echo "DIFFERS (not overwritten): $target"
    fi
done

# Missing mode means test mode. Preserve an existing operator-selected mode.
if [[ ! -e /etc/spark-llm/mode ]]; then
    echo "test" > /etc/spark-llm/mode
    chown "$OPERATOR_USER:$OPERATOR_GROUP" /etc/spark-llm/mode
fi

echo "== Build optional image-generation runtime =="
if [[ "${SPARK_LLM_SKIP_DIFFUSERS:-0}" != "1" ]]; then
    if ! docker image inspect spark-llm-diffusers:latest >/dev/null 2>&1; then
        docker build -f "$SRC/Dockerfile.diffusers" -t spark-llm-diffusers:latest "$SRC"
    fi
    docker run --rm --gpus all spark-llm-diffusers:latest \
        python3 -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
else
    echo "Skipping Diffusers image build (SPARK_LLM_SKIP_DIFFUSERS=1)."
fi

echo "== Install systemd template and scoped sudo rule =="
install -m 0644 "$SRC/vllm@.service" /etc/systemd/system/vllm@.service
systemctl daemon-reload

SYSTEMCTL_PATH="$(command -v systemctl)"
SUDOERS_TMP="$(mktemp)"
trap 'rm -f "$SUDOERS_TMP"' EXIT
printf 'Cmnd_Alias SPARK_LLM_UNITS = %s start vllm@*.service, %s stop vllm@*.service\n%s ALL=(root) NOPASSWD: SPARK_LLM_UNITS\n' \
    "$SYSTEMCTL_PATH" "$SYSTEMCTL_PATH" "$OPERATOR_USER" > "$SUDOERS_TMP"
visudo -cf "$SUDOERS_TMP"
install -m 0440 "$SUDOERS_TMP" /etc/sudoers.d/spark-llm

chown -R "$OPERATOR_USER:$OPERATOR_GROUP" /etc/spark-llm
chmod -R u+rwX,go+rX /etc/spark-llm

if ! id -nG "$OPERATOR_USER" | grep -qw docker; then
    echo
    echo "WARNING: $OPERATOR_USER is not in the docker group."
    echo "Run: sudo usermod -aG docker $OPERATOR_USER"
    echo "Then sign out and back in before using live mode."
fi

echo "== Verify =="
runuser -u "$OPERATOR_USER" -- env SPARK_LLM_DIR=/etc/spark-llm /usr/local/bin/spark-llm status
echo
echo "Installed in TEST mode. Add model weights, review declarations, run"
echo "spark-llm selftest, then explicitly enable mutations with:"
echo "  spark-llm mode live"
