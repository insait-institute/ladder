export LADDER_CACHE=${LADDER_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/ladder}
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
for tool in curl python3 gcc; do
  command -v $tool >/dev/null 2>&1 || { echo "LADDER needs $tool on PATH" >&2; exit 1; }
done

if ! command -v bazel >/dev/null 2>&1; then
  if [ ! -x "$LADDER_CACHE/bin/bazel" ]; then
    [ "$(uname -m)" = x86_64 ] || { echo "LADDER runs on Linux x86_64 only" >&2; exit 1; }
    echo "Downloading Bazelisk to $LADDER_CACHE/bin"
    mkdir -p "$LADDER_CACHE/bin"
    tmp=$(mktemp "$LADDER_CACHE/bin/.bazel.XXXXXX")
    curl -fsSL -o "$tmp" "https://github.com/bazelbuild/bazelisk/releases/download/v1.29.0/bazelisk-linux-amd64" &&
      echo "5a408715e932c0250d28bd84555f12edbf70117de42f9181691c736eacc4a992  $tmp" | sha256sum -c --quiet - || { rm -f "$tmp"; echo "Bazelisk download failed" >&2; exit 1; }
    chmod +x "$tmp" && mv "$tmp" "$LADDER_CACHE/bin/bazel"
  fi
  export PATH="$LADDER_CACHE/bin:$PATH"
fi
