.PHONY: help sync build rebuild smoke clean
.DEFAULT_GOAL := help

# Why this Makefile exists:
#
# `llm-rs` is a PyO3 extension built by maturin. `uv run` auto-syncs
# the venv on every invocation, and uv's wheel cache holds prior
# builds keyed by (name, version). Since the version never bumps
# during development, uv happily reinstalls a stale cached wheel over
# the fresh `.so` that `maturin develop` just installed — silently
# clobbering your edits. Phase A and Phase B both burned time on this.
#
# The workaround is `UV_NO_SYNC=1` (or `uv run --no-sync`) anywhere we
# expect maturin to be the sole source of truth for the `.so`. This
# Makefile bakes that in for the build + smoke-test path, and the
# `help` target reminds you to export `UV_NO_SYNC=1` before running
# your own ad-hoc `uv run` commands.

MATURIN_ENV := PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
RUN := UV_NO_SYNC=1 uv run

help:
	@echo "llm-python Makefile targets:"
	@echo "  make sync     - uv sync (installs maturin + other dev deps)"
	@echo "  make build    - maturin develop (rebuild the .so; run after Rust edits)"
	@echo "  make rebuild  - sync + build (use on a clean clone or after dep changes)"
	@echo "  make smoke    - import llm_rs + assert Phase A/B surfaces are live"
	@echo "  make clean    - remove .venv and cargo target/"
	@echo ""
	@echo "For ad-hoc commands against the maturin-built .so, always use:"
	@echo "    UV_NO_SYNC=1 uv run <cmd>"
	@echo "Raw 'uv run' re-syncs and can reinstall a stale cached wheel."
	@echo "(Or: 'export UV_NO_SYNC=1' once in your shell.)"

sync:
	uv sync

build:
	$(MATURIN_ENV) $(RUN) maturin develop

rebuild: sync build

smoke:
	$(RUN) python -c "import llm_rs; \
c = llm_rs.LlmClient('unused', 'gpt-4o-mini', retries=2); \
assert callable(getattr(c, 'chain', None)), 'chain() missing'; \
assert callable(getattr(c, 'chain_stream', None)), 'chain_stream() missing'; \
assert llm_rs.ChainResult is not None, 'ChainResult missing'; \
print('llm_rs loaded:', llm_rs.__file__); \
print('Phase A + B surfaces present')"

clean:
	rm -rf .venv target
