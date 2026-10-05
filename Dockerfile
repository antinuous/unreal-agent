FROM ghcr.io/astral-sh/uv:0.12.23-trixie-slim@sha256:cb8750e12902fdaca60c1e48a796d19c02e6c2037ca801f322689fa97f0a44f2 AS build
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PYTHON=3.14 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PROJECT_ENVIRONMENT=/opt/unreal-agent
WORKDIR /src
RUN uv python install \
    && cd /opt/python/cpython-*/ \
    && rm -rf include lib/itcl* lib/libtcl* lib/tcl* lib/thread* lib/tk* \
        lib/python3.14/ensurepip lib/python3.14/idlelib lib/python3.14/site-packages/pip* \
        lib/python3.14/tkinter lib/python3.14/turtledemo lib/python3.14/lib-dynload/_tkinter.*
# uv sync --locked reads every workspace member's pyproject.toml.
COPY pyproject.toml uv.lock ./
COPY packages/unreal-agent/pyproject.toml packages/unreal-agent/
COPY benchmarks/harbor/pyproject.toml benchmarks/harbor/
RUN uv sync --locked --no-dev --no-install-workspace --package unreal-agent
COPY packages/unreal-agent/src/ packages/unreal-agent/src/
RUN uv sync --locked --no-dev --no-editable --package unreal-agent

# Debian 13.7 (trixie), image build 2026-09-18.
FROM debian:trixie-slim@sha256:a99cfc517144bc59b1978475ec53b46ecabec7e43635402ee5b77cc54cd1b20a
RUN apt-get update && apt-get install -y --no-install-recommends bash ca-certificates tini \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /workspace /home/agent /state \
    && chmod 1777 /state \
    && chown 10001:10001 /workspace /home/agent
COPY --from=build /opt/python /opt/python
COPY --from=build /opt/unreal-agent /opt/unreal-agent
# Only the runner goes on PATH, so the agent's Bash tool does not see the runner's Python.
# PEP 538 locale coercion sets LC_CTYPE=C.UTF-8 in the C locale, and -I stops
# PYTHONCOERCECLOCALE from disabling it. The launcher restores the caller's
# LC_CTYPE so tool processes inherit the container's environment unchanged.
COPY --chmod=755 <<"EOF" /usr/local/bin/unreal-agent-runner
#!/bin/sh
exec /opt/unreal-agent/bin/python -I -c '
import os, sys
if sys.argv[1]:
    os.environ["LC_CTYPE"] = sys.argv[2]
else:
    os.environ.pop("LC_CTYPE", None)
sys.argv[:3] = ["unreal-agent-runner"]
from unreal_agent.runner.__main__ import main
main()
' "${LC_CTYPE+set}" "${LC_CTYPE-}" "$@"
EOF
COPY LICENSE /usr/share/doc/unreal-agent/LICENSE
ENV HOME=/home/agent SHELL=/bin/bash XDG_STATE_HOME=/state
USER 10001:10001
WORKDIR /workspace
ENTRYPOINT ["/usr/bin/tini", "--", "unreal-agent-runner"]
