FROM python:3.12-slim-bookworm@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254
RUN useradd --create-home agent && mkdir /app && chown agent:agent /app
COPY model.py /model.py
RUN for module in httpx msgspec; do mkdir -p /hostile/$module \
    && echo 'raise ImportError("hostile PYTHONPATH")' > /hostile/$module/__init__.py; done
ENV LANG= PYTHONPATH=/hostile
WORKDIR /app
USER agent
ENTRYPOINT ["python", "/model.py"]
