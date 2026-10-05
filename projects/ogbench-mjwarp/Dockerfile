FROM huggingface/lerobot-gpu@sha256:031a10c9bb2437859da3621e475fd5eff4abfc619c9976247802b0c90706237e

COPY packages/vla-tools /opt/vla/packages/vla-tools
COPY projects/ogbench-mjwarp /opt/vla/projects/ogbench-mjwarp
WORKDIR /opt/vla/projects/ogbench-mjwarp
RUN uv sync --frozen --extra dataset --extra evaluation --extra dev --python /lerobot/.venv/bin/python

ENV PATH="/opt/vla/projects/ogbench-mjwarp/.venv/bin:${PATH}" \
    MUJOCO_GL=egl \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    PYTHONUNBUFFERED=1
ENTRYPOINT ["ogbench-mjwarp"]
CMD ["doctor"]
