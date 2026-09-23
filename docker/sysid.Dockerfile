# SO-101 sysid runner: MuJoCo sysid framework + mjbatch.
# mujoco[sysid] and mjbatch both ship cp313 manylinux wheels; mjbatch pins mujoco==3.13.0.
FROM python:3.13-slim

RUN pip install --no-cache-dir \
    "mujoco[sysid]==3.13.0" \
    "mjbatch==0.1.1" \
    "numpy<2.4" \
    "scipy"

WORKDIR /work
CMD ["python", "scripts/run-so101-sysid.py", "--help"]
