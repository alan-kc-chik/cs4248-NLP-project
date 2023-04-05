cd /hpctmp/yk/CS4248/GAT/

module load singularity

singularity exec /app1/common/singularity-img/3.0.0/pytorch_1.9_cuda_11.3.0-ubuntu20.04-py38-ngc_21.04.simg bash

# inside container
cat /etc/os-release
cat /etc/lsb-release

# enable it when encountering "ERROR: Could not install packages due to an EnvironmentError: [Errno 28] No space left on device"
# export TMPDIR='/home/svu/e0741024/tmp'

# run the following if requirements.txt does not work
chmod +x pip.sh
./pip.sh

pip freeze --path /home/svu/e0741024/PyPackages/CS4248/GAT/lib/python3.8/site-packages > /hpctmp/yk/CS4248/GAT/requirements_exported.txt

