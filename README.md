# Active Learning of Atomistic Surrogate Models for Rare Events (AL-ASMR)

## Docker image
We support both docker and singularity containers. 

We build a docker image as follows
```
cd Docker && docker build . -t activeml && cd ..
```

Push the built docker iamge to the Docker hub as follows:
```
docker tag activeml username/activeml
docker push activeml username/activeml
```

We build a singularity container by using the docker image:
```
singularity build -s activeml docker://username/activeml
```

A few example commands to check the container:
```
docker run -it activeml /bin/bash
docker run -it activeml python ./schnet.py
docker run -it -v "$(pwd)"/AL_ani:/workspace activeml /bin/bash
docker run -it -v "$(pwd)"/AL_schnet:/workspace activeml /bin/bash
```


## Run

Run with docker on a desktop:
```
python active.py --docker --container_name "activeml" --num_workers=4
```

Run with singularity on CADES:
```
python active.py --singularity --container_name "/path/to/activeml" --num_workers=4 --srun
```

Run inside a container:
```
docker run -it -v "$(pwd)"/AL_ani:/workspace activeml python active.py --nocontainer
```
