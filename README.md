# Methods of Contrastive Leraning

## How to use

### Run Docker Container
```bash
docker build -t adwaver4157/torch .
docker run -it --rm --name takanami_torch \
               --gpus all --shm-size=16gb \
               --net=host \
               --mount type=bind,source="$(pwd)",target=/root/workspace \
               adwaver4157/torch:latest
```

### Run train.py