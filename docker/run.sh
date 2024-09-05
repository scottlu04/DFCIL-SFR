DOCKER_IMAGE=$(<version.txt)

#LOCAL_DIR=/home/jesus/Jesus/Gesture_recognition
#LOCAL_DIR=/home/azken/JESUS/Gesture_recognition
LOCAL_DIR=/home/jesus/Gesture_recognition
#LOCAL_DIR=/home/visilab/Jesus/Gesture_recognition

docker run \
    -it --gpus all \
    --name=ogr_cmu_v1-${USER} \
    -e DISPLAY=${DISPLAY} \
    --net=host \
    --ipc=host \
    -v ${HOME}/.Xauthority:/root/.Xauthority:rw \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ${LOCAL_DIR}/ogr_cmu/src:/ogr_cmu/src \
    -v ${LOCAL_DIR}/ogr_cmu/data/:/ogr_cmu/data/ \
    -v ${LOCAL_DIR}/ogr_cmu/output:/ogr_cmu/output \
    -v ${LOCAL_DIR}/ogr_cmu/scripts:/ogr_cmu/scripts \
    -v ${LOCAL_DIR}/ogr_cmu/.vscode:/ogr_cmu/.vscode \
    -v ${LOCAL_DIR}/ogr_cmu/models:/ogr_cmu/models \
${DOCKER_IMAGE}