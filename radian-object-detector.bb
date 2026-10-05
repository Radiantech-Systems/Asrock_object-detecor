
SUMMARY = "Radian YOLO26 DeepStream Object Detector"
DESCRIPTION = "YOLO26 DeepStream object detection with timestamped JPEG snapshots, annotated live stream, automatic event video generation and Flask snapshot server"
LICENSE = "CLOSED"


PACKAGE_ARCH = "aarch64"

# ============================================================
# Source files
# ============================================================

SRC_URI = " \
    file://model/yolo26s.onnx \
    file://model/yolo26s.onnx.data \
    file://model/labels.txt \
    file://config_infer_primary_yolo26.txt \
    file://src/nvdsinfer_custom_impl_Yolo/ \
    file://src/object-detector.cpp \
    file://src/video_event_generator.cpp \
    file://snapshot_server.py \
    file://service/object-detector.service \
    file://service/snapshot-server.service \
"

S = "${WORKDIR}"


# ============================================================
# Build dependencies
# ============================================================

DEPENDS = " \
    pkgconfig-native \
    deepstream-7.1 \
    tensorrt-core \
    tensorrt-plugins-prebuilt \
    cuda-nvcc-native \
    cuda-cudart \
    libcublas \
    opencv \
    gstreamer1.0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-rtsp-server \
    nlohmann-json \
"


# ============================================================
# Runtime dependencies
# ============================================================

RDEPENDS:${PN} += " \
    ffmpeg \
    python3-core \
    python3-flask \
    python3-werkzeug \
"


# ============================================================
# Classes
# ============================================================

inherit pkgconfig systemd


# ============================================================
# DeepStream library path
# ============================================================

TARGET_LDFLAGS:append = " \
    -L${RECIPE_SYSROOT}/opt/nvidia/deepstream/deepstream-7.1/lib \
"


# ============================================================
# Compile
# ============================================================

do_compile() {

    # ========================================================
    # Build YOLO26 custom DeepStream parser
    # ========================================================

    cd ${WORKDIR}/src/nvdsinfer_custom_impl_Yolo

    export CUDA_VER="12.6"

    export NVCC="${STAGING_DIR_NATIVE}/usr/local/cuda-12.6/bin/nvcc"

    echo "============================================================"
    echo "YOLO26 custom parser build"
    echo "============================================================"
    echo "NVCC      = ${NVCC}"
    echo "SYSROOT   = ${RECIPE_SYSROOT}"
    echo "CC        = ${CC}"
    echo "CXX       = ${CXX}"
    echo "============================================================"

    file "${NVCC}"
    test -x "${NVCC}"

    oe_runmake \
        CUDA_VER="${CUDA_VER}" \
        CC="${CC}" \
        CXX="${CXX}" \
        NVCC="${NVCC}" \
        SYSROOT="${RECIPE_SYSROOT}" \
        CPPFLAGS="${CPPFLAGS}" \
        CXXFLAGS="${CXXFLAGS}" \
        LDFLAGS="${LDFLAGS}"


    # ========================================================
    # Verify custom parser
    # ========================================================

    test -f \
        ${WORKDIR}/src/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so


    # ========================================================
    # Build object-detector application
    # ========================================================

    cd ${WORKDIR}/src

    echo "============================================================"
    echo "Building YOLO26 object-detector"
    echo "============================================================"

    echo "GStreamer version:"
    pkg-config --modversion gstreamer-1.0

    echo "GStreamer App version:"
    pkg-config --modversion gstreamer-app-1.0

    echo "GStreamer RTSP Server version:"
    pkg-config --modversion gstreamer-rtsp-server-1.0

    echo "OpenCV version:"
    pkg-config --modversion opencv4


    ${CXX} \
        ${CXXFLAGS} \
        ${CPPFLAGS} \
        -std=c++17 \
        -Wall \
        -Wextra \
        -fPIC \
        -pthread \
        $(pkg-config --cflags gstreamer-1.0) \
        $(pkg-config --cflags gstreamer-base-1.0) \
        $(pkg-config --cflags gstreamer-app-1.0) \
        $(pkg-config --cflags gstreamer-rtsp-server-1.0) \
        $(pkg-config --cflags opencv4) \
        -I${RECIPE_SYSROOT}/opt/nvidia/deepstream/deepstream-7.1/sources/includes \
        -I${RECIPE_SYSROOT}/usr/include \
        object-detector.cpp \
        -o object-detector \
        ${LDFLAGS} \
        $(pkg-config --libs gstreamer-1.0) \
        $(pkg-config --libs gstreamer-base-1.0) \
        $(pkg-config --libs gstreamer-app-1.0) \
        $(pkg-config --libs gstreamer-rtsp-server-1.0) \
        $(pkg-config --libs opencv4) \
        -L${RECIPE_SYSROOT}/opt/nvidia/deepstream/deepstream-7.1/lib \
        -lnvds_meta \
        -lnvdsgst_meta \
        -lnvds_infer \
        -lnvbufsurface \
        -lnvbufsurftransform


    # ========================================================
    # Build 2-minute event video generator
    # ========================================================

    echo "============================================================"
    echo "Building 2-minute event video generator"
    echo "============================================================"

    ${CXX} \
        ${CXXFLAGS} \
        ${CPPFLAGS} \
        -std=c++17 \
        -Wall \
        -Wextra \
        -pthread \
        video_event_generator.cpp \
        -o video_event_generator \
        ${LDFLAGS}
}


# ============================================================
# Install
# ============================================================

do_install() {

    # ========================================================
    # Application executables
    # ========================================================

    install -d ${D}${bindir}

    install -m 0755 \
        ${WORKDIR}/src/object-detector \
        ${D}${bindir}/object-detector

    install -m 0755 \
        ${WORKDIR}/src/video_event_generator \
        ${D}${bindir}/video-event-generator


    # ========================================================
    # Flask Snapshot Server
    # ========================================================

    install -m 0755 \
        ${WORKDIR}/snapshot_server.py \
        ${D}${bindir}/snapshot_server.py


    # ========================================================
    # YOLO26 custom parser
    # ========================================================

    install -d ${D}/opt/radian/ai/lib

    install -m 0755 \
        ${WORKDIR}/src/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so \
        ${D}/opt/radian/ai/lib/libnvdsinfer_custom_impl_Yolo.so


    # ========================================================
    # YOLO26 model
    # ========================================================

    install -d ${D}/opt/radian/ai/model

    install -m 0644 \
        ${WORKDIR}/model/yolo26s.onnx \
        ${D}/opt/radian/ai/model/yolo26s.onnx

    install -m 0644 \
        ${WORKDIR}/model/yolo26s.onnx.data \
        ${D}/opt/radian/ai/model/yolo26s.onnx.data

    install -m 0644 \
        ${WORKDIR}/model/labels.txt \
        ${D}/opt/radian/ai/model/labels.txt


    # ========================================================
    # DeepStream configuration
    # ========================================================

    install -d ${D}/opt/radian/ai/config

    install -m 0644 \
        ${WORKDIR}/config_infer_primary_yolo26.txt \
        ${D}/opt/radian/ai/config/config_infer_primary_yolo26.txt


    # ========================================================
    # Snapshot directories
    # ========================================================

    install -d \
        ${D}/root/video_recorder/snapshots/people

    install -d \
        ${D}/root/video_recorder/snapshots/vehicles

    install -d \
        ${D}/root/video_recorder/snapshots/animals

    install -d \
        ${D}/root/video_recorder/snapshots/electronics

    install -d \
        ${D}/root/video_recorder/snapshots/other


    # ========================================================
    # Event directories
    # ========================================================

    install -d \
        ${D}/root/video_recorder/events/pending

    install -d \
        ${D}/root/video_recorder/events/videos


    # ========================================================
    # Systemd services
    # ========================================================

    install -d \
        ${D}${systemd_system_unitdir}

    install -m 0644 \
        ${WORKDIR}/service/object-detector.service \
        ${D}${systemd_system_unitdir}/object-detector.service

    install -m 0644 \
        ${WORKDIR}/service/snapshot-server.service \
        ${D}${systemd_system_unitdir}/snapshot-server.service
}


# ============================================================
# Package files
# ============================================================

FILES:${PN} += " \
    ${bindir}/object-detector \
    ${bindir}/video-event-generator \
    ${bindir}/snapshot_server.py \
    /opt/radian/ai \
    /opt/radian/ai/lib \
    /opt/radian/ai/lib/libnvdsinfer_custom_impl_Yolo.so \
    /opt/radian/ai/model \
    /opt/radian/ai/model/yolo26s.onnx \
    /opt/radian/ai/model/yolo26s.onnx.data \
    /opt/radian/ai/model/labels.txt \
    /opt/radian/ai/config \
    /opt/radian/ai/config/config_infer_primary_yolo26.txt \
    /root/video_recorder \
    /root/video_recorder/snapshots \
    /root/video_recorder/snapshots/people \
    /root/video_recorder/snapshots/vehicles \
    /root/video_recorder/snapshots/animals \
    /root/video_recorder/snapshots/electronics \
    /root/video_recorder/snapshots/other \
    /root/video_recorder/events \
    /root/video_recorder/events/pending \
    /root/video_recorder/events/videos \
    ${systemd_system_unitdir}/object-detector.service \
    ${systemd_system_unitdir}/snapshot-server.service \
"


# ============================================================
# Systemd
# ============================================================

SYSTEMD_SERVICE:${PN} = " \
    object-detector.service \
    snapshot-server.service \
"

SYSTEMD_AUTO_ENABLE:${PN} = "enable"

