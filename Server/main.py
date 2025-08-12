import os, sys
import asyncio
import time
from collections import deque
import cv2
import numpy as np
import websockets
import struct
import json
from PIL import Image, ImageDraw, ImageFont

from modules import HandTracker_mp, ObjTracker, HandTracker_our, recog_contact
from handtracker.utils.visualize import draw_2d_skeleton

sys.path.append("./hl2ss_")
import hl2ss
import hl2ss_lnm
import hl2ss_mp
import hl2ss_3dcv
import hl2ss_utilities
import socket, threading
import multiprocessing as mp


## args ##
flag_mediapipe = False

## Set HoloLens2 wifi address ##
host = '192.168.50.31'

# Calibration path (must exist but can be empty)
calibration_path = 'calibration'
if not os.path.isdir(calibration_path):
    os.mkdir(calibration_path)

# Front RGB camera parameters
pv_width = 1280
pv_height = 720
pv_fps = 30

# Buffer length in seconds
buffer_size = 2

# process depth image per n frame
num_depth_count = 0     # 0 for only rgb

contact_que = deque([], maxlen=5)


def init_hl2():
    # Start PV Subsystem ------------------------------------------------------
    hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

    # Get RM Depth AHAT calibration -------------------------------------------
    # Calibration data will be downloaded if it's not in the calibration folder
    calibration_ht = hl2ss_3dcv.get_calibration_rm(calibration_path, host, hl2ss.StreamPort.RM_DEPTH_AHAT)

    uv2xy = calibration_ht.uv2xy  # hl2ss_3dcv.compute_uv2xy(calibration_ht.intrinsics, hl2ss.Parameters_RM_DEPTH_AHAT.WIDTH, hl2ss.Parameters_RM_DEPTH_AHAT.HEIGHT)
    xy1, scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, calibration_ht.scale)
    max_depth = calibration_ht.alias / calibration_ht.scale

    xy1_o = hl2ss_3dcv.block_to_list(xy1[:-1, :-1, :])
    xy1_d = hl2ss_3dcv.block_to_list(xy1[1:, 1:, :])

    # Start PV and RM Depth AHAT streams --------------------------------------
    producer = hl2ss_mp.producer()
    producer.configure(hl2ss.StreamPort.PERSONAL_VIDEO,
                       hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, width=pv_width, height=pv_height,
                                       framerate=pv_fps))
    producer.configure(hl2ss.StreamPort.RM_DEPTH_AHAT, hl2ss_lnm.rx_rm_depth_ahat(host, hl2ss.StreamPort.RM_DEPTH_AHAT))
    producer.initialize(hl2ss.StreamPort.PERSONAL_VIDEO, pv_fps * buffer_size)
    producer.initialize(hl2ss.StreamPort.RM_DEPTH_AHAT, hl2ss.Parameters_RM_DEPTH_AHAT.FPS * buffer_size)
    producer.start(hl2ss.StreamPort.PERSONAL_VIDEO)
    producer.start(hl2ss.StreamPort.RM_DEPTH_AHAT)

    consumer = hl2ss_mp.consumer()
    manager = mp.Manager()
    sink_pv = consumer.create_sink(producer, hl2ss.StreamPort.PERSONAL_VIDEO, manager, None)
    sink_ht = consumer.create_sink(producer, hl2ss.StreamPort.RM_DEPTH_AHAT, manager, None)

    sink_pv.get_attach_response()
    sink_ht.get_attach_response()

    # Initialize PV intrinsics and extrinsics ---------------------------------
    pv_intrinsics = hl2ss_3dcv.pv_create_intrinsics_placeholder()
    pv_extrinsics = np.eye(4, 4, dtype=np.float32)

    return [sink_ht, sink_pv, pv_intrinsics, pv_extrinsics, xy1_o, xy1_d, scale, calibration_ht], max_depth, producer


def receive_images(init_variables, flag_depth):

    global depth
    sink_ht, sink_pv, pv_intrinsics, pv_extrinsics, xy1_o, xy1_d, scale, calibration_ht = init_variables

    # Get RM Depth AHAT frame and nearest (in time) PV frame --------------
    _, data_ht = sink_ht.get_most_recent_frame()
    if ((data_ht is None) or (not hl2ss.is_valid_pose(data_ht.pose))):
        return None
    _, data_pv = sink_pv.get_nearest(data_ht.timestamp)
    if ((data_pv is None) or (not hl2ss.is_valid_pose(data_pv.pose))):
        return None

    # Preprocess frames ---------------------------------------------------
    color = data_pv.payload.image
    pv_z = None
    if flag_depth:
        depth = data_ht.payload.depth  # hl2ss_3dcv.rm_depth_undistort(data_ht.payload.depth, calibration_ht.undistort_map)
        z = hl2ss_3dcv.rm_depth_normalize(depth, scale)

    # Update PV intrinsics ------------------------------------------------
    # PV intrinsics may change between frames due to autofocus
    pv_intrinsics = hl2ss_3dcv.pv_update_intrinsics(pv_intrinsics, data_pv.payload.focal_length, data_pv.payload.principal_point)
    color_intrinsics, color_extrinsics = hl2ss_3dcv.pv_fix_calibration(pv_intrinsics, pv_extrinsics)

    # Generate depth map for PV image -------------------------------------
    if flag_depth:
        mask = (depth[:-1, :-1].reshape((-1,)) > 0)
        zv = hl2ss_3dcv.block_to_list(z[:-1, :-1, :])[mask, :]

        ht_to_pv_image = hl2ss_3dcv.camera_to_rignode(calibration_ht.extrinsics) @ hl2ss_3dcv.reference_to_world(
            data_ht.pose) @ hl2ss_3dcv.world_to_reference(data_pv.pose) @ hl2ss_3dcv.rignode_to_camera(
            color_extrinsics) @ hl2ss_3dcv.camera_to_image(color_intrinsics)

        ht_points_o = hl2ss_3dcv.rm_depth_to_points(xy1_o[mask, :], zv)
        pv_uv_o_h = hl2ss_3dcv.transform(ht_points_o, ht_to_pv_image)
        pv_list_depth = pv_uv_o_h[:, 2:]

        ht_points_d = hl2ss_3dcv.rm_depth_to_points(xy1_d[mask, :], zv)
        pv_uv_d_h = hl2ss_3dcv.transform(ht_points_d, ht_to_pv_image)
        pv_d_depth = pv_uv_d_h[:, 2:]

        mask = (pv_list_depth[:, 0] > 0) & (pv_d_depth[:, 0] > 0)

        pv_list_depth = pv_list_depth[mask, :]
        pv_d_depth = pv_d_depth[mask, :]

        pv_list_o = pv_uv_o_h[mask, 0:2] / pv_list_depth
        pv_list_d = pv_uv_d_h[mask, 0:2] / pv_d_depth

        pv_list = np.hstack((pv_list_o, pv_list_d + 1)).astype(np.int32)
        pv_z = np.zeros((pv_height, pv_width), dtype=np.float32)

        u0 = pv_list[:, 0]
        v0 = pv_list[:, 1]
        u1 = pv_list[:, 2]
        v1 = pv_list[:, 3]

        mask0 = (u0 >= 0) & (u0 < pv_width) & (v0 >= 0) & (v0 < pv_height)
        mask1 = (u1 > 0) & (u1 <= pv_width) & (v1 > 0) & (v1 <= pv_height)
        maskf = mask0 & mask1

        pv_list = pv_list[maskf, :]
        pv_list_depth = pv_list_depth[maskf, 0]

        for n in range(0, pv_list.shape[0]):
            u0 = pv_list[n, 0]
            v0 = pv_list[n, 1]
            u1 = pv_list[n, 2]
            v1 = pv_list[n, 3]

            pv_z[v0:v1, u0:u1] = pv_list_depth[n]

    return color, pv_z


def main_single():
    ###################### init models ######################
    if flag_mediapipe:
        track_hand = HandTracker_mp()
    else:
        track_hand = HandTracker_our()


    ###################### init comm. with hololens2 ######################
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    init_variables, max_depth, producer = init_hl2()

    cv2.namedWindow('Prompt')
    cv2.resizeWindow(winname='Prompt', width=500, height=500)
    cv2.moveWindow(winname='Prompt', x=2000, y=200)

    while True:
        flag_depth = False
        ###################### receive input ######################
        result = receive_images(init_variables, flag_depth)
        if result == None:
            continue

        color, depth = result
        color = cv2.resize(color, dsize=(640, 360), interpolation=cv2.INTER_AREA)

        ###################### process hand ######################
        result_hand = track_hand.run(np.copy(color))
        if not isinstance(result_hand, np.ndarray):
            continue

        ###################### visualize ######################
        img_cv = draw_2d_skeleton(color, result_hand)
        cv2.imshow("Prompt", img_cv)
        cv2.waitKey(1)

        ## send to hololens2
        send_data = np.append(result_hand.flatten())
        print("sending ... ", send_data)

    sock.close()

    # Stop PV and RM Depth AHAT streams ---------------------------------------
    sink_ht, sink_pv = init_variables[0], init_variables[1]
    sink_pv.detach()
    sink_ht.detach()
    producer.stop(hl2ss.StreamPort.PERSONAL_VIDEO)
    producer.stop(hl2ss.StreamPort.RM_DEPTH_AHAT)

    # Stop PV subsystem -------------------------------------------------------
    hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)
    cv2.destroyAllWindows()

# binder함수는 서버에서 accept가 되면 생성되는 socket 인스턴스를 통해 client로 부터 데이터를 받으면 echo형태로 재송신하는 메소드이다.
def binder(client_socket, addr):
    # 커넥션이 되면 접속 주소가 나온다.
    print('Connected by', addr)
    try:
        # 접속 상태에서는 클라이언트로 부터 받을 데이터를 무한 대기한다.
        # 만약 접속이 끊기게 된다면 except가 발생해서 접속이 끊기게 된다.
        while True:
            # socket의 recv함수는 연결된 소켓으로부터 데이터를 받을 대기하는 함수입니다. 최초 4바이트를 대기합니다.
            data = client_socket.recv(4)
            # 최초 4바이트는 전송할 데이터의 크기이다. 그 크기는 little big 엔디언으로 byte에서 int형식으로 변환한다.
            # C#의 BitConverter는 big엔디언으로 처리된다.
            length = int.from_bytes(data, "little")
            # 다시 데이터를 수신한다.
            data = client_socket.recv(length)
            # 수신된 데이터를 str형식으로 decode한다.
            msg = data.decode()
            # 수신된 메시지를 콘솔에 출력한다.
            print('Received from', addr, msg)

            # 수신된 메시지 앞에 「echo:」 라는 메시지를 붙힌다.
            msg = "echo : " + msg
            # 바이너리(byte)형식으로 변환한다.
            data = msg.encode()
            # 바이너리의 데이터 사이즈를 구한다.
            length = len(data)
            # 데이터 사이즈를 little 엔디언 형식으로 byte로 변환한 다음 전송한다.
            client_socket.sendall(length.to_bytes(4, byteorder='little'))
            # 데이터를 클라이언트로 전송한다.
            client_socket.sendall(data)
    except:
        # 접속이 끊기면 except가 발생한다.
        print("except : ", addr)
    finally:
        # 접속이 끊기면 socket 리소스를 닫는다.
        client_socket.close()


if __name__ == '__main__':
    main_single()