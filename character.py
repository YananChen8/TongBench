import math
import os
import queue
import threading
import time
from collections import deque
from typing import List
import grpc
import numpy as np

import tongsim.common.enums as en
from tongsim.components.acoustic_component import AcousticComponent
from tongsim.components.character_movement_component import CharacterMovementComponent
from tongsim.components.container_component import ContainerComponent
from tongsim.components.grpc_tags import GrpcTags
from tongsim.common.ue_types import UELocation, UERotation, UEScale
from tongsim.components.input_animation_component import InputAnimationComponent
from tongsim.components.input_component import InputComponent
from tongsim.object.base_object import BaseObject
from tongsim.object.object_base import ObjectBase
from tongsim.components.animation_component import AnimationComponent
from tongsim.components.character_capsule_component import CharacterCapsuleComponent

from tongsim.components.attachment_component import AttachmentComponent
from tongsim.components.character_attribute_component import CharacterAttributeComponent
from tongsim.components.character_energy import CharacterEnergyComponent

from tongsim.components.image_component import ImageComponent
from tongsim.components.segmentation_component import SegmentationComponent
from tongsim.components.facial_animation_component import FacialAnimationComponent
from tongsim.components.face_component import FaceComponent

from tongsim.components.animation_tags import GrpcAnimationTags
from tongsim_api_protocol.component.animation import animation_pb2, sleep_pb2
from tongsim_api_protocol.component.animation import animation_pb2_grpc
from tongsim_api_protocol.component.animation import locomotion_pb2
from tongsim_api_protocol.component.animation import gesture_pb2
from tongsim_api_protocol.component.animation import wash_pb2
from tongsim_api_protocol import basic_pb2
from tongsim_api_protocol.subject import subject_pb2, subject_pb2_grpc

from tongsim_api_protocol.subsystem import (
    grpc_client_pb2_grpc,
    camera_pb2_grpc,
    camera_pb2,
)
from tongsim_api_protocol.component import (
    event_notifier_pb2,
    event_notifier_pb2_grpc,
    door_state_pb2,
    door_state_pb2_grpc,
    attachment_pb2,
    attachment_pb2_grpc,
    character_attribute_pb2,
    character_attribute_pb2_grpc,
)
from tongsim.object.dirt import Dirt
from tongsim_api_protocol.subsystem import scene_pb2, scene_pb2_grpc
from tongsim.common.enums import *

request_image_dict = {
    RequestImage.RIGHT: "_LeftEye",
    RequestImage.LEFT: "_RightEye",
    RequestImage.CENTER: "_CenterEye",
    RequestImage.RLCenterEye: "_RLCenterEye",
}


def eulur2quaternion(eulur):
    # 欧拉角转四元数，只支持xyz旋转
    # print(eulur)
    x, y, z = eulur
    x, y, z = math.radians(x), math.radians(y), math.radians(z)
    roll, pitch, yaw = x, y, z

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    # Quaternion q;
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr

    x = -x
    y = -y
    return UERotation(x, y, z, w)


class Character(ObjectBase):
    """
    表示一个智能体角色，继承自 ObjectBase，包含角色特有的组件和动作控制。
    """

    kAnimationSequence = "SDAS_"
    global request_image_dict

    def __init__(
        self,
        id_name,
        desired_name,
        stream_client,
        callreturn_client,
        scale=UEScale(1, 1, 1),
        proto_insecure_channel=None,
    ):
        """
        初始化 Character 对象。

        :param id_name: 角色的唯一标识名称
        :param desired_name: 角色的期望名称
        :param stream_client: 流客户端对象
        :param callreturn_client: 调用返回客户端对象
        :param scale: 角色的缩放比例 (UEScale)，默认为 (1, 1, 1)
        :param proto_insecure_channel: 不安全的 gRPC 通道对象（可选）
        """
        ObjectBase.__init__(
            self,
            id_name,
            desired_name,
            stream_client,
            callreturn_client,
            scale,
            proto_insecure_channel,
        )
        self.__animation_component = AnimationComponent(self)
        self.__image_component = ImageComponent(self)
        self.__character_capsule_component = CharacterCapsuleComponent(self)
        self.__attachment_component = AttachmentComponent(self)
        self.__character_energy_component = CharacterEnergyComponent(self)
        self.set_type("Character")
        self.__segmentation_component = SegmentationComponent(self)
        self.__facialanimation_component = FacialAnimationComponent(self)
        self.__animation_stub: animation_pb2_grpc.AnimationServiceStub
        self.__input_component = InputComponent(self, proto_insecure_channel)
        self.__acoustic_component = AcousticComponent(self, proto_insecure_channel)
        if proto_insecure_channel is not None:
            self.__animation_stub = animation_pb2_grpc.AnimationServiceStub(
                self.insecure_channel
            )
            self.__attribute_component = CharacterAttributeComponent(
                self, self.insecure_channel
            )
            self.__face_component = FaceComponent(self, proto_insecure_channel)

        self.__character_movement_component = CharacterMovementComponent(self, proto_insecure_channel)
        self.__input_animation_component = InputAnimationComponent(self, self.__animation_stub)
        # image cache
        self.mtx = threading.Lock()
        self.image_map = None
        self.stream_subscribe_image = None
        self.subscribe_image_thread = None

    @property
    def character_movement_component(self):
        """
        获取角色的移动组件。

        :return: 角色的 AnimationComponent 对象
        """
        return self.__character_movement_component

    @property
    def input_animation_component(self):
        """
        获取角色的动画组件。

        :return: 角色的 AnimationComponent 对象
        """
        return self.__input_animation_component

    @property
    def animation_component(self):
        """
        获取角色的动画组件。

        :return: 角色的 AnimationComponent 对象
        """
        return self.__animation_component

    def spawn_rl_camera(self, solution, fov):
        """
        为角色生成一个高帧率中心摄像机
        """
        if len(solution) != 2:
            raise "error, solution shape"

        camera_id = f"{self.id}_RLCenterEye"
        self.get_image_component().spawn_camera(
            f"{self.id}_RLCenterEye",
            UELocation(),
            UERotation(),
            False,
            solution,
            camera_id,
            True,
        )
        self.attach_cameras_to_sockets({
            camera_id: "InnerMidCameraSocket",  # "InnerMidCameraSocket" # "BodyHeadCameraSocket" # "MidCameraSocket"
        })
        self.get_image_component().set_camera_intrinsic_params(
            camera_id, float(fov), float(solution[0]), float(solution[1])
        )
        return camera_id

    def spawn_cameras(
        self, unidque_id="", camera_mode=0, agent_id="", with_shadow=False
    ):
        """
        为角色生成摄像机，并根据不同的模式将摄像机附加到指定的骨骼插槽上。

        :param unidque_id: 摄像机的唯一标识符
        :param camera_mode: 摄像机模式（0：双眼摄像机，1：三眼摄像机，2：四眼摄像机）
        :param agent_id: 智能体的标识符
        """
        if camera_mode == 0:
            self.__image_component.spawn_camera(
                unidque_id + "_LeftEye",
                UELocation(),
                UERotation(),
                False,
                [1280, 720],
                agent_id + "_LeftEye",
                with_shadow,
            )
            self.__image_component.spawn_camera(
                unidque_id + "_RightEye",
                UELocation(),
                UERotation(),
                False,
                [1280, 720],
                agent_id + "_RightEye",
                with_shadow,
            )
            self.attach_cameras_to_sockets({
                unidque_id + "_LeftEye": "LeftEyeCameraSocket",
                unidque_id + "_RightEye": "RightEyeCameraSocket",
            })
        elif camera_mode == 1:
            self.__image_component.spawn_camera(
                unidque_id + "_CenterEye",
                UELocation(),
                UERotation(),
                False,
                [768, 1280],
                agent_id + "_CenterEye",
                with_shadow,
            )
            self.__image_component.spawn_camera(
                unidque_id + "_LeftEye",
                UELocation(),
                UERotation(),
                False,
                [768, 1280],
                agent_id + "_LeftEye",
                with_shadow,
            )
            self.__image_component.spawn_camera(
                unidque_id + "_RightEye",
                UELocation(),
                UERotation(),
                False,
                [768, 1280],
                agent_id + "_RightEye",
                with_shadow,
            )
            self.attach_cameras_to_sockets({
                unidque_id + "_CenterEye": "MidCameraSocket",
                unidque_id + "_LeftEye": "LeftCameraSocket",
                unidque_id + "_RightEye": "RightCameraSocket",
            })
            self.__image_component.set_camera_intrinsic_params(
                unidque_id + "_CenterEye", 60.0, 768.0, 1280.0
            )
            self.__image_component.set_camera_intrinsic_params(
                unidque_id + "_LeftEye", 60.0, 768.0, 1280.0
            )
            self.__image_component.set_camera_intrinsic_params(
                unidque_id + "_RightEye", 60.0, 768.0, 1280.0
            )
        elif camera_mode == 2:
            self.__image_component.spawn_camera(
                unidque_id + "_CenterEye",
                UELocation(),
                UERotation(),
                False,
                [768, 1280],
                agent_id + "_CenterEye",
                with_shadow,
            )
            self.__image_component.spawn_camera(
                unidque_id + "_LeftEye",
                UELocation(),
                UERotation(),
                False,
                [768, 1280],
                agent_id + "_LeftEye",
                with_shadow,
            )
            self.__image_component.spawn_camera(
                unidque_id + "_RightEye",
                UELocation(),
                UERotation(),
                False,
                [768, 1280],
                agent_id + "_RightEye",
                with_shadow,
            )
            self.__image_component.spawn_camera(
                unidque_id + "_Camera_center",
                UELocation(),
                UERotation(),
                False,
                [768, 1280],
                agent_id + "_Camera_center",
                with_shadow,
            )
            self.attach_cameras_to_sockets({
                unidque_id + "_CenterEye": "MidCameraSocket",
                unidque_id + "_LeftEye": "LeftCameraSocket",
                unidque_id + "_RightEye": "RightCameraSocket",
                unidque_id + "_Camera_center": "LeftCameraSocket",
            })
            self.__image_component.set_camera_intrinsic_params(
                unidque_id + "_CenterEye", 60.0, 768.0, 1280.0
            )
            self.__image_component.set_camera_intrinsic_params(
                unidque_id + "_LeftEye", 60.0, 768.0, 1280.0
            )
            self.__image_component.set_camera_intrinsic_params(
                unidque_id + "_RightEye", 60.0, 768.0, 1280.0
            )
            self.__image_component.set_camera_intrinsic_params(
                unidque_id + "_Camera_center", 90.0, 1280.0, 768.0
            )
        elif camera_mode == 3:
            self.__image_component.spawn_camera(
                unidque_id + "center",
                UELocation(),
                UERotation(),
                False,
                [768, 1280],
                agent_id + "center",
                with_shadow,
            )
            self.__image_component.spawn_camera(
                unidque_id + "left",
                UELocation(),
                UERotation(),
                False,
                [768, 1280],
                agent_id + "left",
                with_shadow,
            )
            self.__image_component.spawn_camera(
                unidque_id + "right",
                UELocation(),
                UERotation(),
                False,
                [768, 1280],
                agent_id + "right",
                with_shadow,
            )
            self.__image_component.spawn_camera(
                unidque_id + "back",
                UELocation(),
                UERotation(),
                False,
                [768, 1280],
                agent_id + "back",
                with_shadow,
            )
            self.attach_cameras_to_sockets({
                unidque_id + "center": "MidCameraSocket",
                unidque_id + "left": "LeftCameraSocket",
                unidque_id + "right": "RightCameraSocket",
                unidque_id + "back": "LeftCameraSocket",
            })
            self.__image_component.set_camera_intrinsic_params(
                unidque_id + "center", 60.0, 768.0, 1280.0
            )
            self.__image_component.set_camera_intrinsic_params(
                unidque_id + "left", 60.0, 768.0, 1280.0
            )
            self.__image_component.set_camera_intrinsic_params(
                unidque_id + "right", 60.0, 768.0, 1280.0
            )
            self.__image_component.set_camera_intrinsic_params(
                unidque_id + "back", 90.0, 1280.0, 768.0
            )
        elif camera_mode == 4:
            self.__image_component.spawn_camera(
                unidque_id + "_CenterEye",
                UELocation(),
                UERotation(),
                False,
                [1024, 1024],
                agent_id + "_CenterEye",
                with_shadow,
            )
            self.__image_component.spawn_camera(
                unidque_id + "_Camera_center",
                UELocation(),
                UERotation(),
                False,
                [768, 1280],
                agent_id + "_Camera_center",
                with_shadow,
            )

            self.attach_cameras_to_sockets({
                unidque_id + "_CenterEye": "MidCameraSocket",
                unidque_id + "_Camera_center": "LeftCameraSocket",
            })
            self.__image_component.set_camera_intrinsic_params(
                unidque_id + "_CenterEye", 120.0, 1024.0, 1024.0
            )
            self.__image_component.set_camera_intrinsic_params(
                unidque_id + "_Camera_center", 90.0, 1280.0, 768.0
            )


    def attach_cameras_to_sockets(self, camid_socket_map):
        """
        将摄像机附加到指定的骨骼插槽上。

        :param camid_socket_map: 一个字典，键是摄像机 ID，值是骨骼插槽名称
        """
        self.__image_component.attach_cameras_to_sockets(camid_socket_map)

    def set_segmentation_id(self, seg_id):
        """
        设置角色的分割 ID。

        :param seg_id: 分割 ID
        :return: 设置操作的结果
        """
        return self.__segmentation_component.set_segmentation_id(seg_id)

    def get_segmentation_id(self):
        """
        获取角色的分割 ID。

        :return: 分割 ID
        """
        return self.__segmentation_component.get_segmentation_id()

    def get_images(self, image_requests):
        """
        获取指定摄像机的图像。

        :param image_requests: 图像请求对象列表
        :return: 图像数据
        """
        return self.__image_component.get_camera_image(image_requests)

    def get_image_component(self):
        """
        获取图像组件。

        :return: 图像组件
        """
        return self.__image_component

    def get_luminance(self, camera_name):
        """
        获取指定摄像机的亮度值。

        :param camera_name: 摄像机名称
        :return: 亮度值
        """
        return self.__image_component.get_camera_luminance(camera_name)

    def activate_ttlink(self):
        """
        激活 TTLink 功能。
        """
        self.__animation_component.activate_ttlink()

    def deactivate_ttlink(self):
        """
        取消激活 TTLink 功能。
        """
        self.__animation_component.deactivate_ttlink()

    def get_joint_names(self):
        """
        获取角色的所有关节名称。

        :return: 关节名称列表
        """
        return self.__animation_component.get_joint_names()

    def get_joint_global_locations(self):
        """
        获取角色所有关节的全局位置。

        :return: 关节的全局位置列表
        """
        return self.__animation_component.get_joint_global_locations()

    def get_joint_poses(self):
        """
        获取角色所有关节的姿势。

        :return: 关节姿势列表
        """
        return self.__animation_component.get_joint_poses()

    def get_joint_poses_world_space(self):
        """
        获取角色所有关节在世界空间中的姿势。

        :return: 关节在世界空间中的姿势列表
        """
        return self.__animation_component.get_joint_poses_world_space()

    def set_ttlink_pose(self, joint_name_list, joint_pose_list):
        """
        设置角色的 TTLink 姿势。

        :param joint_name_list: 关节名称列表
        :param joint_pose_list: 关节姿势列表
        """
        self.__animation_component.set_ttlink_pose(joint_name_list, joint_pose_list)

    def open_door(
        self,
        component_name,
        bIs_open,
        which_hand=basic_pb2.EWhichHandAction.RIGHT,
        hand_rotation=eulur2quaternion([60, 0, 0]),
        ignore_error_code = False
    ):
        """
        使用指定的手打开或关闭门。

        :param component_name: 门的组件名称
        :param bIs_open: 布尔值，True 表示打开门，False 表示关闭门
        :param whitch_hand: 使用的手 (EWhichHandAction)，默认是右手
        :param hand_rotation: 手的旋转 (UERotation)，默认是 (0, 0, 0, 0)
        """
        if isinstance(which_hand, WhichHand):
            which_hand = which_hand.value
        self.move_to_component(component_name, True)
        self.turn_around_to_component(component_name, True)
        if hand_rotation != UERotation(0, 0, 0):
            self.hand_rotator_proto(en.HandOutOrBack.HAND_OUT, eulur2quaternion([60, 0, 0]), 5)
            self.hand_rotator_proto(
                en.HandOutOrBack.HAND_OUT, hand_rotation, 1, which_hand
            )
        self.hand_reach_out_component(component_name, True, which_hand)
        self.switch_door_proto(bIs_open, component_name, which_hand, ignore_error_code=ignore_error_code)
        if hand_rotation != UERotation(0, 0, 0):
            self.hand_rotator_proto(
                en.HandOutOrBack.HAND_BACK, hand_rotation, 1, which_hand
            )
        self.hand_reach_back(which_hand, 0)

    def open_object_door(
        self, object: ObjectBase, index=0, which_hand=basic_pb2.EWhichHandAction.RIGHT
    ):
        """
        打开指定对象的门。

        :param object: 要打开的对象
        :param index: 门的索引，默认为 0
        :param which_hand: 使用的手 (EWhichHandAction)，默认是右手
        """
        doors = object.get_all_doors()
        if len(doors) == 0:
            # Todo:
            return
        if len(doors) <= index | len(doors) < 0:
            index = 0
        if isinstance(which_hand, WhichHand):
            which_hand = which_hand.value
        self.open_door(
            doors[index].component_guid, True, which_hand, eulur2quaternion([-20, 0, 0])
        )

    def close_object_door(
        self, object: ObjectBase, index=0, which_hand=basic_pb2.EWhichHandAction.RIGHT
    ):
        """
        关闭指定对象的门。

        :param object: 要关闭的对象
        :param index: 门的索引，默认为 0
        :param which_hand: 使用的手 (EWhichHandAction)，默认是右手
        """
        doors = object.get_all_doors()
        if len(doors) == 0:
            return
        if len(doors) <= index | len(doors) < 0:
            index = 0
        self.open_door(
            doors[index].component_guid,
            False,
            which_hand,
            eulur2quaternion([-20, 0, 0]),
        )

    def open_container_door(
        self, container_object: ObjectBase, which_hand=basic_pb2.EWhichHandAction.RIGHT
    ):
        """
        打开指定对象的门。

        :param container_object: 要打开的对象
        :param which_hand: 使用的手 (EWhichHandAction)，默认是右手
        """
        containers = container_object.get_all_container()
        if len(containers) > 0:
            container = containers[0]
            container_doors = container.get_container_doors()
            if len(container_doors) > 0:
                self.open_door(container_doors[0].id, True, which_hand)
                self.do_task()

    def close_container_door(
        self, container_object: ObjectBase, which_hand=basic_pb2.EWhichHandAction.RIGHT
    ):
        """
        关闭指定对象的门。

        :param container_object: 要关闭的对象
        :param which_hand: 使用的手 (EWhichHandAction)，默认是右手
        """
        containers = container_object.get_all_container()

        if len(containers) > 0:
            container = containers[0]
            container_doors = container.get_container_doors()
            if len(container_doors) > 0:
                self.open_door(container_doors[0].id, False, which_hand)
                self.do_task()

    """ Character Animation APIs """
    """ Character Move To """

    def move_to_location(self, location, move_speed=None, is_can_run=None, accelerate_delay_time=None, max_speed=None, execution_type=None):
        """
        移动角色到指定的位置。

        :param location: 目标位置 (UELocation)
        :param animation_command_id: 动画命令 ID（可选）
        :return: 如果使用 gRPC，则返回移动操作的结果；否则无返回值
        """
        if self.__animation_stub is not None:
            return self.move_to_location_proto(location=location, move_speed=move_speed, is_can_run=is_can_run, accelerate_delay_time=accelerate_delay_time, max_speed=max_speed, execution_type=execution_type)
        self.__animation_component.move_to_location(location)

    def move_to_object(self, object, b_use_socket=False):
        """
        移动角色到指定的对象。

        :param object: 目标对象
        :param b_use_socket: 布尔值，是否使用骨骼插槽，默认为 False
        :return: 如果使用 gRPC，则返回移动操作的结果；否则无返回值
        """
        if self.__animation_stub is not None:
            return self.move_to_object_proto(object, b_use_socket)
        self.__animation_component.move_to_object(object, b_use_socket)

    """角色转向相关方法"""

    def turn_around_to_location(self, location):
        """
        使角色朝向指定的位置。

        :param location: 目标位置 (UELocation)
        :return: 如果使用 gRPC，则返回转向操作的结果；否则无返回值
        """
        if self.__animation_stub is not None:
            return self.turn_around_to_location_proto(location=location)
        self.__animation_component.turn_around_to_location(location)

    def turn_around_to_camera(self):
        """
        使角色朝向摄像机方向。

        :return: 动画命令 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_TurnAroundTowards
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def turn_around_to_object(self, object, b_use_socket=False):
        """
        使角色朝向指定的对象。

        :param object: 目标对象
        :param b_use_socket: 布尔值，是否使用骨骼插槽，默认为 False
        :return: 如果使用 gRPC，则返回转向操作的结果；否则无返回值
        """
        if self.__animation_stub is not None:
            return self.turn_around_to_object_proto(object, b_use_socket)
        self.__animation_component.turn_around_to_object(object, b_use_socket)

    def seek_object(self, object=object, seek_point=""):
        """
        角色打量查看某个物体。

        :param object: 目标对象
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_Check
        request.seek_check.object_id = object.id
        request.seek_check.check_point = seek_point
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def seek_location(self, location: UELocation):
        """
        角色打量查看某个位置。

        :param object: 目标对象
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_Check
        request.seek_check.location.x = location.X
        request.seek_check.location.y = location.Y
        request.seek_check.location.z = location.Z
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def turn_around_to_soket(self, socket, actor_name, component_name):
        """
        使角色朝向指定的插槽。

        :param socket: 插槽名称
        :param actor_name: 目标角色名称
        :param component_name: 组件名称
        """
        self.__animation_component.turn_around_to_soket(
            socket, actor_name, component_name
        )

    def turn_around_to_degree(self, degree):
        """
        使角色朝向指定的角度。

        :param degree: 目标角度
        """
        self.__animation_component.turn_around_to_degree(degree)

    """角色运动状态机相关方法"""

    def sit_down(self):
        """
        使角色坐下。

        :return: 如果使用 gRPC，则返回操作结果；否则无返回值
        """
        if self.__animation_stub is not None:
            return self.sit_down_proto()
        self.__animation_component.sit_down()

    def lie_down(self):
        """
        使角色躺下。

        :return: 如果使用 gRPC，则返回操作结果；否则无返回值
        """
        if self.__animation_stub is not None:
            return self.lie_down_proto()
        self.__animation_component.lie_down()

    def stand_up(self):
        """
        使角色站起来。

        :return: 如果使用 gRPC，则返回操作结果；否则无返回值
        """
        if self.__animation_stub is not None:
            return self.stand_up_proto()
        self.__animation_component.stand_up()

    """ Character Gesture State Machine """

    def wave_hand(self, which_hand=en.WhichHand.RIGHT_HAND):
        """
        使角色挥手。

        :return: 如果使用 gRPC，则返回操作结果；否则无返回值
        """
        if self.__animation_stub is not None:
            return self.wave_hand_proto(which_hand)
        self.__animation_component.wave_hand()

    def raise_hand(self, which_hand=en.WhichHand.RIGHT_HAND):
        """
        使角色举手。

        :return: 如果使用 gRPC，则返回操作结果；否则无返回值
        """
        if self.__animation_stub is not None:
            return self.raise_hand_proto(which_hand)
        self.__animation_component.raise_hand()

    def romp_play_object(self, which_hand=en.WhichHand.RIGHT_HAND, decrease_boredom=0):
        """
        使角色与对象进行嬉戏交互，减少无聊值。

        :param decrease_boredom: 无聊值减少的量
        :param which_hand: 使用的手 (EHandAction)，默认为右手
        :return: 动画命令 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_GestureState
        request.gesture.gesture_status = gesture_pb2.EGestureEnum.ROMPPLAYOBJECT
        request.animation_effect_attribute.decrease_boredom = decrease_boredom
        if which_hand == en.WhichHand.TWO_HANDS:
            raise "currently only one hand support "
        request.gesture.which_hand_action = which_hand.value
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def nod_head(self):
        """
        使角色点头。

        :return: 如果使用 gRPC，则返回操作结果；否则无返回值
        """
        if self.__animation_stub is not None:
            return self.nod_head_proto()
        self.__animation_component.nod_head()

    def shake_head(self):
        """
        使角色摇头。

        :return: 如果使用 gRPC，则返回操作结果；否则无返回值
        """
        if self.__animation_stub is not None:
            return self.shake_head_proto()
        self.__animation_component.shake_head()

    def chat(self):
        """
        使角色进行聊天动作。

        :return: 如果使用 gRPC，则返回操作结果；否则无返回值
        """
        if self.__animation_stub is not None:
            return self.chat_proto()
        self.__animation_component.chat()

    def idle_gesture(self):
        """
        使角色执行空闲状态下的手势。

        :return: 如果使用 gRPC，则返回操作结果；否则无返回值
        """
        if self.__animation_stub is not None:
            return self.idle_gesture_proto()
        self.__animation_component.idle_gesture()

    def wipe_dirt(self, dirt):
        """
        使角色执行擦拭污渍的动作。

        :param dirt: 需要擦拭的污渍对象
        :return: 如果使用 gRPC，则返回操作结果；否则无返回值
        """
        if self.__animation_stub is not None:
            return self.wipe_dirt_proto(dirt=dirt)
        self.__animation_component.wipe_dirt(dirt)

    def eat_or_drink(self):
        """
        使角色执行吃或喝的动作。

        :return: 如果使用 gRPC，则返回操作结果；否则无返回值
        """
        if self.__animation_stub is not None:
            return self.eat_or_drink_proto()
        self.__animation_component.eat_or_drink()

    """ 拿取和放下物体的动画 API """

    def take_object(
        self,
        object,
        b_auto_rotate=True,
        rotation=UERotation(0, 0, 0, 1),
        which_hand=0,
        TargetSocket=False,
        container_name="",
        ignore_error_code = False
    ):
        """
        使角色拿起指定的对象。

        :param object: 要拿取的对象
        :param b_auto_rotate: 是否自动旋转，默认为 True
        :param rotation: 旋转参数 (UERotation)
        :param which_hand: 使用的手 (int)，默认为 0（右手）
        :param TargetSocket: 是否使用插槽，默认为 False
        :param container_name: 容器名称（可选）
        :return: 如果使用 gRPC，则返回操作结果；否则无返回值
        """
        which_hand_enum = Character.get_which_hand_action_enum(which_hand)
        if self.__animation_stub is not None:
            return self.take_object_proto(
                Object=object,
                TargetSocket=TargetSocket,
                container_name=container_name,
                WhichHand=which_hand_enum,
                ignore_error_code=ignore_error_code
            )
        self.hand_reach_out_object(which_hand=which_hand, object=object)
        self.hand_grab_object(which_hand=which_hand, object=object)
        self.hand_reach_back(which_hand)

    def put_down_sth_to_location(
        self,
        location,
        b_auto_rotate=False,
        rotation=UERotation(0, 0, 0, 1),
        b_force_locate=True,
        which_hand=0,
        b_force_release=False,
        b_disable_physics=False,
        hand_offset=UELocation(),
    ):
        """
        将角色手中的物体放置到指定位置。

        :param location: 目标位置 (UELocation)
        :param b_auto_rotate: 是否自动旋转物体，默认为 False
        :param rotation: 物体的旋转角度 (UERotation)，默认为 (0, 0, 0, 1)
        :param b_force_locate: 是否强制定位物体，默认为 True
        :param which_hand: 使用的手 (int)，默认为 0（右手）
        :param b_force_release: 是否强制释放物体，默认为 False
        :param b_disable_physics: 是否禁用物理效果，默认为 False
        :param hand_offset: 手的偏移量 (UELocation)
        :return: 如果使用 gRPC，则返回操作结果；否则无返回值
        """
        if self.__animation_stub is not None:
            which_hand_enum = Character.get_which_hand_action_enum(which_hand)
            return self.put_down_sth_to_location_proto(
                location=location,
                rotation=rotation,
                b_auto_rotate=b_auto_rotate,
                b_force_locate=b_force_locate,
                which_hand=which_hand_enum,
                b_force_release=b_force_release,
                b_disable_physics=b_disable_physics,
                hand_offset=hand_offset,
            )
        self.hand_reach_out_location(which_hand, location)
        self.hand_release(
            which_hand,
            location,
            b_force_locate,
            b_auto_rotate,
            rotation,
            b_force_release,
        )
        self.hand_reach_back(which_hand)

    def move_and_put_down_object_in_container(
        self, container_component:ContainerComponent, obj: BaseObject, which_hand=0, force_release=True
    ):
        """
        移动角色并将手中的物体放入容器中。

        :param container_component: 容器组件
        :param obj: 要放入容器的物体
        :param which_hand: 使用的手 (int)，默认为 0（右手）
        :param force_release: 是否强制释放物体，默认为 True
        :return: 布尔值，表示放置操作是否成功
        """
        container_info = self.get_placed_object_info_from_container(
            obj, container_component.component_guid
        )

        move_location = obj.get_pose().location
        place_location = obj.get_pose().location

        if container_info.b_success:
            move_location = UELocation(
                container_info.move_location.x,
                container_info.move_location.y,
                container_info.move_location.z,
            )
            place_location = UELocation(
                container_info.place_location.x,
                container_info.place_location.y,
                container_info.place_location.z,
            )

        self.move_to_location(move_location)

        self.turn_around_to_location(place_location)
        self.hand_reach_out_location(which_hand, place_location)
        self.hand_release(
            which_hand,
            place_location,
            True,
            True,
            UERotation(0, 0, 0, 1),
            force_release,
            container_component.component_guid,
            b_disable_physics=False,
        )
        self.hand_reach_back(which_hand)
        return container_info.b_success

    def move_and_take_out_object_from_container(
        self, container_component:ContainerComponent, object, which_hand=0, b_force_release=True
    ):
        """
        移动角色并从容器中取出物体。

        :param container_component: 容器组件
        :param object: 要取出的物体
        :param which_hand: 使用的手 (int)，默认为 0（右手）
        :param b_force_release: 是否强制释放物体，默认为 True
        :return: 如果取出成功，执行相关操作；否则打印失败信息
        """
        container_info = self.get_take_out_object_info_from_container(
            object.id, container_component.component_guid
        )
        move_location = UELocation(
            container_info.move_location.x,
            container_info.move_location.y,
            container_info.move_location.z,
        )
        if container_info.b_success:
            self.move_to_location(move_location)
            self.turn_around_to_object(object)
            self.hand_reach_out_object(which_hand=which_hand, object=object)
            self.hand_grab_object(
                which_hand, object, container_component.component_guid
            )
            self.hand_reach_back(which_hand)
        else:
            print("fail grab object from container!")

    """ Todo: To Enable Proto Method"""

    def switch_door(self, door, b_is_open, component_name, which_hand):
        """
        切换门的状态（打开或关闭）。

        :param door: 要操作的门对象
        :param b_is_open: 布尔值，True 表示打开门，False 表示关闭门
        :param component_name: 门的组件名称
        :param which_hand: 使用的手 (EWhichHandAction)
        """
        self.__animation_component.switch_door(
            door, b_is_open, component_name, which_hand
        )

    def switch_door_new(self, door, b_is_open, component_name, which_hand):
        """
        切换门的状态（新方法，使用门对象的 ID）。

        :param door: 要操作的门对象，使用其 ID
        :param b_is_open: 布尔值，True 表示打开门，False 表示关闭门
        :param component_name: 门的组件名称
        :param which_hand: 使用的手 (EWhichHandAction)
        """
        self.__animation_component.switch_door(
            door.id, b_is_open, component_name, which_hand
        )

    """ 瞄准偏移更新动画 """

    def point_at(self, location, which_hand=WhichHand.RIGHT_HAND):
        """
        使角色指向指定位置。

        :param location: 目标位置 (UELocation)
        :return: 如果使用 gRPC，则返回操作结果；否则无返回值
        """
        if self.__animation_stub is not None:
            return self.point_at_proto(location, which_hand)
        self.__animation_component.point_at(location)

    def cancel_point_at(self):
        """
        取消角色指向操作。

        :return: 如果使用 gRPC，则返回操作结果；否则无返回值
        """
        if self.__animation_stub is not None:
            return self.cancel_point_at_proto()
        self.__animation_component.cancel_point_at()

    def normalize_vector(self,location=UELocation()):

        length = math.sqrt(location.X * location.X + location.Y * location.Y + location.Z * location.Z)
        normalize_vec = UELocation()
        if length != 0:
            normalize_vec.X = location.X / length
            normalize_vec.Y = location.Y / length
            normalize_vec.Z = location.Z / length
        return normalize_vec

    def gaze_at(self, location=UELocation(), target=None):
        """
        使角色注视指定位置，并返回当前朝向与目标的夹角

        :param location: 目标位置 (UELocation)
        :return: 夹角（0-360度）及操作结果
        """
        character_location = self.get_pose().location
        ue_direction = location-character_location
        ue_direction_normalized = self.normalize_vector(location=ue_direction)
        ue_forward_vector = self.get_forward_vector()
        forward_vector = np.array([ue_forward_vector.X, ue_forward_vector.Y, ue_forward_vector.Z])
        direction_normalized=np.array([ue_direction_normalized.X, ue_direction_normalized.Y, ue_direction_normalized.Z])
        dot_product = np.dot(forward_vector, direction_normalized)
        base_angle = np.degrees(np.arccos(dot_product))
        cross_product = np.cross(forward_vector,direction_normalized)
        final_angle = 360 - base_angle if cross_product[1] < 0 else base_angle
        if final_angle > 85 or final_angle < -85:
            self.turn_around_to_degree(final_angle)
        if self.__animation_stub is not None:
            return final_angle, self.gaze_at_proto(location, target)
        else:
            self.__animation_component.gaze_at(location)
            return final_angle, None

    def cancel_gaze_at(self):
        """
        取消角色注视操作。

        :return: 如果使用 gRPC，则返回操作结果；否则无返回值
        """
        if self.__animation_stub is not None:
            return self.cancel_gaze_at_proto()
        self.__animation_component.cancel_gaze_at()

    """ 角色时间占位符 """

    def wait_for_time(self, time_length):
        """
        使角色等待指定时间。

        :param time_length: 等待时间长度（秒）
        :return: 如果使用 gRPC，则返回操作结果；否则无返回值
        """
        if self.__animation_stub is not None:
            return self.wait_for_time_proto(TimeDuration=time_length)
        self.__animation_component.wait_for_time(time_length * 1.0)

    def decrease_boredom(self, decrease_boredom):
        """
        使角色减少无聊值。

        :param decrease_boredom: 无聊值减少的量
        :return: 如果使用 gRPC，则返回操作结果；否则无返回值
        """
        if self.__animation_stub is not None:
            return self.wait_for_time_proto(
                decrease_boredom=decrease_boredom, TimeDuration=0
            )
        self.__animation_component.wait_for_time(0 * 1.0)

    def hsi_animation(
        self,
        request_desc: str,
        target_location: UELocation,
        hand_target_location: UELocation,
        seg_num: int,
    ):
        """
        使角色采用HSI动作算法完成指定动作任务。

        :param request_desc: HSI 模型的prompt string
        :param target_location: 智能体要移动到的位置
        :param hand_target_location: 智能体手部要到达的位置
        :param seg_num: 期望HSI模型生成的动作片段数量
        :return: 如果使用 gRPC，则返回操作结果；否则无返回值
        """

        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_HsiAnimation
        request.hsi_anim.request_desc = request_desc
        request.hsi_anim.target_location.x = target_location.X
        request.hsi_anim.target_location.y = target_location.Y
        request.hsi_anim.target_location.z = target_location.Z
        request.hsi_anim.hand_target_location.x = hand_target_location.X
        request.hsi_anim.hand_target_location.y = hand_target_location.Y
        request.hsi_anim.hand_target_location.z = hand_target_location.Z
        request.hsi_anim.seg_num = seg_num
        self.__animation_stub.EnqueAnimationCommand(request)
        pass

    def hsi_take_object_proto(
        self,
        object: BaseObject,
        target_location: UELocation,
        hand_target_location: UELocation,
        request_desc: str,
        seg_num: int,
        which_hand=basic_pb2.RIGHT,
        b_use_socket=True,
    ):
        """
        使角色采用HSI动作算法完成抓取物体的操作。

        :object: 期望角色抓取的物体
        :param request_desc: HSI 模型的prompt string
        :param target_location: 智能体要移动到的位置
        :param hand_target_location: 智能体手部要到达的位置
        :param seg_num: 期望HSI模型生成的动作片段数量
        :which_hand: 指定角色使用那只手抓取物体
        :b_use_socket：是否按照物体的socket位置完成抓取动作
        :return: 如果使用 gRPC，则返回操作结果；否则无返回值
        """

        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_HsiAnimation

        request.hsi_anim.request_desc = request_desc
        request.hsi_anim.target_location.x = target_location.X
        request.hsi_anim.target_location.y = target_location.Y
        request.hsi_anim.target_location.z = target_location.Z
        request.hsi_anim.hand_target_location.x = hand_target_location.X
        request.hsi_anim.hand_target_location.y = hand_target_location.Y
        request.hsi_anim.hand_target_location.z = hand_target_location.Z
        request.hsi_anim.seg_num = seg_num

        request.hsi_anim.hand_grab_object.which_hand_action = which_hand
        request.hsi_anim.hand_grab_object.object_params.b_use_socket = b_use_socket
        request.hsi_anim.hand_grab_object.object_params.target.id = object.id

        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def hsi_put_down_sth_to_location(
        self,
        target_location: UELocation,
        hand_target_location: UELocation,
        request_desc: str,
        seg_num: int,
        which_hand=basic_pb2.RIGHT,
    ):
        """
        使角色采用HSI动作算法完成放置物体的操作。

        :param request_desc: HSI 模型的prompt string
        :param target_location: 智能体要移动到的位置
        :param hand_target_location: 智能体手部要到达的位置
        :param seg_num: 期望HSI模型生成的动作片段数量
        :which_hand: 指定角色使用那只手抓取物体
        :return: 如果使用 gRPC，则返回操作结果；否则无返回值
        """

        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_HsiAnimation

        request.hsi_anim.request_desc = request_desc
        request.hsi_anim.target_location.x = target_location.X
        request.hsi_anim.target_location.y = target_location.Y
        request.hsi_anim.target_location.z = target_location.Z
        request.hsi_anim.hand_target_location.x = hand_target_location.X
        request.hsi_anim.hand_target_location.y = hand_target_location.Y
        request.hsi_anim.hand_target_location.z = hand_target_location.Z
        request.hsi_anim.seg_num = seg_num

        request.hsi_anim.hand_release.which_hand_action = which_hand
        request.hsi_anim.hand_release.target_location.x = hand_target_location.X
        request.hsi_anim.hand_release.target_location.y = hand_target_location.Y
        request.hsi_anim.hand_release.target_location.z = hand_target_location.Z

        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def do_task(self):
        """
        执行角色的任务。

        :return: 如果使用 gRPC，则返回操作结果；否则无返回值
        """
        if self.__animation_stub is not None:
            return self.do_task_proto()
        self.__animation_component.start_animation()

    def call_cancel_task(self, action_id=0):
        """
        取消任务
        """
        request = animation_pb2.AnimationActionIDRequest()
        request.action_id = action_id
        request.component.id = self.__animation_component.component_guid
        self.__animation_stub.CancelTask(request)

    def cancel_task(self, action_id=0):
        """
        取消任务
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_Cancel
        request.cancel.action_id = action_id
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def do_task_and_wait_finish(self):
        """
        执行角色的任务并等待完成。

        :return: 如果使用 gRPC，则返回操作结果；否则无返回值
        """
        time.sleep(0.1)
        if self.__animation_stub is not None:
            self.do_task_proto()
        else:
            self.__animation_component.start_animation()

        while not self.is_all_task_finished():
            time.sleep(0.1)

    def is_all_task_finished(self):
        """
        检查角色的所有任务是否完成。

        :return: 布尔值，True 表示所有任务已完成
        """
        return self.__animation_component.is_all_task_finished()

    """ 角色复合动画 """

    def go_to_take_object(
        self, object, b_auto_rotate=True, rotation=UERotation(0, 0, 0, 1)
    ):
        """
        移动到指定对象并拿起它。

        :param object: 要拿取的对象
        :param b_auto_rotate: 是否自动旋转对象，默认为 True
        :param rotation: 对象的旋转角度 (UERotation)，默认为 (0, 0, 0, 1)
        """
        self.move_to_object(object)
        self.move_to_object(object)
        self.turn_around_to_object(object)
        self.take_object(object, b_auto_rotate, rotation)

    """ 高级移动命令 """

    def go_to_location_take_object(
        self, object, goto_location, b_auto_rotate=True, rotation=UERotation(0, 0, 0, 1)
    ):
        """
        移动到指定位置并拿起对象。

        :param object: 要拿取的对象
        :param goto_location: 目标位置 (UELocation)
        :param b_auto_rotate: 是否自动旋转对象，默认为 True
        :param rotation: 对象的旋转角度 (UERotation)，默认为 (0, 0, 0, 1)
        """
        self.move_to_location(goto_location)
        self.move_to_location(goto_location)
        self.turn_around_to_object(object)
        self.take_object(object, b_auto_rotate, rotation)

    def go_to_eat_or_drink_object(self, object):
        """
        移动到指定对象并进行吃或喝的动作。

        :param object: 要吃或喝的对象
        """
        self.move_to_object(object)
        self.turn_around_to_object(object)
        self.take_object(object)
        self.eat_or_drink()

    """ 角色动画 API 结束 """

    def get_object_in_hand(self):
        """
        获取角色当前手中的对象 ID。

        :return: 当前手中的对象 ID
        """
        return self.__attachment_component.get_object_id_in_hand()

    def get_object_offset_in_hand(self, which_hand=0):
        """
        获取角色当前手中物体的抓握便宜

        :return: 当手距离下表面的距离
        """
        which_hand_enum = Character.get_which_hand_action_enum(which_hand)
        request = animation_pb2.HandGrabOffsetHeightRequest()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.which_hand_action = which_hand_enum
        response = self.__animation_stub.GetHandGrabObjectOffsetHeight(request)
        return response.float

    def get_capsule(self):
        """
        获取角色的胶囊组件信息。

        :return: 胶囊组件信息
        """
        return self.__character_capsule_component.get_capsule()

    def get_capsule_half_height(self):
        """
        获取角色胶囊组件的半高。

        :return: 胶囊组件的半高
        """
        return self.__character_capsule_component.get_capsule_half_height()

    def get_capsule_radius(self):
        """
        获取角色胶囊组件的半径。

        :return: 胶囊组件的半径
        """
        return self.__character_capsule_component.get_capsule_radius()

    def get_hungry_value(self):
        """
        获取角色当前的饥饿值。

        :return: 饥饿值
        """
        return self.__character_energy_component.get_character_energy()[0]

    def get_thirsty_value(self):
        """
        获取角色当前的口渴值。

        :return: 口渴值
        """
        return self.__character_energy_component.get_character_energy()[1]

    def get_exhausting_value(self):
        """
        获取角色当前的疲劳值。

        :return: 疲劳值
        """
        return self.__character_energy_component.get_character_energy()[2]

    def get_character_energy(self):
        """
        获取角色的所有能量值（饥饿值、口渴值、疲劳值）。

        :return: 角色的能量值列表
        """
        return self.__character_energy_component.get_character_energy()

    # def SetHungryValue(self, hungry_value):
    #     value = self.__character_energy_component.GetCharacterEnergy()
    #     value[0] = hungry_value
    #     self.__character_energy_component.SetCharacterEnergy(value[0], value[1], value[2])
    #     pass
    # def SetThirstyValue(self, thirsty_value):
    #     value = self.__character_energy_component.GetCharacterEnergy()
    #     value[1] = thirsty_value
    #     self.__character_energy_component.SetCharacterEnergy(value[0], value[1], value[2])
    #     pass
    # def SetExhaustingValue(self, exhaust_value):
    #     value = self.__character_energy_component.GetCharacterEnergy()
    #     value[2] = exhaust_value
    #     self.__character_energy_component.SetCharacterEnergy(value[0], value[1], value[2])
    #     pass

    def set_character_energy(self, hungry, thirsty, stamina):
        """
        设置角色的能量值，包括饥饿值、口渴值和耐力值。

        :param hungry: 饥饿值
        :param thirsty: 口渴值
        :param stamina: 耐力值
        """
        self.__character_energy_component.set_character_energy(
            hungry * 1.0, thirsty * 1.0, stamina * 1.0
        )

    def set_character_energy_proto(self, hungry, thirsty, stamina, sleepy):
        """
        使用 gRPC 设置角色的能量值，包括饥饿值、口渴值、耐力值和困倦值。

        :param hungry: 饥饿值
        :param thirsty: 口渴值
        :param stamina: 耐力值
        :param sleepy: 困倦值
        :return: 设置后的饥饿值、口渴值、耐力值和困倦值
        """
        from tongsim_api_protocol.component import character_energy_pb2_grpc
        from tongsim_api_protocol.component import character_energy_pb2

        Stub = character_energy_pb2_grpc.CharacterEnergyServiceStub(
            self.insecure_channel
        )
        request = character_energy_pb2.CharacterEnergyInfoRequest()
        request.component.id = self.__character_energy_component.component_guid
        request.energy.hungry = hungry
        request.energy.thirsty = thirsty
        request.energy.stamina = stamina
        response = Stub.SetCharacterEnergy(request)
        return response.hungry, response.thirsty, response.stamina, response.sleepy

    def set_sleepy_value(self, value):
        """
        设置角色的困倦值。

        :param Value: 困倦值
        """
        return self.__attribute_component.set_sleepy_value(value)

    def set_boredom_value(self, value):
        """
        设置角色的无聊值。

        :param Value: 无聊值
        """
        return self.__attribute_component.set_boredom_value(value)

    def set_max_walk_distance(self, MaxWalkDistance):
        """
        设置角色的最大行走距离。

        :param MaxWalkDistance: 最大行走距离
        """
        self.__character_energy_component.set_max_walk_distance(MaxWalkDistance * 1.0)

    def get_max_walk_distance(self):
        """
        获取角色的最大行走距离。

        :return: 最大行走距离
        """
        return self.__character_energy_component.get_max_walk_distance()

    def get_playable_anim_sequences(self):
        """
        获取角色可播放的动画序列。

        :return: 可播放的动画序列列表
        """
        return self.__animation_component.get_playable_anim_sequences()

    def play_anim_sequence(self, anim_sequence_name, decrease_boredom=0, b_async = False, key = "", target_id="", anin_value =0.0):
        """
        播放指定的动画序列。

        :param anim_sequence_name: 动画序列名称
        :param decrease_boredom: 减少无聊值的量，默认为 0
        """
        if self.__animation_stub is not None:
            return self.play_anim_sequence_proto(anim_sequence_name, decrease_boredom, b_async, key, target_id, anin_value)
        self.__animation_component.play_anim_sequence(anim_sequence_name)

    def grab_object(self, target="", rate = 1.0):
        """
        播放抢夺动作
        """
        self.play_anim_sequence("AM_Grab_Toy", key="pickup", target_id=target.id, anin_value = 1.0 - rate)

    def wipe_quadrilateral(
        self,
        conner1: UELocation,
        conner2: UELocation,
        conner3: UELocation,
        conner4: UELocation,
        which_hand=0,
    ):
        """
        使角色擦拭一个四边形区域。

        :param conner1: 四边形的第一个角点 (UELocation)
        :param conner2: 四边形的第二个角点 (UELocation)
        :param conner3: 四边形的第三个角点 (UELocation)
        :param conner4: 四边形的第四个角点 (UELocation)
        :param which_hand: 使用的手 (int)，默认为 0（右手）
        """
        which_hand_enum = Character.get_which_hand_action_enum(which_hand)
        if self.__animation_stub is not None:
            return self.wipe_quadrilateral_proto(
                conner1=conner1,
                conner2=conner2,
                conner3=conner3,
                conner4=conner4,
                WhichHand=which_hand_enum,
            )
        self.__animation_component.wipe_quadrilateral(
            conner1=conner1,
            conner2=conner2,
            conner3=conner3,
            conner4=conner4,
            which_hand=which_hand,
        )

    def free_animation(self, action: str):
        """
        触发一个自由动画动作。

        :param action: 动作的字符串标识
        """
        if self.__animation_stub is not None:
            return self.free_animation_proto(action=action)
        self.__animation_component.free_animation(action)

    def free_animation_proto(self, action: str):
        """
        通过 gRPC 触发一个自由动画动作。

        :param action: 动作的字符串标识
        :return: 动作 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_FreeAnimation
        request.free_idle_action.action_name = action
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    """ 待修复的 Proto 方法 """

    def play_facial_animation_with_sound_file(
        self, music_path, frequency, morph_target_num, frame_num, weights
    ):
        """
        播放带有音频文件的面部动画。

        :param music_path: 音频文件路径
        :param frequency: 音频频率
        :param morph_target_num: 变形目标的数量
        :param frame_num: 帧数
        :param weights: 权重
        """
        self.__facialanimation_component.play_facial_animation_with_sound_file(
            music_path, frequency, morph_target_num, frame_num, weights
        )

    """ 待修复的 Proto 方法 """

    def play_facial_animation_with_sound_data(
        self, music_data, frequency, morph_target_num, frame_num, weights
    ):
        """
        播放带有音频数据的面部动画。

        :param music_data: 音频数据
        :param frequency: 音频频率
        :param morph_target_num: 变形目标的数量
        :param frame_num: 帧数
        :param weights: 权重
        """
        self.__facialanimation_component.play_facial_animation_with_sound_data(
            music_data, frequency, morph_target_num, frame_num, weights
        )

    def query_nav_path(self, target_location):
        """
        查询到达目标位置的导航路径。

        :param target_location: 目标位置 (UELocation)
        :return: 导航路径点列表
        """
        return self.__animation_component.query_nav_path_point(target_location)

    def hand_reach_out_location(self, which_hand, location, hand_reach_from=0):
        """
        使角色的手伸向指定位置。

        :param which_hand: 使用的手 (int)
        :param location: 目标位置 (UELocation)
        :param hand_reach_from: 伸手的方向，0 表示从上方，1 表示从正前方
        :return: 如果使用 gRPC，则返回操作结果；否则无返回值
        """
        which_hand_enum = Character.get_which_hand_action_enum(which_hand)

        HandReachFrom: basic_pb2.EHandReachFrom
        if hand_reach_from == 0:
            HandReachFrom = basic_pb2.EHandReachFrom.ABOVE
        else:
            HandReachFrom = basic_pb2.EHandReachFrom.FRONT

        try:
            return self.finger_reach_out_location_proto(
                Location=location,
                WhichHand=which_hand_enum,
                HandReachFrom=HandReachFrom,
            )
        except:
            print("Proto Server Not Connect")

        self.__animation_component.finger_reach_out_location(
            which_hand, location, hand_reach_from=hand_reach_from
        )

    def hand_reach_out_object(
        self,
        which_hand,
        object,
        b_use_socket=False,
        hand_reach_from=en.HandReachFrom.ABOVE,
        ignore_error_code= False
    ):
        """使角色的手伸向指定对象。

        :param which_hand: 使用的手 (int)
        :param object: 目标对象
        :param b_use_socket: 是否使用 socket，默认为 False
        :param hand_reach_from: 伸手的方向，0 表示从上方，1 表示从正前方
        :return: 如果使用 gRPC，则返回操作结果；否则无返回值
        """
        which_hand_enum = Character.get_which_hand_action_enum(which_hand)
        if self.__animation_stub is not None:
            HandReachFrom: basic_pb2.EHandReachFrom
            if hand_reach_from == 0:
                HandReachFrom = basic_pb2.EHandReachFrom.ABOVE
            else:
                HandReachFrom = basic_pb2.EHandReachFrom.FRONT

            return self.hand_reach_out_object_proto(
                Object=object,
                WhichHand=which_hand_enum,
                TargetSocket=b_use_socket,
                HandReachFrom=HandReachFrom,
                ignore_error_code = ignore_error_code
            )
        self.__animation_component.hand_reach_out_object(
            which_hand=which_hand,
            object=object,
            hand_reach_from=hand_reach_from,
            b_use_socket=b_use_socket,
        )

    def hand_reach_out_soket(
        self, which_hand, soket, actor_name, component_name, hand_reach_from=0
    ):
        """
        使角色的手伸向指定的 socket。

        :param which_hand: 使用的手 (int)
        :param soket: 目标 socket 名称
        :param actor_name: 目标 actor 名称
        :param component_name: 组件名称
        :param hand_reach_from: 伸手的方向，0 表示从上方，1 表示从正前方
        """
        self.__animation_component.hand_reach_out_soket(
            which_hand,
            soket,
            actor_name,
            component_name,
            hand_reach_from=hand_reach_from,
        )

    def hand_grab_object(
        self,
        which_hand,
        object,
        container_unique_name="",
        bis_use_socket=False,
        component_name="",
    ):
        """
        使角色的手抓住指定的对象。

        :param which_hand: 使用的手 (int)
        :param object: 要抓取的对象
        :param container_unique_name: 容器的唯一名称，默认为空字符串
        :param bis_use_socket: 是否使用 socket，默认为 False
        :param component_name: 组件名称，默认为空字符串
        :return: 如果使用 gRPC，则返回操作结果；否则无返回值
        """
        which_hand_enum = Character.get_which_hand_action_enum(which_hand)
        if self.__animation_stub is not None:
            return self.hand_grab_object_proto(
                Object=object,
                WhichHand=which_hand_enum,
                container_unique_name=container_unique_name,
                UseSocket=bis_use_socket,
            )
        self.__animation_component.hand_grab_object(
            which_hand, object, bis_use_socket, container_unique_name
        )

    def hand_release(
        self,
        which_hand,
        desired_location,
        b_force_locate=False,
        b_auto_rotate=False,
        rotation=UERotation(0, 0, 0, 1),
        b_force_release=False,
        container_component_unique_name="",
        b_disable_physics=False,
    ):
        """
        使角色的手释放手中的对象或位置。

        :param which_hand: 使用的手 (int)
        :param desired_location: 释放的位置 (UELocation)
        :param b_force_locate: 是否强制定位，默认为 False
        :param b_auto_rotate: 是否自动旋转对象，默认为 False
        :param rotation: 旋转角度 (UERotation)
        :param b_force_release: 是否强制释放，默认为 False
        :param container_component_unique_name: 容器组件的唯一名称，默认为空字符串
        :param b_disable_physics: 是否禁用物理效果，默认为 False
        :return: 如果使用 gRPC，则返回操作结果；否则无返回值
        """
        which_hand_enum = Character.get_which_hand_action_enum(which_hand)
        if self.__animation_stub is not None:
            return self.hand_release_proto(
                which_hand=which_hand_enum,
                desired_location=desired_location,
                b_force_locate=b_force_locate,
                b_auto_rotate=b_auto_rotate,
                b_force_release=b_force_release,
                rotation=rotation,
                container_uniqueName=container_component_unique_name,
                b_disable_physics=b_disable_physics,
            )
        self.__animation_component.hand_release(
            which_hand,
            desired_location,
            b_force_locate,
            b_auto_rotate,
            rotation,
            b_force_release,
        )

    def hand_reach_back(self, which_hand, hand_reach_from=0):
        """
        使角色的手回到初始位置。

        :param which_hand: 使用的手 (int)
        :param hand_reach_from: 伸手的方向，0 表示从上方，1 表示从正前方
        :return: 如果使用 gRPC，则返回操作结果；否则无返回值
        """
        which_hand_enum = Character.get_which_hand_action_enum(which_hand)
        if self.__animation_stub is not None:
            HandReachFrom: basic_pb2.EHandReachFrom
            if hand_reach_from == 0:
                HandReachFrom = basic_pb2.EHandReachFrom.ABOVE
            else:
                HandReachFrom = basic_pb2.EHandReachFrom.FRONT
            return self.hand_reach_back_proto(
                WhichHand=which_hand_enum, HandReachFrom=HandReachFrom
            )
        self.__animation_component.hand_reach_back(
            which_hand, hand_reach_from=hand_reach_from
        )

    def finger_reach_out_location(self, which_hand, location):
        """
        使角色的手指伸向指定位置。

        :param which_hand: 使用的手 (int)
        :param location: 目标位置 (UELocation)
        :return: 如果使用 gRPC，则返回操作结果；否则无返回值
        """
        which_hand_enum = Character.get_which_hand_action_enum(which_hand)
        if self.__animation_stub is not None:
            return self.finger_reach_out_location_proto(
                WhichHand=which_hand_enum, Location=location
            )
        self.__animation_component.finger_reach_out_location(which_hand, location)

    def finger_reach_out_object(
        self, which_hand, object, b_use_socket=False, component_name="None"
    ):
        """
        使角色的手指伸向指定对象。

        :param which_hand: 使用的手 (int)
        :param object: 目标对象
        :param b_use_socket: 是否使用 socket，默认为 False
        :param component_name: 组件名称，默认为 "None"
        :return: 如果使用 gRPC，则返回操作结果；否则无返回值
        """
        which_hand_enum = Character.get_which_hand_action_enum(which_hand)
        if self.__animation_stub is not None:
            return self.finger_reach_out_object_proto(
                Object=object,
                WhichHand=which_hand_enum,
                TargetSocket=b_use_socket,
                ComponentName=component_name,
            )
        self.__animation_component.hand_reach_out_object(
            which_hand=which_hand, object=object, b_use_socket=b_use_socket
        )

    """ 待修复的 Proto 方法 """

    def enable_idle_ai(self, bEnableAI):
        """
        启用或禁用角色的空闲 AI。

        :param bEnableAI: 布尔值，True 表示启用 AI，False 表示禁用 AI
        """
        self.__animation_component.enable_idle_ai(bEnableAI)

    """ 使用 Proto 的动画方法 """

    def move_to_location_proto(self, location: UELocation, move_speed=None, is_can_run=None, accelerate_delay_time=None, max_speed=None, execution_type=None):
        """
        使用 gRPC 将角色移动到指定位置。

        :param location: 目标位置 (UELocation)
        :return: 动作 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_MoveTo
        request.move_to.location.x = location.X
        request.move_to.location.y = location.Y
        request.move_to.location.z = location.Z
        if execution_type is not None:
            request.animation_execution_type = execution_type
        if move_speed is not None:
            request.move_to.move_speed = move_speed
        if is_can_run is not None:
            request.move_to.b_can_walk = is_can_run
        if accelerate_delay_time is not None:
            request.move_to.accelerate_delay_time = accelerate_delay_time
        if max_speed is not None:
            request.move_to.max_speed = max_speed
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def move_to_object_proto(self, object: ObjectBase, b_use_socket):
        """
        使用 gRPC 将角色移动到指定对象。

        :param object: 目标对象 (ObjectBase)
        :param b_use_socket: 是否使用 socket
        :return: 动作 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_MoveTo
        request.move_to.object_params.target.id = object.id
        request.move_to.object_params.b_use_socket = b_use_socket
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def move_to_component(self, component_name, b_use_socket):
        """
        通过 gRPC 将角色移动到指定组件。

        :param component_name: 组件名称 (str)
        :param b_use_socket: 是否使用 socket (bool)
        :return: 动作 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_MoveTo
        request.move_to.component_params.target.id = component_name
        request.move_to.component_params.b_use_socket = b_use_socket
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def turn_around_to_location_proto(self, location: UELocation):
        """
        通过 gRPC 将角色转向指定位置。

        :param location: 目标位置 (UELocation)
        :return: 动作 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_TurnAroundTowards
        request.turn_around_to.location.x = location.X
        request.turn_around_to.location.y = location.Y
        request.turn_around_to.location.z = location.Z
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def turn_around_to_object_proto(self, object: ObjectBase, b_use_socket):
        """
        通过 gRPC 将角色转向指定对象。

        :param object: 目标对象 (ObjectBase)
        :param b_use_socket: 是否使用 socket (bool)
        :return: 动作 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_TurnAroundTowards
        request.turn_around_to.object_params.target.id = object.id
        request.turn_around_to.object_params.b_use_socket = b_use_socket
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def turn_around_to_component(self, component_name, b_use_socket):
        """
        通过 gRPC 将角色转向指定组件。

        :param component_name: 组件名称 (str)
        :param b_use_socket: 是否使用 socket (bool)
        :return: 动作 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_TurnAroundTowards
        request.turn_around_to.component_params.target.id = component_name
        request.turn_around_to.component_params.b_use_socket = b_use_socket
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def sit_down_proto(self):
        """
        通过 gRPC 使角色坐下。

        :return: 动作 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_LocoMotionState
        request.loco_motion.locomotion_status = locomotion_pb2.ELocoMotionEnum.SIT_DOWN
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def sit_down_object(self, Object):
        """
        通过 gRPC 使角色在指定对象上坐下。

        :param Object: 目标对象 (ObjectBase)
        :return: 动作 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_SitDown
        request.sitdown.object_id = Object.id
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def read_book(self, time=5):
        """
        通过 gRPC 使角色阅读书籍。

        :param time: 阅读时间，默认 5 秒 (int)
        :return: 动作 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_ReadBook
        request.read_book.time = time
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def sit_down_location(self, Location):
        """
        通过 gRPC 使角色在指定位置坐下。

        :param Location: 目标位置 (UELocation)
        :return: 动作 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_SitDown
        request.sitdown.location = Location
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def lie_down_proto(self):
        """
        通过 gRPC 使角色躺下。

        :return: 动作 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_LocoMotionState
        request.loco_motion.locomotion_status = locomotion_pb2.ELocoMotionEnum.LIE_DOWN
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def stand_up_proto(self):
        """
        通过 gRPC 使角色站起。

        :return: 动作 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_LocoMotionState
        request.loco_motion.locomotion_status = locomotion_pb2.ELocoMotionEnum.STAND
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def wave_hand_proto(self, which_hand:en.WhichHand):
        """
        通过 gRPC 使角色挥手。

        :return: 动作 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_GestureState
        request.gesture.gesture_status = gesture_pb2.EGestureEnum.WAVE_HAND
        request.gesture.which_hand_action = which_hand.value
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def raise_hand_proto(self, which_hand:en.WhichHand):
        """
        通过 gRPC 使角色举手。

        :return: 动作 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_GestureState
        request.gesture.gesture_status = gesture_pb2.EGestureEnum.RAISE_HAND
        request.gesture.which_hand_action = which_hand.value
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def hand_rotator_proto(
        self,
        HandBackOrOut,
        rotation: UERotation = UERotation(0, 0, 0, 0),
        AnimTimeLenght=1,
        WhichHandAction=basic_pb2.EWhichHandAction.RIGHT,
    ):
        """
        通过 gRPC 控制角色手部旋转。

        :param HandBackOrOut: 手的动作类型 (EHandOutOrBack)
        :param rotation: 目标旋转 (UERotation)
        :param AnimTimeLenght: 动画持续时间 (int)
        :param WhichHandAction: 手部动作类型 (basic_pb2.EWhichHandAction)
        :return: 动作 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.ACCH_HandRotator
        if isinstance(WhichHandAction, en.WhichHand):
            request.hand_rotator.which_hand_action = WhichHandAction.value
        else :
            request.hand_rotator.which_hand_action = WhichHandAction
        if HandBackOrOut == en.HandOutOrBack.HAND_OUT:
            request.hand_rotator.hand_back_or_out = basic_pb2.EHandBackOrOut.HAND_OUT
        else:
            request.hand_rotator.hand_back_or_out = basic_pb2.EHandBackOrOut.HAND_BACK
        request.hand_rotator.anim_time = AnimTimeLenght
        request.hand_rotator.target_rotation.x = rotation.X
        request.hand_rotator.target_rotation.y = rotation.Y
        request.hand_rotator.target_rotation.z = rotation.Z
        request.hand_rotator.target_rotation.w = rotation.W
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def nod_head_proto(self):
        """
        通过 gRPC 使角色点头。

        :return: 动作 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_GestureState
        request.gesture.gesture_status = gesture_pb2.EGestureEnum.NOD_HEAD
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def shake_head_proto(self):
        """
        通过 gRPC 使角色摇头。

        :return: 动作 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_GestureState
        request.gesture.gesture_status = gesture_pb2.EGestureEnum.SHAKE_HEAD
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def chat_proto(self):
        """
        通过 gRPC 使角色聊天。

        :return: 动作 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_GestureState
        request.gesture.gesture_status = gesture_pb2.EGestureEnum.CHAT
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def eat_or_drink_proto(self):
        """
        通过 gRPC 使角色吃或喝。

        :return: 动作 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_GestureState
        request.gesture.gesture_status = gesture_pb2.EGestureEnum.EAT_OR_DRINK
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def wipe_dirt_proto(self, dirt: Dirt):
        """
        通过 gRPC 使角色擦拭污垢。

        :param dirt: 需要擦拭的污垢对象 (Dirt)
        :return: 动作 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_GestureState
        request.gesture.gesture_status = gesture_pb2.EGestureEnum.WIPE
        request.gesture.wipe_dirt.dirt.id = dirt.id
        try:
            request.gesture.wipe_dirt.dirty_subject.id = dirt.attached_object.id
        except AttributeError:
            request.gesture.wipe_dirt.dirty_subject.id = dirt.id
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def idle_gesture_proto(self):
        """
        通过 gRPC 使角色进入闲置手势状态。

        :return: 动作 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_GestureState
        request.gesture.gesture_status = gesture_pb2.EGestureEnum.IDLE
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def point_at_proto(self, Location: UELocation, WhichHand=WhichHand.RIGHT_HAND):
        """
        通过 gRPC 使角色指向指定位置。

        :param Location: 目标位置 (UELocation)
        :return: 动作 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AAO_PointAt
        request.aim_offset.b_aiming = True
        request.aim_offset.location.x = Location.X
        request.aim_offset.location.y = Location.Y
        request.aim_offset.location.z = Location.Z
        if WhichHand == en.WhichHand.LEFT_HAND:
            request.aim_offset.is_left = True
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def cancel_point_at_proto(self):
        """
        通过 gRPC 取消角色的指向动作。

        :return: 动作 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AAO_PointAt
        request.aim_offset.b_aiming = False
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def gaze_at_proto(self, Location: UELocation, target=None):
        """
        通过 gRPC 使角色注视指定位置。

        :param Location: 目标位置 (UELocation)
        :return: 动作 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AAO_GazeAt
        request.aim_offset.b_aiming = True
        request.aim_offset.location.x = Location.X
        request.aim_offset.location.y = Location.Y
        request.aim_offset.location.z = Location.Z
        if target:
         request.aim_offset.object_id=target.id
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def cancel_gaze_at_proto(self):
        """
        通过 gRPC 取消角色的注视动作。

        :return: 动作 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AAO_GazeAt
        request.aim_offset.b_aiming = False
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def take_object_proto(
        self,
        Object: ObjectBase,
        TargetSocket,
        container_name,
        WhichHand=basic_pb2.EWhichHandAction.RIGHT,
        ignore_error_code=False
    ):
        """
        通过 gRPC 操作让角色抓取物体。

        :param Object: 目标物体 (ObjectBase)
        :param TargetSocket: 是否使用目标插槽 (bool)
        :param container_name: 容器名称 (str)
        :param WhichHand: 使用的手 (basic_pb2.EWhichHandAction)
        :return: 动作 ID
        """
        self.hand_reach_out_object_proto(Object, TargetSocket, WhichHand, ignore_error_code = ignore_error_code)
        self.hand_grab_object_proto(Object, TargetSocket, container_name, WhichHand)
        return self.hand_reach_back_proto(WhichHand)

    def put_down_sth_to_location_proto(
        self,
        location,
        b_auto_rotate=False,
        rotation=UERotation(0, 0, 0, 1),
        b_force_locate=True,
        which_hand=basic_pb2.EWhichHandAction.RIGHT,
        b_force_release=False,
        b_disable_physics=False,
        hand_offset=UELocation(),
    ):
        """
        通过 gRPC 操作让角色将物体放置到指定位置。

        :param location: 目标位置 (UELocation)
        :param b_auto_rotate: 是否自动旋转 (bool)
        :param rotation: 目标旋转 (UERotation)
        :param b_force_locate: 是否强制定位 (bool)
        :param which_hand: 使用的手 (basic_pb2.EWhichHandAction)
        :param b_force_release: 是否强制释放 (bool)
        :param b_disable_physics: 是否禁用物理效果 (bool)
        :param hand_offset: 手的位置偏移 (UELocation)
        :return: 动作 ID
        """
        HandLocation = UELocation(
            location.X + hand_offset.X,
            location.Y + hand_offset.Y,
            location.Z + hand_offset.Z,
        )
        self.hand_reach_out_location_proto(HandLocation, which_hand)
        self.hand_release_proto(
            which_hand,
            location,
            b_force_locate,
            b_auto_rotate,
            rotation,
            b_force_release,
            b_disable_physics=b_disable_physics,
        )
        return self.hand_reach_back_proto(which_hand)

    def wait_for_time_proto(self, TimeDuration: float, decrease_boredom=0):
        """
        通过 gRPC 让角色等待指定时间。

        :param TimeDuration: 等待时长 (float)
        :param decrease_boredom: 降低的无聊值 (int)
        :return: 动作 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_Wait
        request.wait.time_duration = TimeDuration
        request.animation_effect_attribute.decrease_boredom = decrease_boredom
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def hand_reach_out_location_proto(
        self,
        Location: UELocation,
        WhichHand=basic_pb2.EWhichHandAction.RIGHT,
        HandReachFrom=basic_pb2.EHandReachFrom.ABOVE,
        Rotation=UERotation(),
        b_auto_offset=False,
    ):
        """
        通过 gRPC 操作让角色的手伸向指定位置。

        :param Location: 目标位置 (UELocation)
        :param WhichHand: 使用的手 (basic_pb2.EWhichHandAction)
        :param HandReachFrom: 手伸出方向 (basic_pb2.EHandReachFrom)
        :return: 动作 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.ACCH_HandReach
        request.hand_reach.which_hand_action = WhichHand
        request.hand_reach.object_location.x = Location.X
        request.hand_reach.object_location.y = Location.Y
        request.hand_reach.object_location.z = Location.Z
        request.hand_reach.hand_back_or_out = basic_pb2.EHandBackOrOut.HAND_OUT
        request.hand_reach.hand_reach_from = HandReachFrom
        request.hand_reach.object_rotation.x = Rotation.X
        request.hand_reach.object_rotation.y = Rotation.Y
        request.hand_reach.object_rotation.z = Rotation.Z
        request.hand_reach.object_rotation.w = Rotation.W
        request.hand_reach.b_auto_offset = b_auto_offset
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def hand_reach_out_object_proto(
        self,
        Object: ObjectBase,
        TargetSocket=False,
        WhichHand=basic_pb2.EWhichHandAction.RIGHT,
        HandReachFrom=basic_pb2.EHandReachFrom.ABOVE,
        ignore_error_code=False
    ):
        """
        通过 gRPC 操作让角色的手伸向目标物体。

        :param Object: 目标物体 (ObjectBase)
        :param TargetSocket: 是否使用目标插槽 (bool)
        :param WhichHand: 使用的手 (basic_pb2.EWhichHandAction)
        :param HandReachFrom: 手伸出方向 (basic_pb2.EHandReachFrom)
        :return: 动作 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.ACCH_HandReach
        request.hand_reach.which_hand_action = WhichHand
        request.hand_reach.object_params.target.id = Object.id
        request.hand_reach.object_params.b_use_socket = TargetSocket
        request.hand_reach.hand_back_or_out = basic_pb2.EHandBackOrOut.HAND_OUT
        request.hand_reach.hand_reach_from = HandReachFrom
        request.ignore_error_code=ignore_error_code
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def hand_reach_out_component(
        self,
        component_name,
        TargetSocket=False,
        WhichHand=basic_pb2.EWhichHandAction.RIGHT,
        HandReachFrom=basic_pb2.EHandReachFrom.ABOVE,
    ):
        """
        通过 gRPC 操作让角色的手伸向指定组件。

        :param component_name: 组件名称 (str)
        :param TargetSocket: 是否使用目标插槽 (bool)
        :param WhichHand: 使用的手 (basic_pb2.EWhichHandAction)
        :param HandReachFrom: 手伸出方向 (basic_pb2.EHandReachFrom)
        :return: 动作 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.ACCH_HandReach
        request.hand_reach.which_hand_action = WhichHand
        request.hand_reach.component_params.target.id = component_name
        request.hand_reach.component_params.b_use_socket = TargetSocket
        request.hand_reach.hand_back_or_out = basic_pb2.EHandBackOrOut.HAND_OUT
        request.hand_reach.hand_reach_from = HandReachFrom
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def finger_reach_out_location_proto(
        self,
        Location: UELocation,
        WhichHand=basic_pb2.EWhichHandAction.RIGHT,
        HandReachFrom=basic_pb2.EHandReachFrom.ABOVE,
    ):
        """
        通过 gRPC 操作让角色的手指伸向指定位置。

        :param Location: 目标位置 (UELocation)
        :param WhichHand: 使用的手 (basic_pb2.EWhichHandAction)
        :param HandReachFrom: 手伸出方向 (basic_pb2.EHandReachFrom)
        :return: 动作 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.ACCH_FingerReach
        request.hand_reach.which_hand_action = WhichHand
        request.hand_reach.object_location.x = Location.X
        request.hand_reach.object_location.y = Location.Y
        request.hand_reach.object_location.z = Location.Z
        request.hand_reach.hand_back_or_out = basic_pb2.EHandBackOrOut.HAND_OUT
        request.hand_reach.hand_reach_from = HandReachFrom
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def finger_reach_out_object_proto(
        self,
        Object: ObjectBase,
        TargetSocket,
        ComponentName,
        WhichHand=basic_pb2.EWhichHandAction.RIGHT,
        HandReachFrom=basic_pb2.EHandReachFrom.ABOVE,
    ):
        """
        通过 gRPC 操作让角色的手指伸向目标物体。

        :param Object: 目标物体 (ObjectBase)
        :param TargetSocket: 是否使用目标插槽 (bool)
        :param ComponentName: 组件名称 (str)
        :param WhichHand: 使用的手 (basic_pb2.EWhichHandAction)
        :param HandReachFrom: 手伸出方向 (basic_pb2.EHandReachFrom)
        :return: 动作 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.ACCH_FingerReach
        request.hand_reach.which_hand_action = WhichHand
        request.hand_reach.object_params.target.id = Object.id
        request.hand_reach.object_params.b_use_socket = TargetSocket
        request.hand_reach.hand_back_or_out = basic_pb2.EHandBackOrOut.HAND_OUT
        request.hand_reach.hand_reach_from = HandReachFrom
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def hand_grab_object_proto(
        self,
        Object: ObjectBase,
        UseSocket,
        container_unique_name="",
        WhichHand=basic_pb2.EWhichHandAction.RIGHT,
    ):
        """
        通过 gRPC 操作让角色抓取物体。

        :param Object: 目标物体 (ObjectBase)
        :param UseSocket: 是否使用目标插槽 (bool)
        :param container_unique_name: 容器唯一名称 (str)
        :param WhichHand: 使用的手 (basic_pb2.EWhichHandAction)
        :return: 动作 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.ACCH_HandGrabObject
        request.hand_grab_object.which_hand_action = WhichHand
        request.hand_grab_object.object_params.target.id = Object.id
        request.hand_grab_object.object_params.b_use_socket = UseSocket
        request.hand_grab_object.container_unique_name = container_unique_name
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def hand_release_proto(
        self,
        which_hand: basic_pb2.EWhichHandAction,
        desired_location: UELocation,
        b_force_locate=False,
        b_auto_rotate=False,
        rotation=UERotation(0, 0, 0, 1),
        b_force_release=False,
        container_uniqueName="",
        b_disable_physics=False,
    ):
        """
        通过 gRPC 操作让角色释放手中的物体。

        :param which_hand: 使用的手 (basic_pb2.EWhichHandAction)
        :param desired_location: 目标位置 (UELocation)
        :param b_force_locate: 是否强制定位 (bool)
        :param b_auto_rotate: 是否自动旋转 (bool)
        :param rotation: 目标旋转 (UERotation)
        :param b_force_release: 是否强制释放 (bool)
        :param container_uniqueName: 容器唯一名称 (str)
        :param b_disable_physics: 是否禁用物理效果 (bool)
        :return: 动作 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.ACCH_HandRelease
        request.hand_release.which_hand_action = which_hand
        request.hand_release.target_location.x = desired_location.X
        request.hand_release.target_location.y = desired_location.Y
        request.hand_release.target_location.z = desired_location.Z
        request.hand_release.b_auto_rotate = b_auto_rotate
        request.hand_release.b_force_locate = b_force_locate
        request.hand_release.rotation.x = rotation.X
        request.hand_release.rotation.y = rotation.Y
        request.hand_release.rotation.z = rotation.Z
        request.hand_release.rotation.w = rotation.W
        request.hand_release.b_force_release = b_force_release
        request.hand_release.container_unique_name = container_uniqueName
        request.hand_release.b_disable_physics = b_disable_physics
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def hand_reach_back_proto(
        self,
        WhichHand=basic_pb2.EWhichHandAction.RIGHT,
        HandReachFrom=basic_pb2.EHandReachFrom.ABOVE,
    ):
        """
        通过 gRPC 操作让角色的手收回到指定位置。

        :param WhichHand: 使用的手 (basic_pb2.EWhichHandAction)
        :param HandReachFrom: 手伸出方向 (basic_pb2.EHandReachFrom)
        :return: 动作 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.ACCH_HandReach
        request.hand_reach.which_hand_action = WhichHand
        request.hand_reach.hand_back_or_out = basic_pb2.EHandBackOrOut.HAND_BACK
        request.hand_reach.hand_reach_from = HandReachFrom
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def play_anim_sequence_proto(self, anim_sequence_name: str, decrease_boredom=0, b_action_async=False, key="", target_id="", anin_value= 0.0):
        """
        播放指定的动画序列。

        :param anim_sequence_name: 动画序列名称 (str)
        :param decrease_boredom: 减少无聊值 (int)
        :return: 动作 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_PlayAnimSequence
        request.animation_effect_attribute.decrease_boredom = decrease_boredom
        request.play_animation.animation_to_play = (
            self.kAnimationSequence + anim_sequence_name
        )
        request.play_animation.b_action_async = b_action_async
        request.play_animation.key = key
        request.play_animation.target_id = target_id
        request.play_animation.anim_value = anin_value
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def wipe_quadrilateral_proto(
        self,
        conner1: UELocation,
        conner2: UELocation,
        conner3: UELocation,
        conner4: UELocation,
        WhichHand=basic_pb2.EWhichHandAction.RIGHT,
    ):
        """
        通过 gRPC 操作让角色擦拭指定的四边形区域。

        :param conner1: 四边形的第一个角 (UELocation)
        :param conner2: 四边形的第二个角 (UELocation)
        :param conner3: 四边形的第三个角 (UELocation)
        :param conner4: 四边形的第四个角 (UELocation)
        :param WhichHand: 使用的手 (basic_pb2.EWhichHandAction)
        :return: 动作 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.ACCH_WipeQuadrilateral
        request.wipe_quadrilateral.which_hand_action = WhichHand
        request.wipe_quadrilateral.conner1.x = conner1.X
        request.wipe_quadrilateral.conner1.y = conner1.Y
        request.wipe_quadrilateral.conner1.z = conner1.Z
        request.wipe_quadrilateral.conner2.x = conner2.X
        request.wipe_quadrilateral.conner2.y = conner2.Y
        request.wipe_quadrilateral.conner2.z = conner2.Z
        request.wipe_quadrilateral.conner3.x = conner3.X
        request.wipe_quadrilateral.conner3.y = conner3.Y
        request.wipe_quadrilateral.conner3.z = conner3.Z
        request.wipe_quadrilateral.conner4.x = conner4.X
        request.wipe_quadrilateral.conner4.y = conner4.Y
        request.wipe_quadrilateral.conner4.z = conner4.Z
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def do_task_proto(self):
        """
        通过 gRPC 操作执行指定任务。

        :return: None
        """
        request = basic_pb2.EmptyRequest()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        self.__animation_stub.DoTask(request)

    @staticmethod
    def get_which_hand_action_enum(which_hand: int):
        """
        获取手部动作对应的枚举值。

        :param which_hand: 手的编号 (int)
        :return: basic_pb2.EWhichHandAction 枚举值
        """
        WhichHand = basic_pb2.EWhichHandAction.RIGHT
        if which_hand == 1:
            WhichHand = basic_pb2.EWhichHandAction.LEFT
        elif which_hand == 2:
            WhichHand = basic_pb2.EWhichHandAction.TWO
        return WhichHand

    def get_placed_object_info_from_container(self, object, container_unique_name):
        """
        从容器中获取放置对象的信息。

        :param object: 目标对象
        :param container_unique_name: 容器的唯一名称
        :return: 返回对象的位置信息
        """
        request = scene_pb2.ContainerInteractiveParams()
        request.subject.id = object.id
        request.component.id = container_unique_name
        response = scene_pb2_grpc.SceneServiceStub(
            self.insecure_channel
        ).GetPlacedObjectInfoFromContainer(request)
        return response

    def get_take_out_object_info_from_container(
        self, object_name, container_unique_name
    ):
        """
        从容器中获取取出的对象的信息。

        :param object_name: 对象名称
        :param container_unique_name: 容器的唯一名称
        :return: 返回对象的位置信息
        """
        request = scene_pb2.ContainerInteractiveParams()
        request.subject.id = object_name
        request.component.id = container_unique_name
        response = scene_pb2_grpc.SceneServiceStub(
            self.insecure_channel
        ).GetTakeOutObjectInfoFromContainer(request)
        return response

    def switch_door_proto(self, bIs_open, component_name, whitch_hand, ignore_error_code=False):
        """
        通过 gRPC 操作切换门的状态。

        :param bIs_open: 是否打开门 (bool)
        :param component_name: 门的组件名称 (str)
        :param whitch_hand: 使用的手 (basic_pb2.EWhichHandAction)
        :return: 动作 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_SwitchDoor
        request.switch_door.which_hand_action = whitch_hand
        request.switch_door.component.id = component_name
        request.switch_door.b_open = bIs_open
        request.ignore_error_code=ignore_error_code
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def move_and_take_object(self, Object, which_hand=0):
        """
        移动到对象的位置并抓取对象。

        :param Object: 目标对象
        :param which_hand: 使用的手 (int)
        """
        if isinstance(which_hand, WhichHand):
            which_hand = which_hand.value
        self.move_to_object(Object)
        self.turn_around_to_object(Object)
        self.take_object(Object, which_hand=which_hand)

    import queue

    def subscribe_overlap_event(self, event_queue: queue.Queue):
        """
        订阅对象的重叠事件，当事件发生时将其放入指定的队列中。

        :param event_queue: 用于存放事件的队列 (queue.Queue)
        :return: 返回订阅的响应
        """

        def gen_private_request(self):
            request = event_notifier_pb2.OverlapEventRequest()
            request.subject.id = self.id
            request.component.id = self.get_component_guid_by_name(
                GrpcTags.Component_EventNotifier
            )
            request.b_active = True
            yield request

        def private_receive_event(event_queue, responses):
            try:
                for response in responses:
                    print("Collision Occurred!")
                    event_queue.put(response)
            except Exception as e:
                print(e)
                return

        stub = event_notifier_pb2_grpc.EventNotifierServiceStub(self.insecure_channel)
        responses = stub.SubscribeOverlapEvents(gen_private_request(self))

        import threading

        thread = threading.Thread(
            target=private_receive_event, args=(event_queue, responses)
        )
        thread.daemon = True
        thread.start()
        return responses

    def motion_control(
        self,
        simulator_params_queue: queue.Queue,
        animation_sequence_queue: queue.Queue,
        required_params: animation_pb2.MotionControllerRequiredParams,
    ):
        """
        控制对象的动作，通过队列传递模拟参数和动画序列。

        :param simulator_params_queue: 模拟器参数队列 (queue.Queue)
        :param animation_sequence_queue: 动画序列队列 (queue.Queue)
        :param required_params: 动作控制器所需的参数 (animation_pb2.MotionControllerRequiredParams)
        :return: 返回动作控制的响应
        """

        def gen_private_request(
            self,
            animation_sequence_queue: queue.Queue,
            required_params: animation_pb2.MotionControllerRequiredParams,
        ):
            first_request = animation_pb2.MotionControlResult()
            first_request.component.id = self.get_component_guid_by_name(
                GrpcTags.Component_Animation
            )
            first_request.required_params.CopyFrom(required_params)
            yield first_request

            other_request = animation_pb2.MotionControlResult()
            other_request.component.id = self.get_component_guid_by_name(
                GrpcTags.Component_Animation
            )
            while True:
                animation_sequence = animation_sequence_queue.get()
                if animation_sequence is None:
                    break
                other_request.animation_sequence = animation_sequence
                yield other_request

        def private_receive_params(event_queue, responses):
            try:
                for response in responses:
                    event_queue.put(response)
            except Exception as e:
                print(e)
                return

        stub = grpc_client_pb2_grpc.AnimationServiceStub(self.insecure_channel)
        responses = stub.MotionControl(
            gen_private_request(
                self,
                animation_sequence_queue=animation_sequence_queue,
                required_params=required_params,
            )
        )
        import threading

        thread = threading.Thread(
            target=private_receive_params, args=(simulator_params_queue, responses)
        )
        thread.daemon = True
        thread.start()
        return responses

    def motion_control_hand_reach(
        self,
        simulator_params_queue: queue.Queue,
        animation_sequence_queue: queue.Queue,
        required_params: animation_pb2.MotionControllerRequiredParams,
    ):
        """
        控制对象的手部动作，通过队列传递模拟参数和动画序列。

        :param simulator_params_queue: 模拟器参数队列 (queue.Queue)
        :param animation_sequence_queue: 动画序列队列 (queue.Queue)
        :param required_params: 动作控制器所需的参数 (animation_pb2.MotionControllerRequiredParams)
        :return: 返回手部动作控制的响应
        """
        return self.motion_control(
            simulator_params_queue, animation_sequence_queue, required_params
        )

    def set_facial_anim(self, facial_anim_name):
        """
        设置对象的情感状态。

        :param facial_anim_name: 面部表情资产的名字
        :return: 返回情感设置的响应
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.ACCH_EmotionControl
        request.facial_anim_name.facial_anim_name =facial_anim_name
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response
    def set_emotion(self, emotion: dict):
        """
        设置对象的情绪状态

        :param emotion: 动作名称 (dict) eg：{"Joy":0,"Anger":0.5,"Sadness":0.5,"Fear":0}
        :return: 返回更改动作的响应
        """
        request = self.__face_component.set_emotion(emotion)
        return request
    def change_free_animation_action_immediately(self, action_name: str):
        """
        立即更改自由动画的动作。

        :param action_name: 动作名称 (str)
        :return: 返回更改动作的响应
        """
        request = animation_pb2.FreeAnimationActionRequest()
        request.action_name = action_name
        request.component.id = self.__animation_component.component_guid
        response = self.__animation_stub.ChangeFreeAnimationActionImmediately(request)
        return response

    def move_and_open_door_and_put_object_in_container(
        self,
        container_object: BaseObject,
        object: BaseObject,
        container_index: int = 0,
        which_hand=0,
        is_close_container_door=False,
    ):
        """
        移动到指定位置，打开门，并将物体放入容器中。

        :param container_object: 容器对象
        :param object: 需要放入容器的对象
        :param container_index: 放入容器的序号，< 0 为自动选择,
        :param which_hand: 使用的手 (0: 右手, 1: 左手, 2: 双手)
        :param is_close_container_door: 是否在放入物体后关闭容器门 (bool)
        """
        if isinstance(which_hand, WhichHand):
            which_hand = which_hand.value
        containers = container_object.get_all_container()
        state = False
        if len(containers) <= 0:
            return state

        put_container = containers[0]
        if container_index >= 0:
            put_container = containers[container_index]
        else:
            for container in containers:
                container_info = self.get_placed_object_info_from_container(
                    object, container.component_guid
                )
                if not container_info.b_success:
                    continue
                else:
                    put_container = container

        container_doors = put_container.get_container_doors()
        for container_door in container_doors:
            self.open_door(
                container_door.id,
                True,
                hand_rotation=UERotation(0, 0.707, 0, 0.707),
            )

        self.do_task_and_wait_finish()

        state = self.move_and_put_down_object_in_container(
            put_container, object, which_hand
        )
        self.do_task()
        if is_close_container_door:
            for container_door in container_doors:
                self.open_door(
                    container_doors[0].id, False, hand_rotation=UERotation(0, 0.707, 0, 0.707)
                )
            self.do_task()

        return state

    def move_and_open_door_and_take_object_in_container(
        self,
        container_object: BaseObject,
        object: BaseObject,
        which_hand=0,
        is_close_container_door=False,
    ):
        """
        移动到指定位置，打开门，并从容器中取出物体。

        :param container_object: 容器对象
        :param object: 需要取出的对象
        :param which_hand: 使用的手 (0: 右手, 1: 左手, 2: 双手)
        :param is_close_container_door: 是否在取出物体后关闭容器门 (bool)
        """
        containers = container_object.get_all_container()

        if len(containers) > 0:
            container = containers[0]
            container_doors = container.get_container_doors()
            if len(container_doors) > 0:
                self.open_door(
                    container_doors[0].id,
                    True,
                    hand_rotation=UERotation(0, 0.707, 0, 0.707),
                )
                self.do_task_and_wait_finish()
            if not self.is_all_task_finished():
                time.sleep(1)
            self.move_and_take_out_object_from_container(container, object, which_hand)
            self.do_task_and_wait_finish()
            if len(container_doors) > 0 and is_close_container_door:
                self.open_door(
                    container_doors[0].id,
                    False,
                    hand_rotation=UERotation(0, 0.707, 0, 0.707),
                )
            self.do_task_and_wait_finish()

    def climb_platform(self, object):
        """
        让角色爬上指定的平台对象。

        :param object: 目标平台对象
        :return: 返回爬上平台的响应
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_ClimbPlatform
        request.climb_object.object_name = object.id
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response

    def climb_down(self):
        """
        让角色执行下爬动作。

        :return: 返回下爬动作的响应
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_ClimbDown
        request.climb_down.none_msg = "NoneMsg"
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response

    def get_enter_door_location(self, door):
        """
        获取角色进入门的位置。

        :param door: 目标门对象
        :return: 返回进入门的位置 (UELocation)
        """
        request = door_state_pb2.EnterDoorPointRequest()
        doors = door.get_all_doors()
        if len(doors) == 0:
            raise Exception("No door found")
        request.component.id = doors[0].component_guid
        character_location = self.get_pose().location
        request.character_location.x = character_location.X
        request.character_location.y = character_location.Y
        request.character_location.z = character_location.Z
        response = door_state_pb2_grpc.DoorServiceStub(
            self.insecure_channel
        ).GetEnterDoorPoint(request)
        return UELocation(response.x, response.y, response.z)

    def dance(self, name, decrease_boredom=0):
        """
        让角色执行跳舞动画。

        :param name: 动画名称 (str)
        :param decrease_boredom: 减少无聊值 (int)
        """
        self.play_anim_sequence(name, decrease_boredom)

    @staticmethod
    def file_to_bytes(file_path) -> bytes:
        """
        将文件转换为字节数据。

        :param file_path: 文件路径 (str)
        :return: 返回文件的字节数据 (bytes)
        """
        try:
            with open(file_path, "rb") as file:
                sound_data = file.read()
                return sound_data
        except Exception as e:
            print(f"Error reading file: {e}")
            return bytes()

    def play_facial_animation_with_sound_file_proto(
        self, music_path, frequency, morph_target_num, frame_num, weights
    ):
        """
        播放带有声音文件的面部动画。

        :param music_path: 音乐文件路径 (str)
        :param frequency: 频率 (int)
        :param morph_target_num: 变形目标数量 (int)
        :param frame_num: 帧数 (int)
        :param weights: 权重 (list)
        :return: 返回播放动画的命令 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.ACCH_MouthControl
        request.say.frequency = frequency
        request.say.morph_target_num = morph_target_num
        request.say.frame_num = frame_num
        request.say.weights.extend(weights.tolist())
        request.say.sound_data = self.file_to_bytes(music_path)
        request.say.is_use_wav = True
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response.id

    def interact_object_proto(
        self,
        interact_key,
        target_id="",
        str_value="",
        float_value=0,
        bool_value=False,
        location=UELocation(),
        rotation=UERotation(),
    ):
        """
        让角色与对象进行交互。

        :param interact_key: 交互键 (str)
        :param target_id: 目标 ID (str)
        :param str_value: 字符串值 (str)
        :param float_value: 浮点数值 (float)
        :param bool_value: 布尔值 (bool)
        :param location: 交互的位置 (UELocation)
        :param rotation: 交互的旋转 (UERotation)
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_InteractObject
        request.interact_object.interact_key = interact_key
        request.interact_object.interact_param.target_id = target_id
        request.interact_object.interact_param.str_value = str_value
        request.interact_object.interact_param.float_value = float_value
        request.interact_object.interact_param.bool_value = bool_value
        request.interact_object.interact_param.vector_value.x = location.X
        request.interact_object.interact_param.vector_value.y = location.Y
        request.interact_object.interact_param.vector_value.z = location.Z
        request.interact_object.interact_param.rotation_value.x = rotation.X
        request.interact_object.interact_param.rotation_value.y = rotation.Y
        request.interact_object.interact_param.rotation_value.z = rotation.Z
        request.interact_object.interact_param.rotation_value.w = rotation.W
        self.__animation_stub.EnqueAnimationCommand(request)

    def seek_curtain(self, object):
        self.interact_object_proto(
            interact_key="PlayMontage", target_id=object.id, str_value="seek"
        )

    def LockOrUnLock(self, object, status: bool):
        self.interact_object_proto(
            interact_key="PlayMontage",
            target_id=object.id,
            str_value="key",
            bool_value=status,
        )

    def touch_on_off(self, Object, which_hand=0):
        """
        触摸对象以开关设备。

        :param Object: 目标对象 (ObjectBase)
        :param which_hand: 使用的手 (int)，默认为右手
        """
        ONOFFlocaion = Object.get_interact_location()
        self.hand_reach_out_location(which_hand=which_hand, location=ONOFFlocaion)
        self.hand_reach_back(which_hand=which_hand)

    def set_object_state(self, Object, State=True):
        """
        设置对象的状态（开或关）。

        :param Object: 目标对象 (ObjectBase)
        :param State: 目标状态 (bool)，默认为 True (打开)
        """
        self.interact_object_proto(
            interact_key="SetState", target_id=Object.id, bool_value=State
        )

    def set_object_channel(self, Object, Channel=""):
        """
        设置对象的频道。

        :param Object: 目标对象 (ObjectBase)
        :param Channel: 目标频道 (str)，默认为空字符串
        """
        self.interact_object_proto(
            interact_key="SetChannel", target_id=Object.id, str_value=Channel
        )

    def mop_floor(self, dirt=None, location=UELocation()):
        """
        执行拖地动作。

        :param dirt: 可选的污垢对象 (Dirt)
        :param location: 拖地的位置 (UELocation)
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_MopFloor
        request.mop_floor.dirt_id = ""
        if not dirt is None:
            request.mop_floor.dirt_id = dirt.id
        request.mop_floor.location.x = location.X
        request.mop_floor.location.y = location.Y
        request.mop_floor.location.z = location.Z
        response = self.__animation_stub.EnqueAnimationCommand(request)

    def wash_hands(self):
        """
        让角色执行洗手动作。
        """
        self.play_anim_sequence("Wash_Hands")

    def wash_face(self):
        """
        让角色执行洗脸动作。
        """
        self.play_anim_sequence("Wash_Face")

    def wash_self(self, wash_type: en.WashType, faucetobject):
        """
        让角色执行洗手或洗脸动作。

        :param wash_type: 洗涤类型 (wash_pb2.WashType)
        :param faucetobject: 水龙头对象 (ObjectBase)
        :return: 返回洗涤动作的命令 ID
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_Wash
        request.wash_param.wash_type = wash_type.value
        request.wash_param.faucetobjectname = faucetobject.id
        response = self.__animation_stub.EnqueAnimationCommand(request)
        return response

    def wash(self, wash_type: en.WashType, faucetobject):
        """
        执行洗涤动作。

        :param WashType: 洗涤类型 (EWashType)
        :param faucetobject: 水龙头对象 (ObjectBase)
        """
        if wash_type == en.WashType.HANDS:
            self.wash_self(wash_type, faucetobject.id)
        elif wash_type == en.WashType.FACE:
            self.wash_self(wash_type, faucetobject.id)
        elif wash_type == en.WashType.WASH_OBJECT_IN_HAND:
            self.play_anim_sequence_proto("CleanRag")
        else:
            print("无效的洗涤类型")

    def get_object_id_in_hand(self, which_hand):
        """
        获取手中握住的对象的 ID。

        :param which_hand: 指定的手 (basic_pb2.EWhichHandAction)
        :return: 返回对象的响应 (attachment_pb2.ObjectInHandResponse)
        """
        request = attachment_pb2.ObjectInHandParam()
        request.component.id = self.__attachment_component.component_guid
        if isinstance(which_hand, WhichHand):
            request.which_hand_action = which_hand.value
        else:
            request.which_hand_action = which_hand
        response = attachment_pb2_grpc.AttachmentServiceStub(
            self.insecure_channel
        ).GetObjectInHand(request)
        return response

    def get_acoustics_data(self):
        acoustics_data = self.__acoustic_component.get_acoustics_data()
        return acoustics_data

    def slice_object(self, object_id, location=UELocation()):
        """
        执行切割对象的动作。

        :param object_id: 目标对象的 ID (str)
        :param location: 切割的位置 (UELocation)
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_SliceFood
        request.slice_food.food_id = ""
        if object_id is not None:
            request.slice_food.food_id = object_id.id
        request.slice_food.location.x = location.X
        request.slice_food.location.y = location.Y
        request.slice_food.location.z = location.Z
        response = self.__animation_stub.EnqueAnimationCommand(request)

    def move_and_sleep_down(self, sleep_subject):
        location = sleep_subject.get_interact_location()
        print(location)
        self.move_to_location_proto(location)
        self.turn_around_to_object(sleep_subject)
        self.sleep_down(sleep_subject)

    def sleep_down(self, sleep_subject):
        """
        执行睡觉的动作。

        :param sleep_subject: 睡觉的目标对象 (BaseObject)
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_Sleep
        request.sleep.sleep_type = sleep_pb2.SLEEP_DOWN
        request.sleep.subject = sleep_subject.id
        response = self.__animation_stub.EnqueAnimationCommand(request)

    def sleep_up(self):
        """
        执行起床的动作。
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_Sleep
        request.sleep.sleep_type = sleep_pb2.SLEEP_UP
        response = self.__animation_stub.EnqueAnimationCommand(request)

    def pour_water(
        self, object, location=UELocation(), which_hand=0, offset=UELocation()
    ):
        """
        执行倒水的动作。

        :param object: 目标对象 (BaseObject)
        :param location: 倒水的位置 (UELocation)
        :param which_hand: 使用的手 (int)，默认为右手
        :param offset: 倒水的偏移位置 (UELocation)
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_PourWater
        request.pour_water.which_hand_action = Character.get_which_hand_action_enum(
            which_hand
        )
        request.pour_water.target_id = ""
        if object is not None:
            request.pour_water.target_id = object.id

        request.pour_water.location.x = location.X
        request.pour_water.location.y = location.Y
        request.pour_water.location.z = location.Z

        request.pour_water.offset.x = offset.X
        request.pour_water.offset.y = offset.Y
        request.pour_water.offset.z = offset.Z

        response = self.__animation_stub.EnqueAnimationCommand(request)

    def put_on_sth(self, object):
        """
        This method is used to command the character to put on a specified object.

        :param object: The object that will be put on.
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_PutOn
        request.put_on.object_id = object

        response = self.__animation_stub.EnqueAnimationCommand(request)

    def take_off_sth(self, object_list):
        """
        This method is used to command the character to take off a specified dress.

        :param dress: The dress that will be taken off.
        """
        request = animation_pb2.AnimationCommandParams()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.command_code = GrpcAnimationTags.AC_TakeOff
        request.take_off.object_id.extend(object_list)

        response = self.__animation_stub.EnqueAnimationCommand(request)


    def get_anim_name(self, action_name):
        """
        根据动作名称获取动画的名称。

        :param action_name: 动作名称 (str)
        :return: 动画名称的响应 (animation_pb2.KeyNameResponse)
        """
        request = animation_pb2.KeyName()
        request.subject.id = self.id
        request.component.id = self.__animation_component.component_guid
        request.key_name = action_name
        return self.__animation_stub.GetAnimNames(request)

    def interact_object(self, object, which_hand=0):
        self.move_to_location(object.get_interact_location())
        self.turn_around_to_location(object.get_interact_location())
        self.hand_reach_out_location(which_hand, object.get_interact_location())
        self.hand_reach_back(which_hand)

    def get_input_component(self):
        return self.__input_component

    def input_action(self, move_vec, walk_speed, angular_speed, jump, crouch):
        """
        处理输入的动作信息。

        :param move_vec: 运动向量，表示移动的方向和力度。
        :param walk_speed: 步行速度，表示角色的步行速度。
        :param angular_speed: 角速度，表示角色的旋转速度。
        :param jump: 布尔值，True 表示角色跳跃，False 表示不跳跃。
        :param crouch: 布尔值，True 表示角色蹲下，False 表示站立。
        """
        self.__input_component.input_action(
            move_vec, walk_speed, angular_speed, jump, crouch
        )

    def input_move_vec(self, move_vec):
        """
        设置移动向量。

        :param move_vec: 运动向量，表示角色的移动方向和力度。
        """
        self.__input_component.input_move_vec(move_vec)

    def input_jump(self):
        """
        触发跳跃动作。

        此函数会指示角色执行跳跃动作。
        """
        self.__input_component.input_jump()

    def input_crouch(self, is_crouch):
        """
        设置蹲下状态。

        :param is_crouch: 布尔值，True 表示角色蹲下，False 表示角色站立。
        """
        self.__input_component.input_crouch(is_crouch)

    def chang_walk_speed(self, walk_speed):
        """
        修改步行速度。

        :param walk_speed: 新的步行速度值。
        """
        self.__input_component.chang_walk_speed(walk_speed)

    def chang_angular_speed(self, angular_speed):
        """
        修改角速度。

        :param angular_speed: 新的角速度值。
        """
        self.__input_component.chang_angular_speed(angular_speed)

    def stream_input(self, requests):
        """
        流失输入接口

        :param requests: 输入变量的proto结构体。
        """
        self.__input_component.stream_input(requests)

    def init_camera_channel(self, ip_port):
        self.camera_channel = grpc.insecure_channel(
            ip_port,
            options=[
                ("grpc.max_send_message_length", 200 * 1024 * 1024),
                ("grpc.max_receive_message_length", 200 * 1024 * 1024),
            ],
        )

    def subscribe_image(
        self,
        request_images: List[RequestImage],
        rgb=True,
        depth=False,
        segmentation=False,
        mirror_segmentation=False,
        camera_channel=None,
    ):
        print(f"camera_channel: {camera_channel}")
        camera_stub = camera_pb2_grpc.CameraServiceStub(camera_channel)

        # image的request，给出相机id和图像类型
        request = camera_pb2.ImageRequest()
        for request_image in request_images:
            socket_name = request_image_dict[request_image]
            request.camera_config_list.extend([
                camera_pb2.CameraConfig(
                    camera_id=self.id + socket_name,
                    b_rgb=rgb,
                    b_depth=depth,
                    b_segmentation=segmentation,
                    b_mirror_segmentation=mirror_segmentation,
                )
            ])

        return camera_stub.SubscribeImage(request)

    def request_image_to_file(
        self,
        request_images: List[RequestImage],
        rgb=True,
        depth=False,
        segmentation=False,
        mirror_segmentation=False,
    ):
        response: camera_pb2.ImageResponse
        # 获取UE推送返回的数据并存储，包括四类数据rgd-png, depth-hdr, segment-png, mirrorsegment-png
        for response in self.subscribe_image(
            request_images, rgb, depth, segmentation, mirror_segmentation
        ):
            for image in response.camera_image_list:
                script_dir = os.path.dirname(os.path.realpath(__file__))
                output_dir = os.path.dirname(script_dir)
                output_dir = os.path.dirname(output_dir)
                output_dir = os.path.dirname(output_dir)
                output_dir = output_dir + "/output/"
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                print(output_dir)
                is_print = False
                if rgb:
                    file_path = os.path.normpath(
                        output_dir + "image_" + str(image.render_time) + "rgb.png"
                    )
                    with open(file_path, "wb") as file_object:
                        file_object.write(image.rgb)
                        file_object.close()
                    print(image.render_time)
                    print_time = True

                if depth:
                    file_path = os.path.normpath(
                        output_dir + "image_" + str(image.render_time) + ".hdr"
                    )
                    with open(file_path, "wb") as file_object:
                        file_object.write(image.depth)
                        file_object.close()
                    print(image.render_time)
                    print_time = True

                if segmentation:
                    file_path = os.path.normpath(
                        output_dir + "image_" + str(image.render_time) + "segment.png"
                    )
                    with open(file_path, "wb") as file_object:
                        file_object.write(image.segmentation)
                        file_object.close()
                    print(image.render_time)
                    print_time = True

                if mirror_segmentation:
                    file_path = os.path.normpath(
                        output_dir
                        + "image_"
                        + str(image.render_time)
                        + "mirrorsegment.png"
                    )
                    with open(file_path, "wb") as file_object:
                        file_object.write(image.mirror_segmentation)
                        file_object.close()

    def cache_request_image(
        self,
        request_images: List[RequestImage],
        cache_num=1,
        rgb=True,
        depth=False,
        segmentation=False,
        mirror_segmentation=False,
        camera_channel=None
    ):
        if camera_channel is None:
            camera_channel = self.camera_channel
        # camera_channel =grpc.insecure_channel(
        #     client_ip + ":5056",
        #     options=[
        #         ("grpc.max_send_message_length", 200 * 1024 * 1024),
        #         ("grpc.max_receive_message_length", 200 * 1024 * 1024),
        #     ],
        # )
        with self.mtx:
            self.image_map = {}
            for request_image in request_images:
                # 以相机位置为键存储，值为一个子字典
                if request_image not in self.image_map:
                    key = self.id + request_image_dict[request_image]
                    if key not in self.image_map:
                        self.image_map[key] = deque(maxlen=cache_num)  # 先创建空字典

        def subscribe_and_process():
            try:
                self.stream_subscribe_image = self.subscribe_image(request_images, rgb, depth, segmentation, mirror_segmentation, camera_channel)

                for response in self.stream_subscribe_image:
                    for image in response.camera_image_list:
                        with self.mtx:
                            self.image_map[image.camera_id].append(image)
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.CANCELLED:
                    # print("Image Grpc Stream cancelled")
                    pass
                else:
                    print(f"gRPC Error occurred: {e}")
                return
            except Exception as e:
                if str(e) != "is_success":
                    print(f"Unexpected error in response processing: {e}")
                return

        # 启动订阅和处理响应的线程
        self.subscribe_image_thread = threading.Thread(target=subscribe_and_process)
        self.subscribe_image_thread.daemon = True  # 可选：设置为守护线程
        self.subscribe_image_thread.start()

    def stop_cache_request_image(self):
        """
        停止 cache_request_image 线程和图像流。
        """
        # 取消 gRPC 调用以中断阻塞
        if self.stream_subscribe_image is not None:
            self.stream_subscribe_image.cancel()
            self.stream_subscribe_image = None

        # 等待响应线程结束
        if self.subscribe_image_thread is not None:
            self.subscribe_image_thread.join()
            self.subscribe_image_thread = None
            # print("subscribe image thread stopped")

    def get_last_cache_image(self, request_image: RequestImage):
        with self.mtx:
            assert self.image_map, "你必须首先调用 cache_request_image 方法"

            # 根据传入的 RequestImage 生成键
            key = self.id + request_image_dict[request_image]

            # 检查键是否在 image_map 中
            if key in self.image_map:
                # 获取相机的图像缓存
                if len(self.image_map[key]) > 0:
                    return self.image_map[key][-1]

        # 返回最后缓存的图像字典
        return None

    # def __del__(self):
    #     self.stop_cache_request_image()

    def set_move_ability(self, walk_speed, run_speed, jump_z_velocity, crouch_speed, move_friction = 0.5):
        """
        设置移动能力

        :param walk_speed: 行走速度
        :param run_speed: 奔跑速度
        :param jump_z_velocity: 跳跃z轴初速度
        :param crouch_speed: 蹲伏移动速度
        """
        request = character_attribute_pb2.MoveAttribute()
        request.component.id = self.__attribute_component.component_guid
        request.walk_speed = walk_speed
        request.run_speed = run_speed
        request.jump_velocity = jump_z_velocity
        request.crouch_speed = crouch_speed
        request.move_friction = move_friction
        character_attribute_pb2_grpc.CharacterAttributeServiceStub(self.insecure_channel).SetMoveAbility(request)
    
    def set_taste(self, action_type, sweet=None, acid=None, salty=None, peppery=None, bitter=None):
        # action_type: 0 表示设置 agent 当前的口味状态，1 表示设置 agent 的目标口味，2 表示设置口味的减少速度
        request = character_attribute_pb2.TasteAttribute()
        request.component.id = self.__attribute_component.component_guid
        request.action_type = action_type
        if sweet is not None:
            request.sweet = sweet
        if acid is not None:
            request.acid = acid
        if salty is not None:
            request.salty = salty
        if peppery is not None:
            request.peppery = peppery
        if bitter is not None:
            request.bitter = bitter
        character_attribute_pb2_grpc.CharacterAttributeServiceStub(self.insecure_channel).SetTaste(request)
    
    def set_nutrition(self, action_type, vitamin=None, protein=None, fat=None, carbohydrate=None, fibre=None, inorganic_salt=None):
        # action_type: 0 表示设置 agent 当前的营养状态，1 表示设置营养的减少速度
        request = character_attribute_pb2.NutritionAttribute()
        request.component.id = self.__attribute_component.component_guid
        request.action_type = action_type
        if vitamin is not None:
            request.vitamin = vitamin
        if protein is not None:
            request.protein = protein
        if fat is not None:
            request.fat = fat
        if carbohydrate is not None:
            request.carbohydrate = carbohydrate
        if fibre is not None:
            request.fibre = fibre
        if inorganic_salt is not None:
            request.inorganic_salt = inorganic_salt
        character_attribute_pb2_grpc.CharacterAttributeServiceStub(self.insecure_channel).SetNutrition(request)

    def set_is_can_run(self, can_run):
        """
        设置当baby moveto时是否会奔跑

        :param can_run: 是否可以奔跑。
        """
        request = character_attribute_pb2.MoveAttribute()
        request.component.id = self.__attribute_component.component_guid
        request.can_walk_run = can_run
        character_attribute_pb2_grpc.CharacterAttributeServiceStub(
            self.insecure_channel
        ).SetMoveAbility(request)

    def set_move_to_accelerate_delay_time(self, accelerate_delay_time):
        """
        设置当baby moveto时加速的延迟时间

        :param accelerate_delay_time: 加速的延迟时间
        """
        request = character_attribute_pb2.MoveAttribute()
        request.component.id = self.__attribute_component.component_guid
        request.accelerate_delay_time = accelerate_delay_time
        character_attribute_pb2_grpc.CharacterAttributeServiceStub(
            self.insecure_channel
        ).SetMoveAbility(request)

    def set_walk_speed(self, walk_speed):
        """
        设置行走速度

        :param walk_speed: 行走速度（单位：厘米每秒）
        """
        request = character_attribute_pb2.MoveAttribute()
        request.component.id = self.__attribute_component.component_guid
        request.walk_speed = walk_speed
        character_attribute_pb2_grpc.CharacterAttributeServiceStub(
            self.insecure_channel
        ).SetMoveAbility(request)

    def set_run_speed(self, run_speed):
        """
        设置奔跑速度

        :param run_speed: 奔跑速度（单位：厘米每秒）
        """
        request = character_attribute_pb2.MoveAttribute()
        request.component.id = self.__attribute_component.component_guid
        request.run_speed = run_speed
        character_attribute_pb2_grpc.CharacterAttributeServiceStub(
            self.insecure_channel
        ).SetMoveAbility(request)

    def set_move_friction(self, move_friction):
        """
        设置奔跑速度

        :param run_speed: 奔跑速度（单位：厘米每秒）
        """
        request = character_attribute_pb2.MoveAttribute()
        request.component.id = self.__attribute_component.component_guid
        request.move_friction = move_friction
        character_attribute_pb2_grpc.CharacterAttributeServiceStub(self.insecure_channel).SetMoveAbility(request)

    def set_agent_id(self, agent_id):
        """
        设置agent_id

        :param agent_id: 设置的字段
        """
        request = subject_pb2.AgentIDRequest()
        request.id = self.id
        request.str = agent_id
        subject_pb2_grpc.SubjectServiceStub(self.insecure_channel).SetAgentID(request)

    def set_jump_z_velocity(self, jump_z_velocity):
        """
        设置跳跃上升初速度

        :param set_jump_z_velocity: 上升初速度
        """
        request = character_attribute_pb2.MoveAttribute()
        request.component.id = self.__attribute_component.component_guid
        request.jump_velocity = jump_z_velocity
        character_attribute_pb2_grpc.CharacterAttributeServiceStub(self.insecure_channel).SetMoveAbility(request)

    def set_crouch_speed(self, crouch_speed):
        """
        设置蹲伏移动速度

        :param crouch_speed: 蹲伏移动速度
        """
        request = character_attribute_pb2.MoveAttribute()
        request.component.id = self.__attribute_component.component_guid
        request.crouch_speed = crouch_speed
        character_attribute_pb2_grpc.CharacterAttributeServiceStub(self.insecure_channel).SetMoveAbility(request)

    def set_is_enable_physics_body(self, is_enable):
        """
        设置agent是否开启Mesh物理碰撞
        :param is_enable 是否开启
        """
        request = basic_pb2.BoolRequest()
        request.bool = is_enable
        request.component.id = self.__attribute_component.component_guid
        character_attribute_pb2_grpc.CharacterAttributeServiceStub(self.insecure_channel).SetIsEnablePhysicsBody(request)
        
    def speak_text_with_tongos(self, text: str):
        """
        调用 TongOS 的 TTS 和 Avatar 服务，获取 声音 和 表情 数据，并在 TongSim 中播放。
        :param text 想要说的话
        """
        response = self.__face_component.speak_with_tongos(text)
        return response
