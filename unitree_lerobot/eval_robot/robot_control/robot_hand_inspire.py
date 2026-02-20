from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize  # dds
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_  # idl
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

import numpy as np
from enum import IntEnum
import threading
import time
from multiprocessing import Process, Array

import logging_mp

logger_mp = logging_mp.get_logger(__name__)

Inspire_Num_Motors = 6
kTopicInspireCommand = "rt/inspire/cmd"
kTopicInspireState = "rt/inspire/state"
kTopicInspireFTPLeftCommand = "rt/inspire_hand/ctrl/l"
kTopicInspireFTPRightCommand = "rt/inspire_hand/ctrl/r"
kTopicInspireFTPLeftState = "rt/inspire_hand/state/l"
kTopicInspireFTPRightState = "rt/inspire_hand/state/r"


class Inspire_Controller:
    def __init__(
        self,
        left_hand_array,
        right_hand_array,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
        fps=100.0,
        Unit_Test=False,
        simulation_mode=False,
    ):
        logger_mp.info("Initialize Inspire_Controller...")
        self.fps = fps
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode

        if self.simulation_mode:
            ChannelFactoryInitialize(1)
        else:
            ChannelFactoryInitialize(0)

        # initialize handcmd publisher and handstate subscriber
        self.HandCmb_publisher = ChannelPublisher(kTopicInspireCommand, MotorCmds_)
        self.HandCmb_publisher.Init()

        self.HandState_subscriber = ChannelSubscriber(kTopicInspireState, MotorStates_)
        self.HandState_subscriber.Init()

        # Shared Arrays for hand states
        self.left_hand_state_array = Array("d", Inspire_Num_Motors, lock=True)
        self.right_hand_state_array = Array("d", Inspire_Num_Motors, lock=True)

        # initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        while True:
            if any(self.right_hand_state_array):  # any(self.left_hand_state_array) and
                break
            time.sleep(0.01)
            logger_mp.warning("[Inspire_Controller] Waiting to subscribe dds...")
        logger_mp.info("[Inspire_Controller] Subscribe dds ok.")

        hand_control_process = Process(
            target=self.control_process,
            args=(
                left_hand_array,
                right_hand_array,
                self.left_hand_state_array,
                self.right_hand_state_array,
                dual_hand_data_lock,
                dual_hand_state_array,
                dual_hand_action_array,
            ),
        )
        hand_control_process.daemon = True
        hand_control_process.start()

        logger_mp.info("Initialize Inspire_Controller OK!\n")

    def _subscribe_hand_state(self):
        while True:
            hand_msg = self.HandState_subscriber.Read()
            if hand_msg is not None:
                for idx, id in enumerate(Inspire_Left_Hand_JointIndex):
                    self.left_hand_state_array[idx] = hand_msg.states[id].q
                for idx, id in enumerate(Inspire_Right_Hand_JointIndex):
                    self.right_hand_state_array[idx] = hand_msg.states[id].q
            time.sleep(0.002)

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """
        Set current left, right hand motor state target q
        """
        for idx, id in enumerate(Inspire_Left_Hand_JointIndex):
            self.hand_msg.cmds[id].q = left_q_target[idx]
        for idx, id in enumerate(Inspire_Right_Hand_JointIndex):
            self.hand_msg.cmds[id].q = right_q_target[idx]

        self.HandCmb_publisher.Write(self.hand_msg)
        # logger_mp.debug("hand ctrl publish ok.")

    def control_process(
        self,
        left_hand_array,
        right_hand_array,
        left_hand_state_array,
        right_hand_state_array,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
    ):
        self.running = True

        left_q_target = np.full(Inspire_Num_Motors, 1.0)
        right_q_target = np.full(Inspire_Num_Motors, 1.0)

        # initialize inspire hand's cmd msg
        self.hand_msg = MotorCmds_()
        self.hand_msg.cmds = [
            unitree_go_msg_dds__MotorCmd_()
            for _ in range(len(Inspire_Right_Hand_JointIndex) + len(Inspire_Left_Hand_JointIndex))
        ]

        for idx, id in enumerate(Inspire_Left_Hand_JointIndex):
            self.hand_msg.cmds[id].q = 1.0
        for idx, id in enumerate(Inspire_Right_Hand_JointIndex):
            self.hand_msg.cmds[id].q = 1.0

        try:
            while self.running:
                start_time = time.time()

                # get dual hand state
                with left_hand_array.get_lock():
                    left_hand_mat = np.array(left_hand_array[:]).copy()
                with right_hand_array.get_lock():
                    right_hand_mat = np.array(right_hand_array[:]).copy()

                # Read left and right q_state from shared arrays
                state_data = np.concatenate((np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:])))

                action_data = np.concatenate((left_hand_mat, right_hand_mat))
                if dual_hand_data_lock is not None:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:] = state_data
                        dual_hand_action_array[:] = action_data

                if dual_hand_state_array and dual_hand_action_array:
                    with dual_hand_data_lock:
                        left_q_target = left_hand_mat
                        right_q_target = right_hand_mat

                self.ctrl_dual_hand(left_q_target, right_q_target)
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("Inspire_Controller has been closed.")


class Inspire_Controller_FTP:
    """Inspire FTP controller for policy-output hand actions.

    Different from teleop/XR pipeline, this controller expects normalized policy actions
    in [0, 1] (6 dof per hand), then scales them to [0, 1000] for FTP DDS messages.
    """

    def __init__(
        self,
        left_hand_array,
        right_hand_array,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
        fps=100.0,
        Unit_Test=False,
        simulation_mode=False,
    ):
        logger_mp.info("Initialize Inspire_Controller_FTP...")
        self.fps = fps
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode

        if self.simulation_mode:
            ChannelFactoryInitialize(1)
        else:
            ChannelFactoryInitialize(0)

        try:
            from inspire_sdkpy import inspire_dds
            import inspire_sdkpy.inspire_hand_defaut as inspire_hand_default
        except Exception as exc:
            raise ModuleNotFoundError(
                "Inspire FTP requires `inspire_sdkpy` (modules `inspire_dds` and "
                "`inspire_hand_defaut`). Install it in the runtime environment."
            ) from exc

        self._inspire_dds = inspire_dds
        self._get_inspire_hand_ctrl = inspire_hand_default.get_inspire_hand_ctrl

        self.LeftHandCmd_publisher = ChannelPublisher(
            kTopicInspireFTPLeftCommand, self._inspire_dds.inspire_hand_ctrl
        )
        self.LeftHandCmd_publisher.Init()
        self.RightHandCmd_publisher = ChannelPublisher(
            kTopicInspireFTPRightCommand, self._inspire_dds.inspire_hand_ctrl
        )
        self.RightHandCmd_publisher.Init()

        self.LeftHandState_subscriber = ChannelSubscriber(
            kTopicInspireFTPLeftState, self._inspire_dds.inspire_hand_state
        )
        self.LeftHandState_subscriber.Init()
        self.RightHandState_subscriber = ChannelSubscriber(
            kTopicInspireFTPRightState, self._inspire_dds.inspire_hand_state
        )
        self.RightHandState_subscriber.Init()

        self.left_hand_state_array = Array("d", Inspire_Num_Motors, lock=True)
        self.right_hand_state_array = Array("d", Inspire_Num_Motors, lock=True)

        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        wait_count = 0
        while not (any(self.left_hand_state_array) or any(self.right_hand_state_array)):
            if wait_count % 100 == 0:
                logger_mp.info("[Inspire_Controller_FTP] Waiting for FTP hand state...")
            time.sleep(0.01)
            wait_count += 1
            if wait_count > 500:
                logger_mp.warning(
                    "[Inspire_Controller_FTP] Timeout waiting initial state. Continue anyway."
                )
                break

        hand_control_process = Process(
            target=self.control_process,
            args=(
                left_hand_array,
                right_hand_array,
                self.left_hand_state_array,
                self.right_hand_state_array,
                dual_hand_data_lock,
                dual_hand_state_array,
                dual_hand_action_array,
            ),
        )
        hand_control_process.daemon = True
        hand_control_process.start()
        logger_mp.info("Initialize Inspire_Controller_FTP OK!\n")

    def _subscribe_hand_state(self):
        while True:
            left_state_msg = self.LeftHandState_subscriber.Read()
            if left_state_msg is not None and hasattr(left_state_msg, "angle_act"):
                if len(left_state_msg.angle_act) >= Inspire_Num_Motors:
                    with self.left_hand_state_array.get_lock():
                        for i in range(Inspire_Num_Motors):
                            self.left_hand_state_array[i] = left_state_msg.angle_act[i] / 1000.0

            right_state_msg = self.RightHandState_subscriber.Read()
            if right_state_msg is not None and hasattr(right_state_msg, "angle_act"):
                if len(right_state_msg.angle_act) >= Inspire_Num_Motors:
                    with self.right_hand_state_array.get_lock():
                        for i in range(Inspire_Num_Motors):
                            self.right_hand_state_array[i] = right_state_msg.angle_act[i] / 1000.0
            time.sleep(0.002)

    def _to_norm6(self, values):
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size < Inspire_Num_Motors:
            pad = np.full(Inspire_Num_Motors - arr.size, 1.0, dtype=np.float64)
            arr = np.concatenate([arr, pad], axis=0)
        elif arr.size > Inspire_Num_Motors:
            arr = arr[:Inspire_Num_Motors]
        return np.clip(arr, 0.0, 1.0)

    def _send_hand_command(self, left_q_target, right_q_target):
        left_cmd_msg = self._get_inspire_hand_ctrl()
        right_cmd_msg = self._get_inspire_hand_ctrl()
        left_cmd_msg.angle_set = [int(np.clip(v * 1000.0, 0.0, 1000.0)) for v in left_q_target]
        right_cmd_msg.angle_set = [int(np.clip(v * 1000.0, 0.0, 1000.0)) for v in right_q_target]
        left_cmd_msg.mode = 0b0001
        right_cmd_msg.mode = 0b0001
        self.LeftHandCmd_publisher.Write(left_cmd_msg)
        self.RightHandCmd_publisher.Write(right_cmd_msg)

    def control_process(
        self,
        left_hand_array,
        right_hand_array,
        left_hand_state_array,
        right_hand_state_array,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
    ):
        self.running = True
        left_q_target = np.full(Inspire_Num_Motors, 1.0)
        right_q_target = np.full(Inspire_Num_Motors, 1.0)

        try:
            while self.running:
                start_time = time.time()

                with left_hand_array.get_lock():
                    left_hand_mat = np.array(left_hand_array[:]).copy()
                with right_hand_array.get_lock():
                    right_hand_mat = np.array(right_hand_array[:]).copy()

                left_q_target = self._to_norm6(left_hand_mat)
                right_q_target = self._to_norm6(right_hand_mat)

                state_data = np.concatenate((np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:])))
                action_data = np.concatenate((left_q_target, right_q_target))
                if dual_hand_data_lock is not None:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:] = state_data
                        dual_hand_action_array[:] = action_data

                self._send_hand_command(left_q_target, right_q_target)
                time_elapsed = time.time() - start_time
                time.sleep(max(0, (1 / self.fps) - time_elapsed))
        finally:
            logger_mp.info("Inspire_Controller_FTP has been closed.")


# Update hand state, according to the official documentation, https://support.unitree.com/home/en/G1_developer/inspire_dfx_dexterous_hand
# the state sequence is as shown in the table below
# ┌──────┬───────┬──────┬────────┬────────┬────────────┬────────────────┬───────┬──────┬────────┬────────┬────────────┬────────────────┐
# │ Id   │   0   │  1   │   2    │   3    │     4      │       5        │   6   │  7   │   8    │   9    │    10      │       11       │
# ├──────┼───────┼──────┼────────┼────────┼────────────┼────────────────┼───────┼──────┼────────┼────────┼────────────┼────────────────┤
# │      │                    Right Hand                                │                   Left Hand                                  │
# │Joint │ pinky │ ring │ middle │ index  │ thumb-bend │ thumb-rotation │ pinky │ ring │ middle │ index  │ thumb-bend │ thumb-rotation │
# └──────┴───────┴──────┴────────┴────────┴────────────┴────────────────┴───────┴──────┴────────┴────────┴────────────┴────────────────┘
class Inspire_Right_Hand_JointIndex(IntEnum):
    kRightHandPinky = 0
    kRightHandRing = 1
    kRightHandMiddle = 2
    kRightHandIndex = 3
    kRightHandThumbBend = 4
    kRightHandThumbRotation = 5


class Inspire_Left_Hand_JointIndex(IntEnum):
    kLeftHandPinky = 6
    kLeftHandRing = 7
    kLeftHandMiddle = 8
    kLeftHandIndex = 9
    kLeftHandThumbBend = 10
    kLeftHandThumbRotation = 11
