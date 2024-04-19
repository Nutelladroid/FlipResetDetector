import time
import numpy as np
import math

from rlbot.agents.base_script import BaseScript
from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator, GameInfoState
from rlbot.utils.structures.game_data_struct import GameTickPacket

class FlipResetDetector(BaseScript):
    def __init__(self):
        super().__init__("Flip Reset Detector")
        self.hasReset = [False] * 8
        self.prev_canJump = [False] * 8
        self.last_render_time = 0
        self.render_duration = 3  # Clear the text after 3 seconds
        self.game_interface.renderer.clear_screen()
        self.tick_skip = 8  # Check every 8 ticks to emulate a standard RLGym bot tick skip
        self.ticks = 0
        self.JumpTimeCount = [0] * 8
        self.FlipTimeCount = [0] * 8
        self.AirTimeCount = [0] * 8
        self.prevFlipTimeCount = [0] * 8
        self.prevHasFlipped = [False] * 8
        self.notOnWall = [False] * 8
        self.prevPitchAngular_velocity = [0.0] * 8
        self.prevForwardVelocity = [0] * 8
        self.flipAcc = [0] * 8

    def _euler_to_rotation(self, pyr: np.ndarray):
        CP = math.cos(pyr[0])
        SP = math.sin(pyr[0])
        CY = math.cos(pyr[1])
        SY = math.sin(pyr[1])
        CR = math.cos(pyr[2])
        SR = math.sin(pyr[2])

        theta = np.empty((3, 3))

        # front direction
        theta[0, 0] = CP * CY
        theta[1, 0] = CP * SY
        theta[2, 0] = SP

        # left direction
        theta[0, 1] = CY * SP * SR - CR * SY
        theta[1, 1] = SY * SP * SR + CR * CY
        theta[2, 1] = -CP * SR

        # up direction
        theta[0, 2] = -CR * CY * SP - SR * SY
        theta[1, 2] = -CR * SY * SP + SR * CY
        theta[2, 2] = CP * CR

        return theta

    def render_text(self, text):
        self.game_interface.renderer.clear_screen()
        self.game_interface.renderer.begin_rendering()
        self.game_interface.renderer.draw_string_2d(20, 100, 1, 1, text, self.renderer.lime())
        self.game_interface.renderer.end_rendering()

    def clear_text_if_expired(self):
        if time.time() - self.last_render_time > self.render_duration:
            self.game_interface.renderer.begin_rendering()
            self.game_interface.renderer.draw_string_2d(20, 100, 1, 1, "", self.renderer.white())
            self.game_interface.renderer.end_rendering()

    def start(self):
        BALL_RADIUS = 92.75
        CEILING_Z = 2044
        SIDE_WALL_X = 4096
        BACK_WALL_Y = 5120
        CAR_UNDER_THRESHOLD = -0.6

        while 1:
            # when packet available
            packet = self.wait_game_tick_packet()

            if not packet.game_info.is_round_active:
                continue

            self.ticks += 1
            if self.ticks < self.tick_skip:
                continue
            self.ticks = 0

            for p in range(packet.num_cars):
                car = packet.game_cars[p]
                ball = packet.game_ball

                # Car velocity
                car_velocity = np.array([car.physics.velocity.x, car.physics.velocity.y, car.physics.velocity.z])

                # Car rotation matrix
                car_rotation = np.array([car.physics.rotation.pitch, -car.physics.rotation.yaw, -car.physics.rotation.roll])
                car_rotation = self._euler_to_rotation(car_rotation)
                car_forward = car_rotation[0]
                car_up = car_rotation[1]
                car_right = car_rotation[2]
                

                car_location = np.array([car.physics.location.x, car.physics.location.y, car.physics.location.z])
                ball_location = np.array([ball.physics.location.x, ball.physics.location.y, ball.physics.location.z])

                near_ball = np.linalg.norm(car_location - ball_location) < 170.0
                height_check = (car.physics.location.z < 300.0) or (car.physics.location.z > CEILING_Z - 300.0)
                dir_to_ball = (ball_location - car_location) / np.linalg.norm(ball_location - car_location)
                under_ball = np.dot(car_up, dir_to_ball)> CAR_UNDER_THRESHOLD
                wall_dis_check = ((-SIDE_WALL_X + 700.0) > car.physics.location.x) or \
                                ((SIDE_WALL_X - 700.0) < car.physics.location.x) or \
                                ((-BACK_WALL_Y + 700.0) > car.physics.location.y) or \
                                ((BACK_WALL_Y - 700.0) < car.physics.location.y)

                on_ground = car.has_wheel_contact
                canJump = not car.jumped
                gotReset = self.prev_canJump[p] < canJump
                has_flipped = car.double_jumped
                justFlipped = self.prevHasFlipped[p] < has_flipped
                car_Fvelocity = np.dot(car_forward, car_velocity)

                # Update counters
                if car.jumped:
                    self.JumpTimeCount[p] += 1
                if on_ground:
                    self.JumpTimeCount[p] = 0

                if has_flipped:
                    self.FlipTimeCount[p] += 1
                if on_ground:
                    self.FlipTimeCount[p] = 0

                if not on_ground:
                    self.AirTimeCount[p] += 1
                if on_ground:
                    self.AirTimeCount[p] = 0

                self.notOnWall[p] = car.physics.location.z < 100

                if on_ground or has_flipped or car.jumped:
                    if self.hasReset[p]:
                        self.hasReset[p] = False
                        print(f"Player {p}: Has Reset: False")

                if justFlipped:
                    self.flipAcc[p] = car_Fvelocity - self.prevForwardVelocity[p]

                if near_ball and not height_check and not wall_dis_check:
                    if gotReset:
                        self.hasReset[p] = True
                        print(f"Player {p}: Above Ball: {under_ball}, Got Reset: {self.hasReset[p]}")
                        self.render_text(f"Player {p}: Above Ball: {under_ball}, Got Reset: {self.hasReset[p]}")
                        self.last_render_time = time.time()

                if self.notOnWall[p] and on_ground and self.prevHasFlipped[p] and self.prevFlipTimeCount[p] < 15 and self.flipAcc[p] > 300:
                    print(f"Player {p}: Maybe forward wave dashed")

                self.prev_canJump[p] = canJump
                self.prevFlipTimeCount[p] = self.FlipTimeCount[p]
                self.prevHasFlipped[p] = has_flipped
                self.prevPitchAngular_velocity[p] = car.physics.angular_velocity.x
                self.clear_text_if_expired()
                self.prevForwardVelocity[p] = car_Fvelocity

if __name__ == "__main__":
    flip_detector = FlipResetDetector()
    flip_detector.start()
