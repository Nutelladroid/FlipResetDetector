#FlipResetDetector.py 
import time
import numpy as np 
import math

from rlbot.agents.base_script import BaseScript
from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator, GameInfoState
from rlbot.utils.structures.game_data_struct import GameTickPacket


class FlipResetDetector(BaseScript):
    def __init__(self):
        super().__init__("Flip Reset Detector")
        self.hasReset = False
        self.prev_canJump = False
        self.last_render_time = 0
        self.render_duration = 3  # Clear the text after 3 seconds
        self.game_interface.renderer.clear_screen()
        self.tick_skip = 8 #Check every 8 ticks to emulate a standard RLGym bot tick skip
        self.ticks = 0
    
    def _euler_to_rotation(self, pyr: np.ndarray):
        # Convert degrees to radians
        pitch = math.radians(pyr[0])
        yaw = math.radians(pyr[1])
        roll = math.radians(pyr[2])

        # Compute the rotation matrix
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        cr = math.cos(roll)
        sr = math.sin(roll)

        rotation_matrix = np.array([
            [cp * cy, cp * sy, sp],
            [sy * sr - cy * sp * cr, -cy * sr - sy * sp * cr, cp * cr],
            [-cy * sp * sr - sy * cr, sy * sp * sr - cy * cr, cp * sr]
        ])

        return rotation_matrix
        
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
        CAR_UNDER_THRESHOLD = 0
       
        while 1:
            # when packet available
            packet = self.wait_game_tick_packet()

            if not packet.game_info.is_round_active:
                continue
            
            self.ticks += 1
            if self.ticks < self.tick_skip:
                continue
            self.ticks = 0
            
            car_states = {}
            for p in range(packet.num_cars):
                car_id = p
                car = packet.game_cars[p]
                ball = packet.game_ball

                #Car rotation matrix
                pitch = car.physics.rotation.pitch
                yaw = car.physics.rotation.yaw
                roll = car.physics.rotation.roll

                # Create the rotation matrix
                car_rotation = np.array([pitch, yaw, roll])
                car_rotation = self._euler_to_rotation(car_rotation)
                car_forward = car_rotation[0]
                car_up = car_rotation[1]
                car_right = car_rotation[2]

                car_location = np.array([car.physics.location.x, car.physics.location.y, car.physics.location.z])
                
                ball_location = np.array([ball.physics.location.x, ball.physics.location.y, ball.physics.location.z])
                
                near_ball = np.linalg.norm(car_location - ball_location) < 170.0
                
                height_check = (car.physics.location.z < 300.0) or (car.physics.location.z > CEILING_Z - 300.0)
                
                # Check if the player is under the ball
                dir_to_ball = (ball_location - car_location) / np.linalg.norm(ball_location - car_location)
                
                under_ball = np.dot(car_up, dir_to_ball) > CAR_UNDER_THRESHOLD

 
                wall_dis_check = ((-SIDE_WALL_X + 700.0) > car.physics.location.x) or \
                                ((SIDE_WALL_X - 700.0) < car.physics.location.x) or \
                                ((-BACK_WALL_Y + 700.0) > car.physics.location.y) or \
                                ((BACK_WALL_Y - 700.0) < car.physics.location.y)
                
                on_ground = car.has_wheel_contact
                canJump = not car.jumped
                
                gotReset = self.prev_canJump < canJump
                
                has_flipped = car.double_jumped
                
                
                if(on_ground or has_flipped or car.jumped):
                    if(self.hasReset):
                        self.hasReset = False
                        print("Has Reset: False")
                
                if(near_ball and not height_check and not wall_dis_check):
                    
                    if(gotReset): #or on_ground
                        self.hasReset = True
                        print("Under Ball reset:", under_ball)
                        print("gotReset:",self.hasReset)
                        self.render_text("Under Ball reset: {}, Got Reset{}".format(under_ball, self.hasReset))
                        self.last_render_time = time.time()
                
                self.prev_canJump = canJump
                self.clear_text_if_expired()
                


if __name__ == "__main__":
    flip_detector = FlipResetDetector()
    flip_detector.start()
