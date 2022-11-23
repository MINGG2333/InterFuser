import os
import json
import datetime
import pathlib
import time
import imp
import cv2
import carla
from collections import deque

import torch
import carla
import numpy as np
from PIL import Image
from easydict import EasyDict
from team_code.planner import RoutePlanner

from torchvision import transforms
from leaderboard.autoagents import autonomous_agent
from timm.models import create_model
from interfuser_team_code.utils import lidar_to_histogram_features, transform_2d_points
from interfuser_team_code.interfuser_controller import InterfuserController
from interfuser_team_code.render import render, render_self_car, render_waypoints
from interfuser_team_code.tracker import Tracker

import math
import yaml

try:
    import pygame
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")


SAVE_PATH = os.environ.get("SAVE_PATH", 'eval')
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class DisplayInterface(object):
    def __init__(self):
        self._width = 1200
        self._height = 600
        self._surface = None

        pygame.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        self._display = pygame.display.set_mode(
            (self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("Human Agent")

    def run_interface(self, input_data):
        rgb = input_data['rgb']
        rgb_left = input_data['rgb_left']
        rgb_right = input_data['rgb_right']
        rgb_focus = input_data['rgb_focus']
        map = input_data['map']
        surface = np.zeros((600, 1200, 3),np.uint8)
        surface[:, :800] = rgb
        surface[:400,800:1200] = map
        surface[400:600,800:1000] = input_data['map_t1']
        surface[400:600,1000:1200] = input_data['map_t2']
        surface[:150,:200] = input_data['rgb_left']
        surface[:150, 600:800] = input_data['rgb_right']
        surface[:150, 325:475] = input_data['rgb_focus']
        surface = cv2.putText(surface, input_data['control'], (20,580), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255), 1)
        surface = cv2.putText(surface, input_data['meta_infos'][0], (20,560), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255), 1)
        surface = cv2.putText(surface, input_data['meta_infos'][1], (20,540), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255), 1)
        surface = cv2.putText(surface, input_data['time'], (20,520), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255), 1)

        surface = cv2.putText(surface, 'Left  View', (40,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)
        surface = cv2.putText(surface, 'Focus View', (335,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)
        surface = cv2.putText(surface, 'Right View', (640,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)

        surface = cv2.putText(surface, 'Future Prediction', (940,420), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 2)
        surface = cv2.putText(surface, 't', (1160,385), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0), 2)
        surface = cv2.putText(surface, '0', (1170,385), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 2)
        surface = cv2.putText(surface, 't', (960,585), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0), 2)
        surface = cv2.putText(surface, '1', (970,585), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 2)
        surface = cv2.putText(surface, 't', (1160,585), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0), 2)
        surface = cv2.putText(surface, '2', (1170,585), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0), 2)

        surface[:150,198:202]=0
        surface[:150,323:327]=0
        surface[:150,473:477]=0
        surface[:150,598:602]=0
        surface[148:152, :200] = 0
        surface[148:152, 325:475] = 0
        surface[148:152, 600:800] = 0
        surface[430:600, 998:1000] = 255
        surface[0:600, 798:800] = 255
        surface[0:600, 1198:1200] = 255
        surface[0:2, 800:1200] = 255
        surface[598:600, 800:1200] = 255
        surface[398:400, 800:1200] = 255


        # display image
        self._surface = pygame.surfarray.make_surface(surface.swapaxes(0, 1))
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))

        pygame.display.flip()
        pygame.event.get()
        return surface

    def _quit(self):
        pygame.quit()

# jxy: addition; (add display.py and fix RoutePlanner.py)
from team_code.display import HAS_DISPLAY, Saver, debug_display
# addition from team_code/map_agent.py
from carla_project.src.common import CONVERTER, COLOR
from carla_project.src.carla_env import draw_traffic_lights, get_nearby_lights


def get_entry_point():
    return "InterfuserAgent"


class Resize2FixedSize:
    def __init__(self, size):
        self.size = size

    def __call__(self, pil_img):
        pil_img = pil_img.resize(self.size)
        return pil_img


def create_carla_rgb_transform(
    input_size, need_scale=True, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
):

    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size
    tfl = []

    if isinstance(input_size, (tuple, list)):
        input_size_num = input_size[-1]
    else:
        input_size_num = input_size

    if need_scale:
        if input_size_num == 112:
            tfl.append(Resize2FixedSize((170, 128)))
        elif input_size_num == 128:
            tfl.append(Resize2FixedSize((195, 146)))
        elif input_size_num == 224:
            tfl.append(Resize2FixedSize((341, 256)))
        elif input_size_num == 256:
            tfl.append(Resize2FixedSize((288, 288)))
        else:
            raise ValueError("Can't find proper crop size")
    tfl.append(transforms.CenterCrop(img_size))
    tfl.append(transforms.ToTensor())
    tfl.append(transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)))

    return transforms.Compose(tfl)


class InterfuserAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        self.config_path = path_to_conf_file
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False

        return AgentSaver

        # jxy: add return AgentSaver and init_ads (setup keep 5 lines); rm save_path;
    def init_ads(self, path_to_conf_file):

        if HAS_DISPLAY: # jxy: change
            self._hic = DisplayInterface()
        self.lidar_processed = list()
        self.rgb_front_transform = create_carla_rgb_transform(224)
        self.rgb_left_transform = create_carla_rgb_transform(128)
        self.rgb_right_transform = create_carla_rgb_transform(128)
        self.rgb_center_transform = create_carla_rgb_transform(128, need_scale=False)

        self.tracker = Tracker()

        self.input_buffer = {
            "rgb": deque(),
            "rgb_left": deque(),
            "rgb_right": deque(),
            "rgb_rear": deque(),
            "lidar": deque(),
            "gps": deque(),
            "thetas": deque(),
        }

        self.config = imp.load_source("MainModel", path_to_conf_file).GlobalConfig()
        self.skip_frames = self.config.skip_frames
        self.controller = InterfuserController(self.config)
        if isinstance(self.config.model, list):
            self.ensemble = True
        else:
            self.ensemble = False

        if self.ensemble:
            for i in range(len(self.config.model)):
                self.nets = []
                net = create_model(self.config.model[i])
                path_to_model_file = self.config.model_path[i]
                print('load model: %s' % path_to_model_file)
                net.load_state_dict(torch.load(path_to_model_file)["state_dict"])
                net.cuda()
                net.eval()
                self.nets.append(net)
        else:
            self.net = create_model(self.config.model)
            path_to_model_file = self.config.model_path
            print('load model: %s' % path_to_model_file)
            self.net.load_state_dict(torch.load(path_to_model_file)["state_dict"])
            self.net.cuda()
            self.net.eval()
        self.softmax = torch.nn.Softmax(dim=1)
        self.traffic_meta_moving_avg = np.zeros((400, 7))
        self.momentum = self.config.momentum
        self.prev_lidar = None
        self.prev_control = None
        self.prev_surround_map = None

        # self.save_path = None
        # if SAVE_PATH is not None:
        #     now = datetime.datetime.now()
        #     string = pathlib.Path(os.environ["ROUTES"]).stem + "_"
        #     string += "_".join(
        #         map(
        #             lambda x: "%02d" % x,
        #             (now.month, now.day, now.hour, now.minute, now.second),
        #         )
        #     )

        #     print(string)

        #     self.save_path = pathlib.Path(SAVE_PATH) / string
        #     self.save_path.mkdir(parents=True, exist_ok=False)
        #     (self.save_path / "meta").mkdir(parents=True, exist_ok=False)

    def _init(self):
        self._route_planner = RoutePlanner(4.0, 50.0)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True

        super()._init() # jxy add

    def _get_position(self, tick_data):
        gps = tick_data["gps"]
        gps = (gps - self._route_planner.mean) * self._route_planner.scale
        return gps

    def sensors(self):
        return [
            {
                "type": "sensor.camera.rgb",
                "x": 1.3,
                "y": 0.0,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "width": 800,
                "height": 600,
                "fov": 100,
                "id": "rgb",
            },
            {
                "type": "sensor.camera.rgb",
                "x": 1.3,
                "y": 0.0,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -60.0,
                "width": 400,
                "height": 300,
                "fov": 100,
                "id": "rgb_left",
            },
            {
                "type": "sensor.camera.rgb",
                "x": 1.3,
                "y": 0.0,
                "z": 2.3,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 60.0,
                "width": 400,
                "height": 300,
                "fov": 100,
                "id": "rgb_right",
            },
            {
                "type": "sensor.lidar.ray_cast",
                "x": 1.3,
                "y": 0.0,
                "z": 2.5,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -90.0,
                "id": "lidar",
            },
            {
                "type": "sensor.other.imu",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.05,
                "id": "imu",
            },
            {
                "type": "sensor.other.gnss",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.01,
                "id": "gps",
            },
            {"type": "sensor.speedometer", "reading_frequency": 20, "id": "speed"},
            # jxy: addition from team_code/map_agent.py
            {
                'type': 'sensor.camera.semantic_segmentation',
                'x': 0.0, 'y': 0.0, 'z': 100.0,
                'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                'width': 512, 'height': 512, 'fov': 5 * 10.0,
                'id': 'map'
                },
    ]

    def tick(self, input_data):

        rgb = cv2.cvtColor(input_data["rgb"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_left = cv2.cvtColor(input_data["rgb_left"][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_right = cv2.cvtColor(
            input_data["rgb_right"][1][:, :, :3], cv2.COLOR_BGR2RGB
        )
        gps = input_data["gps"][1][:2]
        speed = input_data["speed"][1]["speed"]
        compass = input_data["imu"][1][-1]
        if (
            math.isnan(compass) == True
        ):  # It can happen that the compass sends nan for a few frames
            compass = 0.0

        result = {
            "rgb": rgb,
            "rgb_left": rgb_left,
            "rgb_right": rgb_right,
            "gps": gps,
            "speed": speed,
            "compass": compass,
        }

        pos = self._get_position(result)

        lidar_data = input_data['lidar'][1]
        result['raw_lidar'] = lidar_data

        lidar_unprocessed = lidar_data[:, :3]
        lidar_unprocessed[:, 1] *= -1
        full_lidar = transform_2d_points(
            lidar_unprocessed,
            np.pi / 2 - compass,
            -pos[0],
            -pos[1],
            np.pi / 2 - compass,
            -pos[0],
            -pos[1],
        )
        lidar_processed = lidar_to_histogram_features(full_lidar, crop=224)
        if self.step % 2 == 0 or self.step < 4:
            self.prev_lidar = lidar_processed
        result["lidar"] = self.prev_lidar

        result["gps"] = pos
        next_wp, next_cmd = self._route_planner.run_step(pos)
        result["next_command"] = next_cmd.value
        result['measurements'] = [pos[0], pos[1], compass, speed]

        theta = compass + np.pi / 2
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        local_command_point = np.array([next_wp[0] - pos[0], next_wp[1] - pos[1]])
        local_command_point = R.T.dot(local_command_point)
        result["target_point"] = local_command_point

        # jxy addition:
        result['far_command'] = next_cmd

        result['R_pos_from_head'] = R
        result['offset_pos'] = np.array([pos[0], pos[1]])
        # from team_code/map_agent.py:
        self._actors = self._world.get_actors()
        self._traffic_lights = get_nearby_lights(self._vehicle, self._actors.filter('*traffic_light*'))
        topdown = input_data['map'][1][:, :, 2]
        topdown = draw_traffic_lights(topdown, self._vehicle, self._traffic_lights)
        result['topdown'] = COLOR[CONVERTER[topdown]]
        return result

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        self.step += 1
        if self.step % self.skip_frames != 0 and self.step > 4:
            return self.prev_control

        tick_data = self.tick(input_data)

        velocity = tick_data["speed"]
        command = tick_data["next_command"]

        rgb = (
            self.rgb_front_transform(Image.fromarray(tick_data["rgb"]))
            .unsqueeze(0)
            .cuda()
            .float()
        )
        rgb_left = (
            self.rgb_left_transform(Image.fromarray(tick_data["rgb_left"]))
            .unsqueeze(0)
            .cuda()
            .float()
        )
        rgb_right = (
            self.rgb_right_transform(Image.fromarray(tick_data["rgb_right"]))
            .unsqueeze(0)
            .cuda()
            .float()
        )
        rgb_center = (
            self.rgb_center_transform(Image.fromarray(tick_data["rgb"]))
            .unsqueeze(0)
            .cuda()
            .float()
        )

        cmd_one_hot = [0, 0, 0, 0, 0, 0]
        cmd = command - 1
        cmd_one_hot[cmd] = 1
        cmd_one_hot.append(velocity)
        mes = np.array(cmd_one_hot)
        mes = torch.from_numpy(mes).float().unsqueeze(0).cuda()

        input_data = {}
        input_data["rgb"] = rgb
        input_data["rgb_left"] = rgb_left
        input_data["rgb_right"] = rgb_right
        input_data["rgb_center"] = rgb_center
        input_data["measurements"] = mes
        input_data["target_point"] = (
            torch.from_numpy(tick_data["target_point"]).float().cuda().view(1, -1)
        )
        input_data["lidar"] = (
            torch.from_numpy(tick_data["lidar"]).float().cuda().unsqueeze(0)
        )
        if self.ensemble:
            outputs = []
            with torch.no_grad():
                for net in self.nets:
                    output = net(input_data)
                    outputs.append(output)
            traffic_meta = torch.mean(torch.stack([x[0] for x in outputs]), 0)
            pred_waypoints = torch.mean(torch.stack([x[1] for x in outputs]), 0)
            is_junction = torch.mean(torch.stack([x[2] for x in outputs]), 0)
            traffic_light_state = torch.mean(torch.stack([x[3] for x in outputs]), 0)
            stop_sign = torch.mean(torch.stack([x[4] for x in outputs]), 0)
            bev_feature = torch.mean(torch.stack([x[5] for x in outputs]), 0)
        else:
            with torch.no_grad():
                (
                    traffic_meta,
                    pred_waypoints,
                    is_junction,
                    traffic_light_state,
                    stop_sign,
                    bev_feature,
                ) = self.net(input_data)
        traffic_meta = traffic_meta.detach().cpu().numpy()[0]
        bev_feature = bev_feature.detach().cpu().numpy()[0]
        pred_waypoints = pred_waypoints.detach().cpu().numpy()[0]
        is_junction = self.softmax(is_junction).detach().cpu().numpy().reshape(-1)[0]
        traffic_light_state = (
            self.softmax(traffic_light_state).detach().cpu().numpy().reshape(-1)[0]
        )
        stop_sign = self.softmax(stop_sign).detach().cpu().numpy().reshape(-1)[0]


        if self.step % 2 == 0 or self.step < 4:
            traffic_meta = self.tracker.update_and_predict(traffic_meta.reshape(20, 20, -1), tick_data['gps'], tick_data['compass'], self.step // 2)
            traffic_meta = traffic_meta.reshape(400, -1)
            self.traffic_meta_moving_avg = (
                self.momentum * self.traffic_meta_moving_avg
                + (1 - self.momentum) * traffic_meta
            )
        traffic_meta = self.traffic_meta_moving_avg

        tick_data["raw"] = traffic_meta
        tick_data["bev_feature"] = bev_feature

        # jxy: points_world
        steer, throttle, brake, meta_infos, pid_metadata, points_world = self.controller.run_step(
            velocity,
            pred_waypoints,
            is_junction,
            traffic_light_state,
            stop_sign,
            self.traffic_meta_moving_avg,
        )
        self.pid_metadata = pid_metadata

        if brake < 0.05:
            brake = 0.0
        if brake > 0.1:
            throttle = 0.0

        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)

        surround_map, box_info = render(traffic_meta.reshape(20, 20, 7), pixels_per_meter=20)
        surround_map = surround_map[:400, 160:560]
        surround_map = np.stack([surround_map, surround_map, surround_map], 2)

        self_car_map = render_self_car(
            loc=np.array([0, 0]),
            ori=np.array([0, -1]),
            box=np.array([2.45, 1.0]),
            color=[1, 1, 0], pixels_per_meter=20
        )[:400, 160:560]

        pred_waypoints = pred_waypoints.reshape(-1, 2)
        safe_index = 10
        for i in range(10):
            if pred_waypoints[i, 0] ** 2 + pred_waypoints[i, 1] ** 2> (meta_infos[3]+0.5) ** 2:
                safe_index = i
                break
        wp1 = render_waypoints(pred_waypoints[:safe_index], pixels_per_meter=20, color=(0, 255, 0))[:400, 160:560]
        wp2 = render_waypoints(pred_waypoints[safe_index:], pixels_per_meter=20, color=(255, 0, 0))[:400, 160:560]
        wp = wp1 + wp2

        surround_map = np.clip(
            (
                surround_map.astype(np.float32)
                + self_car_map.astype(np.float32)
                + wp.astype(np.float32)
            ),
            0,
            255,
        ).astype(np.uint8)

        map_t1, box_info = render(traffic_meta.reshape(20, 20, 7), pixels_per_meter=20, t=1)
        map_t1 = map_t1[:400, 160:560]
        map_t1 = np.stack([map_t1, map_t1, map_t1], 2)
        map_t1 = np.clip(map_t1.astype(np.float32) + self_car_map.astype(np.float32), 0, 255).astype(np.uint8)
        map_t1 = cv2.resize(map_t1, (200, 200))
        map_t2, box_info = render(traffic_meta.reshape(20, 20, 7), pixels_per_meter=20, t=2)
        map_t2 = map_t2[:400, 160:560]
        map_t2 = np.stack([map_t2, map_t2, map_t2], 2)
        map_t2 = np.clip(map_t2.astype(np.float32) + self_car_map.astype(np.float32), 0, 255).astype(np.uint8)
        map_t2 = cv2.resize(map_t2, (200, 200))


        if self.step % 2 != 0 and self.step > 4:
            control = self.prev_control
        else:
            self.prev_control = control
            self.prev_surround_map = surround_map

        tick_data["map"] = self.prev_surround_map
        tick_data["map_t1"] = map_t1
        tick_data["map_t2"] = map_t2
        tick_data["rgb_raw"] = tick_data["rgb"]
        tick_data["rgb_left_raw"] = tick_data["rgb_left"]
        tick_data["rgb_right_raw"] = tick_data["rgb_right"]

        tick_data["rgb"] = cv2.resize(tick_data["rgb"], (800, 600))
        tick_data["rgb_left"] = cv2.resize(tick_data["rgb_left"], (200, 150))
        tick_data["rgb_right"] = cv2.resize(tick_data["rgb_right"], (200, 150))
        tick_data["rgb_focus"] = cv2.resize(tick_data["rgb_raw"][244:356, 344:456], (150, 150))
        tick_data["control"] = "throttle: %.2f, steer: %.2f, brake: %.2f" % (
            control.throttle,
            control.steer,
            control.brake,
        )
        tick_data["meta_infos"] = meta_infos
        tick_data["box_info"] = "car: %d, bike: %d, pedestrian: %d" % (
            box_info["car"],
            box_info["bike"],
            box_info["pedestrian"],
        )
        tick_data["mes"] = "speed: %.2f" % velocity
        tick_data["time"] = "time: %.3f" % timestamp

        if HAS_DISPLAY: # jxy: change
            surface = self._hic.run_interface(tick_data)
            tick_data["surface"] = surface
            # debug_display(tick_data, control.steer, control.throttle, control.brake, self.step)

        self.record_step(tick_data, control, points_world) # jxy: add
        return control

    # jxy: add record_step
    def record_step(self, tick_data, control, pred_waypoint=[]):
        # draw pred_waypoint
        if len(pred_waypoint):
            # pred_waypoint[:,1] *= -1
            pred_waypoint = tick_data['R_pos_from_head'].dot(pred_waypoint.T).T
        self._route_planner.run_step2(pred_waypoint, is_gps=False, store=False) # metadata['wp_1'] relative to ego head (as y)
        # addition: from leaderboard/team_code/auto_pilot.py
        speed = tick_data['speed']
        self._recorder_tick(control) # trjs
        ego_bbox = self.gather_info() # metrics
        self._route_planner.run_step2(ego_bbox + tick_data['offset_pos'], is_gps=True, store=False)
        self._route_planner.show_route()

        if self.save_path is not None and self.step % self.record_every_n_step == 0:
            self.save(control.steer, control.throttle, control.brake, tick_data)


# jxy: mv save in AgentSaver & rm destroy
class AgentSaver(Saver):
    def __init__(self, path_to_conf_file, dict_, list_):
        self.config_path = path_to_conf_file

        # jxy: according to sensor
        self.rgb_list = ['rgb_raw', 'rgb_left_raw', 'rgb_right_raw', 'topdown', ] # 'bev', 
        self.add_img = [] # 'flow', 'out', 
        self.lidar_list = [] # 'lidar_0', 'lidar_1',
        self.dir_names = self.rgb_list + self.add_img + self.lidar_list + ['pid_metadata']

        super().__init__(dict_, list_)

    def run(self): # jxy: according to init_ads
        path_to_conf_file = self.config_path
        self.config = imp.load_source("MainModel", path_to_conf_file).GlobalConfig()
        self.skip_frames = self.config.skip_frames

        super().run()

    def _save(self, tick_data):    
        # addition
        # save_action_based_measurements = tick_data['save_action_based_measurements']
        self.save_path = tick_data['save_path']
        if not (self.save_path / 'ADS_log.csv' ).exists():
            # addition: generate dir for every total_i
            self.save_path.mkdir(parents=True, exist_ok=True)
            for dir_name in self.dir_names:
                (self.save_path / dir_name).mkdir(parents=True, exist_ok=False)

            # according to self.save data_row_list
            title_row = ','.join(
                ['frame_id', 'far_command', 'speed', 'steering', 'throttle', 'brake',] + \
                self.dir_names
            )
            with (self.save_path / 'ADS_log.csv' ).open("a") as f_out:
                f_out.write(title_row+'\n')

        self.step = tick_data['frame']
        self.save(tick_data['steer'],tick_data['throttle'],tick_data['brake'], tick_data)

    # addition: modified from leaderboard/team_code/auto_pilot.py
    def save(self, steer, throttle, brake, tick_data):
        # frame = self.step // 10
        frame = self.step

        # 'gps' 'thetas'
        pos = tick_data['gps']
        speed = tick_data['speed']
        far_command = tick_data['far_command']
        data_row_list = [frame, far_command.name, speed, steer, throttle, brake,]

        if frame >= self.skip_frames: # jxy: according to run_step
            # images
            for rgb_name in self.rgb_list + self.add_img:
                path_ = self.save_path / rgb_name / ('%04d.png' % frame)
                Image.fromarray(tick_data[rgb_name]).save(path_)
                data_row_list.append(str(path_))
            # lidar
            for i, rgb_name in enumerate(self.lidar_list):
                path_ = self.save_path / rgb_name / ('%04d.png' % frame)
                Image.fromarray(cm.gist_earth(tick_data['lidar_processed'][0][0, i], bytes=True)).save(path_)
                data_row_list.append(str(path_))

            # pid_metadata
            pid_metadata = tick_data['pid_metadata']
            path_ = self.save_path / 'pid_metadata' / ('%04d.json' % frame)
            outfile = open(path_, 'w')
            json.dump(pid_metadata, outfile, indent=4)
            outfile.close()
            data_row_list.append(str(path_))

        # collection
        data_row = ','.join([str(i) for i in data_row_list])
        with (self.save_path / 'ADS_log.csv' ).open("a") as f_out:
            f_out.write(data_row+'\n')



    # def save(self, tick_data):
    #     frame = self.step // self.skip_frames
    #     Image.fromarray(tick_data["surface"]).save(
    #         self.save_path / "meta" / ("%04d.jpg" % frame)
    #     )
    #     return

    # def destroy(self):
    #     if self.ensemble:
    #         del self.nets
    #     else:
    #         del self.net
