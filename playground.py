from typing import Any, Dict

import gym
import numpy as np
import numpy.typing as npt
from gym import spaces
from lux.config import EnvConfig

class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:

        super().__init__(env)
        self.prev_step_metrics = dict()
        self.prev_step_metrics["player_0"] = None
        self.prev_step_metrics["player_1"] = None

    def step(self, action):
        # agent = "player_0"
        # opp_agent = "player_1"

        obs, reward, done, info = self.env.step(action)

        # we collect stats on teams here. These are useful stats that can be used to help generate reward functions
        for agent in self.env.agents:

            stats: StatsStateDict = self.env.state.stats[agent]

            info = dict()
            metrics = dict()
            metrics["ice_dug"] = (stats["generation"]["ice"]["HEAVY"] + stats["generation"]["ice"]["LIGHT"])
            metrics["water_produced"] = stats["generation"]["water"]

            # we save these two to see often the agent updates robot action queues and how often enough
            # power to do so and succeed (less frequent updates = more power is saved)
            metrics["action_queue_updates_success"] = stats["action_queue_updates_success"]
            metrics["action_queue_updates_total"] = stats["action_queue_updates_total"]

            # we can save the metrics to info so we can use tensorboard to log them to get a glimpse into how our agent is behaving
            info["metrics"] = metrics

            temp_reward = 0
            if self.prev_step_metrics[agent] is not None:
                # we check how much ice and water is produced and reward the agent for generating both
                ice_dug_this_step = metrics["ice_dug"] - self.prev_step_metrics[agent]["ice_dug"]
                water_produced_this_step = (
                        metrics["water_produced"] - self.prev_step_metrics[agent]["water_produced"]
                )
                # we reward water production more as it is the most important resource for survival
                temp_reward = ice_dug_this_step / 100 + water_produced_this_step

            self.prev_step_metrics[agent] = copy.deepcopy(metrics)
            reward[agent] = temp_reward
        return obs, reward, done, info





mapSize = EnvConfig.map_size
print(mapSize)

class PixelToPixelWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def get_observation_spec(self, mapSize = mapSize) -> spaces.Dict:
        x= mapSize
        y= mapSize
        obspace = spaces.Dict(
            {
                # Player and Location Specific
                "OwnLightRobot" : spaces.MultiBinary((1,x,y)),
                "OwnHeavyRobot" : spaces.MultiBinary((1,x,y)),
                "OppLightRobot" : spaces.MultiBinary((1,x,y)),
                "OppHeavyRobot" : spaces.MultiBinary((1,x,y)),
                "LightRobotPresent" : spaces.MultiBinary((1,x,y)),
                "HeavyRobotPresent" : spaces.MultiBinary((1,x,y)),
                "OwnLichenPresent" : spaces.MultiBinary((1,x,y)),
                "OppLichenPresent" : spaces.MultiBinary((1,x,y)),
                "NoLichenPresent" : spaces.MultiBinary((1,x,y)),
                #Normalised from 0 - 150
                "LightRobotPower": spaces.Box(0.,1., shape=(1,x,y), dtype=np.float32),
                #Normalised from 0- 3000
                "HeavyRobotPower": spaces.Box(0.,1., shape=(1,x,y), dtype=np.float32),
                #Normalised from 0 - 100
                "LichenInCellAmount": spaces.Box(0.,1., shape=(1,x,y), dtype=np.float32),

                #Light robot cargo capacity - 100, so normalised from 0 - 100
                "LightRobotIce":spaces.Box(0.,1., shape=(1,x,y), dtype=np.float32),
                "LightRobotMetal":spaces.Box(0.,1., shape=(1,x,y), dtype=np.float32),
                "LightRobotWater":spaces.Box(0.,1., shape=(1,x,y), dtype=np.float32),
                "LightRobotOre": spaces.Box(0., 1., shape=(1, x, y), dtype=np.float32),
                # Heavy robot cargo capacity - 1000, so normalised from 0 - 1000
                "HeavyRobotIce": spaces.Box(0., 1., shape=(1, x, y), dtype=np.float32),
                "HeavyRobotMetal": spaces.Box(0., 1., shape=(1, x, y), dtype=np.float32),
                "HeavyRobotWater": spaces.Box(0., 1., shape=(1, x, y), dtype=np.float32),
                "HeavyRobotOre": spaces.Box(0., 1., shape=(1, x, y), dtype=np.float32),
                "LightRobotCargoFull": spaces.MultiBinary((1,x,y)),
                "HeavyRobotCargoFull": spaces.MultiBinary((1,x,y)),
                # factory related
                "FactoryPresent": spaces.MultiBinary((1,x,y)),
                "OwnFactory": spaces.MultiBinary((1,x,y)),
                "OppFactory": spaces.MultiBinary((1,x,y)),
                #Normalised from 0 - 50000
                "FactoryPower": spaces.Box(0.,1., shape=(1,x,y), dtype=np.float32),
                "FactoryWater": spaces.Box(0.,1., shape=(1,x,y), dtype=np.float32),
                "FactoryMetal": spaces.Box(0.,1., shape=(1,x,y), dtype=np.float32),
                "FactoryLichen": spaces.Box(0.,1., shape=(1,x,y), dtype=np.float32),
                # normalised with mapsize 48
                # these features are for the robot to know where the closest resource is
                "ClosestIceTileXFromCurrentPosition": spaces.Box(0.,1., shape=(1,x,y), dtype=np.float32),
                "ClosestIceTileYFromCurrentPosition": spaces.Box(0.,1., shape=(1,x,y), dtype=np.float32),
                "ClosestMetalTileXFromCurrentPosition": spaces.Box(0.,1., shape=(1,x,y), dtype=np.float32),
                "ClosestMetalTileYFromCurrentPosition": spaces.Box(0.,1., shape=(1,x,y), dtype=np.float32),
                "ClosestFactoryXFromCurrentPosition": spaces.Box(0.,1., shape=(1,x,y), dtype=np.float32),
                "ClosestFactoryYFromCurrentPosition": spaces.Box(0.,1., shape=(1,x,y), dtype=np.float32),

                # Player specific location agnostic
                "OwnLightRobots": spaces.Box(0.,1., shape=(1,1), dtype=np.float32),
                "OwnHeavyRobots": spaces.Box(0.,1., shape=(1,1), dtype=np.float32),
                "OppLightRobots": spaces.Box(0.,1., shape=(1,1), dtype=np.float32),
                "OppHeavyRobots": spaces.Box(0.,1., shape=(1,1), dtype=np.float32),
                "OwnLichens": spaces.Box(0.,1., shape=(1,1), dtype=np.float32),
                "OppLichens": spaces.Box(0.,1., shape=(1,1), dtype=np.float32),
                "OwnFactories": spaces.Box(0.,1., shape=(1,1), dtype=np.float32),
                "OppFactories": spaces.Box(0.,1., shape=(1,1), dtype=np.float32),
                "OwnTotalWater": spaces.Box(0.,1., shape=(1,1), dtype=np.float32),
                "OppTotalWater": spaces.Box(0.,1., shape=(1,1), dtype=np.float32),

                # Player Agnostic
                "RawResourcePresent": spaces.MultiBinary((1, x, y)),
                "IceTile": spaces.MultiBinary((1, x, y)),
                "MetalTile": spaces.MultiBinary((1, x, y)),
                "RubbleAmount": spaces.Box(0.,1., shape=(1,x,y), dtype=np.float32),
                # True when it is night
                "Night": spaces.MultiDiscrete(np.zeros((1, 1)) + 2),
                # The turn number % 50
                "DayNightCycle": gym.spaces.Box(0., 1., shape=(1, 1)),
                # The turn number // 50
                "Phase": gym.spaces.MultiDiscrete(
                    np.zeros((1, 1)) + 20),
                # The turn number, normalized from 0-1000
                "Turn": gym.spaces.Box(0., 1., shape=(1, 1)),















            }
        )
    def observation(self, obs: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        return obs