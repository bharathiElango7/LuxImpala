from typing import Any, Dict

import gym
import numpy as np
import numpy.typing as npt
from gym import spaces
from lux.config import EnvConfig
from copy import Deepcopy

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
P=2
class PixelToPixelWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        #self.empty_obs = {}

    def get_observation_spec(self, mapSize = mapSize) -> spaces.Dict:
        x= mapSize
        y= mapSize
        obspace = spaces.Dict(
            {
                # Player and Location Specific
                "LightRobot" : spaces.MultiBinary((1,P,x,y)),
                "HeavyRobot" : spaces.MultiBinary((1,P,x,y)),
                "LichenPresent" : spaces.MultiBinary((1,P,x,y)),
                #Normalised from 0 - 150
                "LightRobotPower": spaces.Box(0.,1., shape=(1,P,x,y)),
                #Normalised from 0- 3000
                "HeavyRobotPower": spaces.Box(0.,1., shape=(1,P,x,y)),
                #Normalised from 0 - 100
                "LichenInCellAmount": spaces.Box(0.,1., shape=(1,P,x,y)),

                #Light robot cargo capacity - 100, so normalised from 0 - 100
                "LightRobotIce":spaces.Box(0.,1., shape=(1,P,x,y)),
                "LightRobotMetal":spaces.Box(0.,1., shape=(1,P,x,y)),
                "LightRobotWater":spaces.Box(0.,1., shape=(1,P,x,y)),
                "LightRobotOre": spaces.Box(0., 1., shape=(1,P,x,y)),
                # Heavy robot cargo capacity - 1000, so normalised from 0 - 1000
                "HeavyRobotIce": spaces.Box(0., 1., shape=(1,P,x,y)),
                "HeavyRobotMetal": spaces.Box(0., 1., shape=(1,P,x,y)),
                "HeavyRobotWater": spaces.Box(0., 1., shape=(1,P,x,y)),
                "HeavyRobotOre": spaces.Box(0., 1., shape=(1,P,x,y)),
                "LightRobotCargoFull": spaces.MultiBinary((1,P,x,y)),
                "HeavyRobotCargoFull": spaces.MultiBinary((1,P,x,y)),
                # factory related
                "FactoryPresent": spaces.MultiBinary((1,P,x,y)),
                #Normalised from 0 - 50000
                "FactoryPower": spaces.Box(0.,1., shape=(1,P,x,y)),
                "FactoryWater": spaces.Box(0.,1., shape=(1,P,x,y)),
                "FactoryMetal": spaces.Box(0.,1., shape=(1,P,x,y)),
                "FactoryLichen": spaces.Box(0.,1., shape=(1,P,x,y)),
                # normalised with mapsize 48
                # these features are for the robot to know where the closest resource is
                "ClosestIceTileXFromCurrentPosition": spaces.Box(0.,1., shape=(1,x,y)),
                "ClosestIceTileYFromCurrentPosition": spaces.Box(0.,1., shape=(1,x,y)),
                "ClosestOreTileXFromCurrentPosition": spaces.Box(0.,1., shape=(1,x,y)),
                "ClosestOreTileYFromCurrentPosition": spaces.Box(0.,1., shape=(1,x,y)),
                "ClosestFactoryXFromCurrentPosition": spaces.Box(0.,1., shape=(1,P,x,y)),
                "ClosestFactoryYFromCurrentPosition": spaces.Box(0.,1., shape=(1,P,x,y)),

                # Player specific location agnostic
                "TotalLightRobots": spaces.Box(0.,1., shape=(1,P,1)),
                "TotalHeavyRobots": spaces.Box(0.,1., shape=(1,P,1)),
                "TotalLichen": spaces.Box(0.,1., shape=(1,P,1)),
                "TotalFactories": spaces.Box(0.,1., shape=(1,P,1)),
                "TotalWater": spaces.Box(0.,1., shape=(1,P,1)),

                # Player Agnostic
                "IceTile": spaces.MultiBinary((1, x, y)),
                "OreTile": spaces.MultiBinary((1, x, y)),
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
        return obspace
    def observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:

        LightRobotCargoCapacity = 100
        HeavyRobotCargoCapacity = 1000
        LightRobotPowerCapacity = 150
        HeavyRobotPowerCapacity = 3000
        MaxRubble = 100
        MaxLichen = 100
        MapSize = 48
        NewObs = {}
        env_cfg = EnvConfig
        for key, spec in self.get_observation_spec(MapSize).spaces.items():
            if isinstance(spec, gym.spaces.MultiBinary) or isinstance(spec, gym.spaces.MultiDiscrete):
                NewObs[key] = np.zeros(spec.shape, dtype=np.int64)
            elif isinstance(spec, gym.spaces.Box):
                NewObs[key] = np.zeros(spec.shape, dtype=np.float32) + spec.low
            else:
                raise NotImplementedError(f"{type(spec)} is not an accepted observation space.")

        Agents = ["player_0", "player_1"]
        shared_obs = obs["player_0"]
        for agent in Agents:
            if agent == "player_0":
                P = 0
            else:
                P = 1

            units = shared_obs["units"][agent]
            # Robots
            for k in units.keys():
                unit = units[k]
                x,y = unit["pos"]
                cargo_space = env_cfg.ROBOTS[unit["unit_type"]].CARGO_SPACE
                battery_cap = env_cfg.ROBOTS[unit["unit_type"]].BATTERY_CAPACITY
                ResourceTotal = 0
                for resource in unit["cargo"].keys():
                    ResourceTotal += unit["cargo"][resource]
                if unit["unit_type"] == "LIGHT":
                    NewObs["LightRobot"][0][P][x][y] = 1
                    NewObs["LightRobotPower"][0][P][x][y] = unit["power"] / battery_cap
                    NewObs["LightRobotIce"][0][P][x][y] = unit["cargo"]["ice"] / cargo_space
                    NewObs["LightRobotOre"][0][P][x][y] = unit["cargo"]["ore"] / cargo_space
                    NewObs["LightRobotWater"][0][P][x][y] = unit["cargo"]["water"] / cargo_space
                    NewObs["LightRobotMetal"][0][P][x][y] = unit["cargo"]["metal"] / cargo_space
                    if ResourceTotal == cargo_space:
                        NewObs["LightRobotCargoFull"][0][P][x][y] = 1
                    NewObs["TotalLightRobots"][0][P][0] += 1 # have to normalise this later
                else:
                    NewObs["HeavyRobot"][0][P][x][y] = 1
                    NewObs["HeavyRobotPower"][0][P][x][y] = unit["power"] / battery_cap
                    NewObs["HeavyRobotIce"][0][P][x][y] = unit["cargo"]["ice"] / cargo_space
                    NewObs["HeavyRobotOre"][0][P][x][y] = unit["cargo"]["ore"] / cargo_space
                    NewObs["HeavyRobotWater"][0][P][x][y] = unit["cargo"]["water"] / cargo_space
                    NewObs["HeavyRobotMetal"][0][P][x][y] = unit["cargo"]["metal"] / cargo_space
                    if ResourceTotal == cargo_space:
                        NewObs["HeavyRobotCargoFull"][0][P][x][y] = 1
                    NewObs["TotalHeavyRobots"][0][P][0] += 1  # have to normalise this later

            # Factories
            factories = shared_obs["factories"][agent]
            factoryLocations = []
            for k in factories.keys():
                factory = factories[k]
                x,y = factory["pos"]
                factoryLocations.append([x,y])
                NewObs["FactoryPresent"][0][P][x][y] = 1
                # have to normaise the following
                NewObs["FactoryIce"][0][P][x][y] = factory["cargo"]["ice"]
                NewObs["FactoryOre"][0][P][x][y] = factory["cargo"]["ore"]
                NewObs["FactoryWater"][0][P][x][y] = factory["cargo"]["water"]
                NewObs["FactoryMetal"][0][P][x][y] = factory["cargo"]["metal"]
                NewObs["FactoryPower"][0][P][x][y] = factory["power"]
                # have to find out how much lichen this factory has
                strainId = factory["strain_id"]
                LichenMap = shared_obs["board"]["lichen"]
                LichenStrainMap = shared_obs["board"]["lichen_strains"]
                Factory_Strain_locations = np.argwhere(LichenStrainMap == strainId)
                for m,n in Factory_Strain_locations:
                    NewObs["FactoryLichen"][0][P][x][y] += LichenMap[m][n]
                    NewObs["TotalLichen"][0][P][0] += LichenMap[m][n]
                    NewObs["LichenPresent"][0][P][m][n] = 1
                    NewObs["LichenInCellAmount"][0][P][m][n] = LichenMap[m][n] / MaxLichen

                NewObs["TotalFactories"][0][P][0] += 1

            factoryLocations = np.array(factoryLocations)

            for m in range(MapSize):
                for n in range(MapSize):
                    pos = np.array([m, n]) / MapSize
                    factory_distances = np.mean((factoryLocations - np.array(m, n)) ** 2, 1)
                    # normalize the ice tile location
                    closest_factory = (factoryLocations[np.argmin(factory_distances)] / MapSize)
                    NewObs["ClosestFactoryXFromCurrentPosition"][0][P][m][n] = abs(closest_factory[0] - pos[0])
                    NewObs["ClosestFactoryYFromCurrentPosition"][0][P][m][n] = abs(closest_factory[1] - pos[1])

            NewObs["TotalWater"][0][P][0] = shared_obs["teams"][agent]["water"]

        # Player Agnostic
        # Rubble
        RubbleMap = shared_obs["board"]["rubble"] / MaxRubble
        NewObs["RubbleAmount"][0] = Deepcopy(RubbleMap)
        ice_map = shared_obs["board"]["ice"]
        NewObs['IceTile'][0] = Deepcopy(ice_map)
        ice_tile_locations = np.argwhere(ice_map == 1)
        ore_map = shared_obs["board"]["ore"]
        NewObs['OreTile'][0] = Deepcopy(ore_map)
        ore_tile_locations = np.argwhere(ore_map == 1)
        # compute closest ice tile
        for m in range(MapSize):
            for n in range(MapSize):
                pos = np.array([m, n]) / MapSize
                ice_tile_distances = np.mean((ice_tile_locations - np.array(m,n)) ** 2, 1)
        # normalize the ice tile location
                closest_ice_tile = (ice_tile_locations[np.argmin(ice_tile_distances)] / MapSize)

                NewObs["ClosestIceTileXFromCurrentPosition"][0][m][n] = abs(closest_ice_tile[0] - pos[0])
                NewObs["ClosestIceTileYFromCurrentPosition"][0][m][n] = abs(closest_ice_tile[1] - pos[1])

                ore_tile_distances = np.mean((ore_tile_locations - np.array(m, n)) ** 2, 1)
                # normalize the ore tile location
                closest_ore_tile = (ore_tile_locations[np.argmin(ore_tile_distances)] / MapSize)
                NewObs["ClosestOreTileXFromCurrentPosition"][0][m][n] = abs(closest_ore_tile[0] - pos[0])
                NewObs["ClosestOreTileYFromCurrentPosition"][0][m][n] = abs(closest_ore_tile[1] - pos[1])

        #check if night
        envstep = shared_obs["real_env_steps"]
        if envstep < 0:
            envstep = 0
        phase = int((envstep/50))
        NewObs["Phase"][0] = phase
        # normalised turn
        turn = envstep/1000
        NewObs["Turn"][0] = turn
        DaynightCycle = envstep%50
        NewObs["DaynightCycle"][0] = DaynightCycle
        if DaynightCycle >= 30:
            NewObs["Night"][0] = 1

        return obs


def get_observation_spec2(mapSize = mapSize) -> spaces.Dict:
    x= mapSize
    y= mapSize
    obspace = spaces.Dict(
            {
                # Player and Location Specific
                "OwnLightRobot" : spaces.MultiBinary((1,x,y)),
                #Normalised from 0 - 100
                "LichenInCellAmount": spaces.Box(0.,1., shape=(1,x,y), dtype=np.float32),
                "Night": spaces.MultiDiscrete(np.zeros((1, 1)) + 2)

            }
        )
    return obspace


print(type(get_observation_spec2(4)))
tempobs = {}
for key, spec in get_observation_spec2(3).spaces.items():
    if isinstance(spec, gym.spaces.MultiBinary) or isinstance(spec, gym.spaces.MultiDiscrete):
        tempobs[key] = np.zeros(spec.shape, dtype=np.int64)
    elif isinstance(spec, gym.spaces.Box):
        tempobs[key] = np.zeros(spec.shape, dtype=np.float32) + spec.low
    else:
        raise NotImplementedError(f"{type(spec)} is not an accepted observation space.")

print(tempobs)