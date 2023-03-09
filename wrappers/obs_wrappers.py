from typing import Any, Dict

import gym
import numpy as np
import numpy.typing as npt
from gym import spaces


from lux.config import EnvConfig




mapSize = EnvConfig().map_size
print(mapSize)
P=2
class SimpleUnitObservationWrapper(gym.ObservationWrapper):
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
        env_cfg = EnvConfig()
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

        return NewObs











class SimpleUnitObservationWrapper2(gym.ObservationWrapper):
    """
    A simple state based observation to work with in pair with the SimpleUnitDiscreteController

    It contains info only on the first robot, the first factory you own, and some useful features. If there are no owned robots the observation is just zero.
    No information about the opponent is included. This will generate observations for all teams.

    Included features:
    - First robot's stats
    - distance vector to closest ice tile
    - distance vector to first factory

    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        #self.observation_space = spaces.Box(-999, 999, shape=(13,))


    def observation(self, obs):
        return SimpleUnitObservationWrapper.convert_obs(obs, self.env.state.env_cfg)

    # we make this method static so the submission/evaluation code can use this as well
    @staticmethod
    def convert_obs(obs: Dict[str, Any], env_cfg: Any) -> Dict[str, npt.NDArray]:
        observation = dict()
        shared_obs = obs["player_0"]
        ice_map = shared_obs["board"]["ice"]
        ice_tile_locations = np.argwhere(ice_map == 1)

        for agent in obs.keys():
            obs_vec = np.zeros(
                13,
            )

            factories = shared_obs["factories"][agent]
            factory_vec = np.zeros(2)
            for k in factories.keys():
                # here we track a normalized position of the first friendly factory
                factory = factories[k]
                factory_vec = np.array(factory["pos"]) / env_cfg.map_size
                break
            units = shared_obs["units"][agent]
            for k in units.keys():
                unit = units[k]

                # store cargo+power values scaled to [0, 1]
                cargo_space = env_cfg.ROBOTS[unit["unit_type"]].CARGO_SPACE
                battery_cap = env_cfg.ROBOTS[unit["unit_type"]].BATTERY_CAPACITY
                cargo_vec = np.array(
                    [
                        unit["power"] / battery_cap,
                        unit["cargo"]["ice"] / cargo_space,
                        unit["cargo"]["ore"] / cargo_space,
                        unit["cargo"]["water"] / cargo_space,
                        unit["cargo"]["metal"] / cargo_space,
                    ]
                )
                unit_type = (
                    0 if unit["unit_type"] == "LIGHT" else 1
                )  # note that build actions use 0 to encode Light
                # normalize the unit position
                pos = np.array(unit["pos"]) / env_cfg.map_size
                unit_vec = np.concatenate(
                    [pos, [unit_type], cargo_vec, [unit["team_id"]]], axis=-1
                )

                # we add some engineered features down here
                # compute closest ice tile
                ice_tile_distances = np.mean(
                    (ice_tile_locations - np.array(unit["pos"])) ** 2, 1
                )
                # normalize the ice tile location
                closest_ice_tile = (
                    ice_tile_locations[np.argmin(ice_tile_distances)] / env_cfg.map_size
                )
                obs_vec = np.concatenate(
                    [unit_vec, factory_vec - pos, closest_ice_tile - pos], axis=-1
                )
                break
            observation[agent] = obs_vec

        return observation
