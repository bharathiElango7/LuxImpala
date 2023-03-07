import sys
from typing import Any, Dict

import numpy as np
import numpy.typing as npt
from gym import spaces
from lux.config import EnvConfig
import math

# Controller class copied here since you won't have access to the luxai_s2 package directly on the competition server
class Controller:
    def __init__(self, action_space: spaces.Space) -> None:
        self.action_space = action_space

    def action_to_lux_action(
        self, agent: str, obs: Dict[str, Any], action: npt.NDArray
    ):
        """
        Takes as input the current "raw observation" and the parameterized action and returns
        an action formatted for the Lux env
        """
        raise NotImplementedError()

    def action_masks(self, agent: str, obs: Dict[str, Any]):
        """
        Generates a boolean action mask indicating in each discrete dimension whether it would be valid or not
        """
        raise NotImplementedError()



DIRECTIONS = ["CENTER","UP","RIGHT","DOWN","LEFT"]
RESOURCES = ["ICE","ORE","WATER","METAL","POWER"]

class SimpleUnitDiscreteController(Controller):
    def __init__(self, env_cfg) -> None:
        """
        A simple controller that controls only the robot that will get spawned.
        Moreover, it will always try to spawn one heavy robot if there are none regardless of action given

        For the robot unit
        - 4 cardinal direction movement (4 dims)
        - a move center no-op action (1 dim)
        - transfer action just for transferring ice in 4 cardinal directions or center (5)
        - pickup action for power (1 dims)
        - dig action (1 dim)
        - no op action (1 dim) - equivalent to not submitting an action queue which costs power

        It does not include
        - self destruct action
        - recharge action
        - planning (via actions executing multiple times or repeating actions)
        - factory actions
        - transferring power or resources other than ice

        To help understand how to this controller works to map one action space to the original lux action space,
        see how the lux action space is defined in luxai_s2/spaces/action.py

        """
        self.Mapsize = 48
        self.env_cfg = env_cfg
        self.move_act_dims = 4
        # 5 transfer directions * 5 resources
        self.transfer_act_dims = 25
        self.pickup_act_dims = 1
        self.dig_act_dims = 1
        self.no_op_dims = 1

        self.move_dim_high = self.move_act_dims
        self.transfer_dim_high = self.move_dim_high + self.transfer_act_dims
        self.pickup_dim_high = self.transfer_dim_high + self.pickup_act_dims
        self.dig_dim_high = self.pickup_dim_high + self.dig_act_dims
        self.no_op_dim_high = self.dig_dim_high + self.no_op_dims

        self.total_act_dims = self.no_op_dim_high
        action_space = self.get_action_space(self.Mapsize)
        self.RobotActions = []
        for movingaction in DIRECTIONS:
            self.RobotActions.append("Move" + movingaction)
        for transferaction in DIRECTIONS:
            for resource in RESOURCES:
                self.RobotActions.append("Transfer" + resource + transferaction)
        # for pickupaction in RESOURCES:
        # RobotActions.append("Pickup"+pickupaction)
        # pickup only power from factory
        self.RobotActions.append("PickupPower")
        self.RobotActions.append("Dig")
        # Move center is NO-OP for a robot

        self.TotalRobotActions = len(self.RobotActions)

        self.FactoryActions = ["BuildLightRobot", "BuildHeavyRobot", "WaterForLichen", "NO-OP"]
        self.TotalFactoryActions = len(self.FactoryActions)

        super().__init__(action_space)
    def get_action_space(self,MapSize:int):
        x = MapSize
        y = MapSize
        P=2
        # Transfer all resources in cargo -  to make it discrete - otherwise too many actions
        # Pickup upto cargo capacity - to make it discrete - otherwise too many actions

        action_space = spaces.Dict({
            "LightRobot": spaces.MultiDiscrete(np.zeros((1, P, x, y), dtype=int) + self.TotalRobotActions),
            "HeavyRobot": spaces.MultiDiscrete(np.zeros((1, P, x, y), dtype=int) + self.TotalRobotActions),
            "Factory": spaces.MultiDiscrete(np.zeros((1, P, x, y), dtype=int) + self.TotalFactoryActions)

        })

        return action_space

    def is_move_action(self, id):
        return id < self.move_dim_high
    def get_move_action(self, id):
        # move direction is id + 1 since we don't allow move center here
        # keep moving forever until change of direction
        return np.array([0, id+1, 0, 0, 0, 9999])

    def is_transfer_action(self, id):
        return id < self.transfer_dim_high
    def get_transfer_action(self, direction,resource, amount):
        # its a one step action, no use of repeating
        return np.array([1, direction, resource, amount, 0, 1])

    def is_pickup_action(self, id):
        return id < self.pickup_dim_high

    def is_dig_action(self, id):
        return id < self.dig_dim_high

    def get_pickup_action(self, amount):
        # max battery capacity is 3000 for heavy robot, so pick up max amount of power
        # 4 means power
        # its a one step action, no use of repeating
        return np.array([2, 0, 4, 3000, 0, 1])

    def get_dig_action(self, id):
        # keep diggin until cargo is full
        return np.array([3, 0, 0, 0, 0, 9999])

    def process_action(self,obs,actionTensor:Dict[str, np.ndArray]):
        shared_obs = obs["player_0"]
        Agents = ["player_0", "player_1"]
        env_cfg = EnvConfig
        FinalAction = {}
        for agent in Agents:
            lux_action = dict()
            if agent == "player_0":
                P = 0
            else:
                P = 1

            units = shared_obs["units"][agent]
            for unit_id in units.keys():
                unit = units[unit_id]
                if unit["unit_type"] == "LIGHT":
                    unitType = "LightRobot"
                else:
                    unitType = "HeavyRobot"
                cargo_space = env_cfg.ROBOTS[unit["unit_type"]].CARGO_SPACE
                battery_cap = env_cfg.ROBOTS[unit["unit_type"]].BATTERY_CAPACITY
                x,y = unit["pos"]
                choice = actionTensor[unitType][0][P][x][y]
                action_queue = []
                no_op = False
                if self.is_move_action(choice):
                    action_queue = [self.get_move_action(choice)]
                elif self.is_transfer_action(choice):
                    # there are 25 transfer actions
                    # assign 5 for each resource
                    id = choice
                    id = id - self.move_dim_high
                    direction = id % 5
                    resource = int(id / 5)
                    # transfer all resources in cargo
                    amount = unit["cargo"][resource]
                    action_queue = [self.get_transfer_action(direction,resource,amount)]
                elif self.is_pickup_action(choice):
                    # pickup only power from factory
                    chargeleft = battery_cap - unit["power"]
                    action_queue = [self.get_pickup_action(chargeleft)]
                elif self.is_dig_action(choice):
                    action_queue = [self.get_dig_action(choice)]
                else:
                    # action is a no_op, so we don't update the action queue
                    no_op = True

                # simple trick to help agents conserve power is to avoid updating the action queue
                # if the agent was previously trying to do that particular action already
                if len(unit["action_queue"]) > 0 and len(action_queue) > 0:
                    same_actions = (unit["action_queue"][0] == action_queue[0]).all()
                    if same_actions:
                        no_op = True
                if not no_op:
                    lux_action[unit_id] = action_queue

            factories = shared_obs["factories"][agent]
            for unit_id in factories.keys():
                factory = factories[unit_id]
                x, y = factory["pos"]
                choice = actionTensor["Factory"][0][P][x][y]
                lux_action[unit_id] = choice

            FinalAction[agent] = lux_action

        return FinalAction

    # start from center of factory and do a dfs
    # subtract 9 from result.
    # this will number of connected lichen tiles + new lichen tiles
    def dfs(self,board,visited,x,y,strainId):
        if x<0 or y<0 or x>=self.Mapsize or y>=self.Mapsize:
            return 0
        if (x,y) in visited or board["rubble"][x][y] !=0 or board["ice"][x][y] !=0 or board["ore"][x][y] !=0:
            return 0
        if board["lichen_strains"][x][y] != strainId:
            # new lichen tile to be be planted
            visited.add((x, y))
            return 1
        visited.add((x,y))
        # if current cell has lichen, continue dfs
        return 1 + self.dfs(board,visited,x+1,y,strainId) + self.dfs(board,visited,x-1,y,strainId) \
            +self.dfs(board,visited,x,y+1,strainId) + self.dfs(board,visited,x,y-1,strainId)

    def actionMask2(self,obs):
        actionspace = self.get_action_space(self.Mapsize)

        # create an actions mask nd array with all actions allowed and set to true.
        # last dimension is number of possible actions for each unit type
        available_actions_mask = {
            key: np.ones(space.shape + (self.TotalRobotActions if key == "Robot" else self.TotalFactoryActions,), dtype=bool)
            for key, space in actionspace.spaces.items()
        }
        shared_obs = obs["player_0"]
        Agents = ["player_0", "player_1"]

        for agent in Agents:
            lux_action = dict()
            if agent == "player_0":
                P = 0
            else:
                P = 1
            # factory actions - build robots only if metal cost + power cost is met
            # 0 - light
            # 1- heavy
            # 2- water
            # 3- no op
            factories = shared_obs["factories"][agent]
            for unit_id in factories.keys():
                factory = factories[unit_id]
                x,y = factory["pos"]
                # check if factory has enough metal and power to build a robot
                if factory["power"]<50 or factory["metal"]<10:
                    available_actions_mask["Factory"][0][P][x][y][0] = False
                    available_actions_mask["Factory"][0][P][x][y][1] = False
                elif factory["power"] < 500 or factory["metal"] < 100:
                    available_actions_mask["Factory"][0][P][x][y][1] = False
                strainId = factory["strain_id"]
                water = factory["water"]
                visited = set()
                waterRequiredForLichen = math.ceil((self.dfs(shared_obs["board"],visited,x,y,strainId)-9)/10)
                if water < waterRequiredForLichen:
                    available_actions_mask["Factory"][0][P][x][y][2] = False

            # robot actions







    def action_masks(self, agent: str, obs: Dict[str, Any]):
        """
        Defines a simplified action mask for this controller's action space

        Doesn't account for whether robot has enough power
        """

        # compute a factory occupancy map that will be useful for checking if a board tile
        # has a factory and which team's factory it is.
        shared_obs = obs[agent]
        factory_occupancy_map = (
            np.ones_like(shared_obs["board"]["rubble"], dtype=int) * -1
        )
        factories = dict()
        for player in shared_obs["factories"]:
            factories[player] = dict()
            for unit_id in shared_obs["factories"][player]:
                f_data = shared_obs["factories"][player][unit_id]
                f_pos = f_data["pos"]
                # store in a 3x3 space around the factory position it's strain id.
                factory_occupancy_map[
                    f_pos[0] - 1 : f_pos[0] + 2, f_pos[1] - 1 : f_pos[1] + 2
                ] = f_data["strain_id"]

        units = shared_obs["units"][agent]
        action_mask = np.zeros((self.total_act_dims), dtype=bool)
        for unit_id in units.keys():
            action_mask = np.zeros(self.total_act_dims)
            # movement is always valid
            action_mask[:4] = True

            # transferring is valid only if the target exists
            unit = units[unit_id]
            pos = np.array(unit["pos"])
            # a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
            move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])
            for i, move_delta in enumerate(move_deltas):
                transfer_pos = np.array(
                    [pos[0] + move_delta[0], pos[1] + move_delta[1]]
                )
                # check if theres a factory tile there
                if (
                    transfer_pos[0] < 0
                    or transfer_pos[1] < 0
                    or transfer_pos[0] >= len(factory_occupancy_map)
                    or transfer_pos[1] >= len(factory_occupancy_map[0])
                ):
                    continue
                factory_there = factory_occupancy_map[transfer_pos[0], transfer_pos[1]]
                if factory_there in shared_obs["teams"][agent]["factory_strains"]:
                    action_mask[
                        self.transfer_dim_high - self.transfer_act_dims + i
                    ] = True

            factory_there = factory_occupancy_map[pos[0], pos[1]]
            on_top_of_factory = (
                factory_there in shared_obs["teams"][agent]["factory_strains"]
            )

            # dig is valid only if on top of tile with rubble or resources or lichen
            board_sum = (
                shared_obs["board"]["ice"][pos[0], pos[1]]
                + shared_obs["board"]["ore"][pos[0], pos[1]]
                + shared_obs["board"]["rubble"][pos[0], pos[1]]
                + shared_obs["board"]["lichen"][pos[0], pos[1]]
            )
            if board_sum > 0 and not on_top_of_factory:
                action_mask[
                    self.dig_dim_high - self.dig_act_dims : self.dig_dim_high
                ] = True

            # pickup is valid only if on top of factory tile
            if on_top_of_factory:
                action_mask[
                    self.pickup_dim_high - self.pickup_act_dims : self.pickup_dim_high
                ] = True
                action_mask[
                    self.dig_dim_high - self.dig_act_dims : self.dig_dim_high
                ] = False

            # no-op is always valid
            action_mask[-1] = True
            break
        return action_mask

# have to test env
