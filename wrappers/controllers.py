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
            "Robot": spaces.MultiDiscrete(np.zeros((1, P, x, y), dtype=int) + self.TotalRobotActions),
            #"HeavyRobot": spaces.MultiDiscrete(np.zeros((1, P, x, y), dtype=int) + self.TotalRobotActions),
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
                """
                if unit["unit_type"] == "LIGHT":
                    unitType = "LightRobot"
                else:
                    unitType = "HeavyRobot"
                """
                cargo_space = env_cfg.ROBOTS[unit["unit_type"]].CARGO_SPACE
                battery_cap = env_cfg.ROBOTS[unit["unit_type"]].BATTERY_CAPACITY
                x,y = unit["pos"]
                choice = actionTensor["Robot"][0][P][x][y]
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
                    if resource == 4:
                        amount = unit["power"]
                    else:
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


    def PosToUnitDict(self,obs):
        shared_obs = obs["player_0"]
        PlayerDict = {"player_0": {},"player_1": {}}
        Agents = ["player_0", "player_1"]
        for agent in Agents:
            units = shared_obs["units"][agent]
            for unit_id in units.keys():
                unit = units[unit_id]
                x,y = unit["pos"]
                PlayerDict[agent][(x,y)]= unit_id

        return PlayerDict

    def PosToFactoryDict(self, obs):
        shared_obs = obs["player_0"]
        PlayerDict = {"player_0": {}, "player_1": {}}
        Agents = ["player_0", "player_1"]
        for agent in Agents:
            factories = shared_obs["factories"][agent]
            for unit_id in factories.keys():
                factory = factories[unit_id]
                x, y = factory["pos"]
                PlayerDict[agent][(x, y)]=unit_id
                PlayerDict[agent][(x+1, y)]=unit_id
                PlayerDict[agent][(x, y+1)]=unit_id
                PlayerDict[agent][(x+1, y+1)]=unit_id
                PlayerDict[agent][(x+1, y-1)]=unit_id
                PlayerDict[agent][(x-1, y)]=unit_id
                PlayerDict[agent][(x-1, y-1)]=unit_id
                PlayerDict[agent][(x-1, y+1)]=unit_id
                PlayerDict[agent][(x, y-1)]=unit_id

        return PlayerDict



    def nextPosLoc(self,x,y,direction):
        if direction == 0:
            return (x,y-1)
        if direction == 1:
            return (x+1,y)
        if direction == 2:
            return (x,y+1)
        if direction == 3:
            return (x-1,y)



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
        env_cfg = EnvConfig

        for agent in Agents:
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
            units = shared_obs["units"][agent]
            for unit_id in units.keys():
                unit = units[unit_id]
                x,y = unit["pos"]

                # lets first check the move locations
                # not valid if outside grid
                # not valid if next loc is an enemy factory
                # will handle collisions later
                FactoryDict = self.PosToFactoryDict(obs)
                if agent == "player_0":
                    opponentFactoryLocations = FactoryDict["player_1"]
                else:
                    opponentFactoryLocations = FactoryDict["player_0"]
                for movedirection in range(self.move_dim_high):
                    nextPosX,nextPosY = self.nextPosLoc(x,y,movedirection)
                    if nextPosX <0 or nextPosY <0 or nextPosX >= self.Mapsize or nextPosY >= self.Mapsize:
                        available_actions_mask["Robot"][0][P][x][y][movedirection] = False
                    if (nextPosX,nextPosY) in opponentFactoryLocations:
                        available_actions_mask["Robot"][0][P][x][y][movedirection] = False

                    # lets check if we have enough power for rubble
                    rubbleval = shared_obs["board"]["rubble"][nextPosX][nextPosY]
                    if unit["unit_type"] == "LIGHT":
                        powerRequired = math.floor(1 + (0.05 * rubbleval))
                    else:
                        powerRequired = math.floor(20 + (0.05 * rubbleval))
                    if unit["power"] < powerRequired:
                        available_actions_mask["Robot"][0][P][x][y][movedirection] = False

                #TRANSFERS


                # make all transfers valid in terms of directions - we cant be sure which robot will move to our adjacent tile,
                # which might enable a transfer
                # just check if resource > 0

                UnitLocations = self.PosToUnitDict(obs)
                AgentUnits = UnitLocations[agent]

                for choice in range(self.move_dim_high,self.transfer_dim_high):
                    id = choice
                    id = id - self.move_dim_high
                    direction = id % 5
                    resource = int(id / 5)
                    # transfer all resources in cargo
                    if resource == 4:
                        amount = unit["power"]
                    else:
                        amount = unit["cargo"][resource]
                    if amount < 1:
                        available_actions_mask["Robot"][0][P][x][y][choice] = False

                    if direction !=0:
                        # direction number 0 transfer to tile you are currently on(CENTER), not adjacent
                        # direction -1 because my directions function returns based on 0 index
                        nextPosX, nextPosY = self.nextPosLoc(x, y, direction-1)
                        if nextPosX < 0 or nextPosY < 0 or nextPosX >= self.Mapsize or nextPosY >= self.Mapsize:
                            available_actions_mask["Robot"][0][P][x][y][choice] = False
                        if (nextPosX, nextPosY) in opponentFactoryLocations:
                            available_actions_mask["Robot"][0][P][x][y][choice] = False

                        # lets check if receiving unit has enough cargo
                        if (nextPosX,nextPosY) in AgentUnits:
                            adjUnitId = AgentUnits[(nextPosX,nextPosY)]
                            adjUnit = units[adjUnitId]
                            cargo_space = env_cfg.ROBOTS[adjUnit["unit_type"]].CARGO_SPACE
                            battery_cap = env_cfg.ROBOTS[adjUnit["unit_type"]].BATTERY_CAPACITY
                            if resource == 4:
                                spaceleft = battery_cap - adjUnit["power"]
                            else:
                                spaceleft = cargo_space - adjUnit["cargo"][resource]
                            if amount > spaceleft:
                                available_actions_mask["Robot"][0][P][x][y][choice] = False




                # pickup action
                # we pick up only power if on top of factory or battery is not charged
                x, y = unit["pos"]
                battery_cap = env_cfg.ROBOTS[unit["unit_type"]].BATTERY_CAPACITY
                OwnFactoryLocations = FactoryDict[agent]
                if (x, y) not in OwnFactoryLocations or unit["power"] == battery_cap:
                    available_actions_mask["Robot"][0][P][x][y][self.transfer_dim_high] = False

                # DIG ACTION
                # cant dig if on own factory location or own lichen tile
                x, y = unit["pos"]
                OwnFactoryLocations = FactoryDict[agent]
                if (x,y) in OwnFactoryLocations:
                    available_actions_mask["Robot"][0][P][x][y][self.pickup_dim_high] = False

                if shared_obs["board"]["lichen_strains"][x][y] in shared_obs["teams"][agent]["factory_strains"]:
                    available_actions_mask["Robot"][0][P][x][y][self.pickup_dim_high] = False


                # no op action is always valid!
        return available_actions_mask

































    # added comment for testing




# have to test env
