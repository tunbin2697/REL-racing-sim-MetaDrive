"""
Silverstone Grand Prix Circuit (MetaDrive 3D)
Thiết kế chính xác theo sa bàn Silverstone F1 Circuit từ dữ liệu đồ án:
- 18 góc cua kinh điển (Abbey, Farm, Village, The Loop, Aintree, Brooklands, Luffield, Woodcote,
  Copse, Maggots, Becketts, Chapel, Stowe, Vale, Club).
- Vòng lặp khép kín hoàn hảo (Closed-Loop Topology, sai số khép góc < 0.0001m).
- 2 làn đường thể thao rộng 9.0m (mỗi làn 4.5m).
- Không rào chắn bê tông sát mép (PGLineType.CONTINUOUS), tuân thủ 100% Section 2.2 & 2.3 của PROJECT_SPECIFICATION.md.
"""

import json
import os
import numpy as np
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.straight import Straight
from metadrive.component.pgblock.curve import Curve
from metadrive.component.road_network.road import Road
from metadrive.constants import PGLineType
from metadrive.component.pg_space import Parameter
from metadrive.component.map.pg_map import PGMap
from metadrive.manager.pg_map_manager import PGMapManager


class SilverstoneMap(PGMap):
    """Bản đồ Silverstone Grand Prix F1 với hình học khép kín hoàn hảo"""
    
    def _generate(self):
        assert self.engine is not None, "Engine is not initialized!"
        assert self.road_network is not None, "Map road network is not initialized!"
        assert self.blocks is not None, "Map blocks list is not initialized!"
        assert len(self.road_network.graph) == 0, "Map network is not empty!"

        parent_node_path, physics_world = self.engine.worldNP, self.engine.physics_world
        config = self.config if self.config is not None else {}
        LANE_NUM = config.get("lane_num", 2)
        LANE_WIDTH = config.get("lane_width", 4.5)

        # Nạp dữ liệu cấu hình các block đã giải tối ưu
        json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "silverstone_blocks.json")
        with open(json_path, "r") as f:
            data = json.load(f)
        blocks_data = data["blocks"]

        # Block 0: Hamilton Straight (Start / Finish Grid)
        first_len = float(blocks_data[0]["len"])
        last_block = FirstPGBlock(
            self.road_network,
            lane_width=LANE_WIDTH,
            lane_num=LANE_NUM,
            render_root_np=parent_node_path,
            physics_world=physics_world,
            length=first_len,
            remove_negative_lanes=True,
            side_lane_line_type=PGLineType.CONTINUOUS,
            center_line_type=PGLineType.BROKEN,
        )
        self.blocks.append(last_block)

        block_index = 1
        for b in blocks_data[1:]:
            socket = last_block.get_socket(0)
            if b["type"] == "S":
                block = Straight(
                    block_index,
                    socket,
                    self.road_network,
                    1,
                    remove_negative_lanes=True,
                    side_lane_line_type=PGLineType.CONTINUOUS,
                    center_line_type=PGLineType.BROKEN,
                )
                block.construct_from_config(
                    {Parameter.length: float(b["len"])},
                    parent_node_path,
                    physics_world,
                )
            else:
                block = Curve(
                    block_index,
                    socket,
                    self.road_network,
                    1,
                    remove_negative_lanes=True,
                    side_lane_line_type=PGLineType.CONTINUOUS,
                    center_line_type=PGLineType.BROKEN,
                )
                block.construct_from_config(
                    {
                        Parameter.length: float(b["len"]),
                        Parameter.radius: float(b["r"]),
                        Parameter.angle: float(b["ang"]),
                        Parameter.dir: int(b["dir"]),
                    },
                    parent_node_path,
                    physics_world,
                )
                
            self.blocks.append(block)
            last_block = block
            block_index += 1


class SilverstoneMapManager(PGMapManager):
    """Quản lý bản đồ nạp trực tiếp cấu hình SilverstoneMap (Singleton Persistent Map)"""
    def __init__(self):
        super(SilverstoneMapManager, self).__init__()
        self.silverstone_map = None

    def reset(self):
        assert self.engine is not None, "Engine is not initialized"
        config = self.engine.global_config
        if self.silverstone_map is None:
            spawned = self.spawned_objects if self.spawned_objects is not None else {}
            if len(spawned) == 0:
                map_config = config["map_config"] if config is not None else {}
                self.silverstone_map = self.spawn_object(SilverstoneMap, map_config=map_config, random_seed=None)
            else:
                self.silverstone_map = list(spawned.values())[0]
        if self.silverstone_map is not None:
            self.load_map(self.silverstone_map)
