from macad_gym.carla.scenarios import Scenarios


class CustomScenarios(Scenarios):

    SSUI4C_TOWN3_C = {"map": "Town03", "actors": {
        "car1": {"start": [170.5, 80, 0.4], "end": [144, 59, 0]},
        "car2": {"start": [195, 59, 0.4], "end": [167, 75.7, 0.13]},
        "car3": {"start": [140.6, 62.6, 0.4], "end": [191.2, 62.7, 0]},
        "car4": {"start": [188, 59, 0.4], "end": [139.2, 59, 0]}},
        "num_vehicles": 0, "num_pedestrians": 10, "weather_distribution": [0], "max_steps": 2000}
    SUI4C_TOWN3_C = {"map": "Town03", "actors": {
        "car1": {"start": [70, -132.8, 8], "end": [130, -132, 8]},
        "car2": {"start": [84.3, -118, 9], "end": [125, -132, 8]},
        "car3": {"start": [43, -133, 4], "end": [120, -132, 8]},
        "car4": {"start": [84.3, -112, 9], "end": [115, -132, 8]}},
        "num_vehicles": 0, "num_pedestrians": 0, "weather_distribution": [0], "max_steps": 2000}
    CTI4C_TOWN3_C = {"map": "Town03", "actors": {
        "car1": {"start": [-6.4, 92, 0.4], "end": [26.2, 134.4, 0]},
        "car2": {"start": [-6.4, 85, 0.4], "end": [20.2, 134.4, 0]},
        "car3": {"start": [6, 162, 0.4], "end": [5.5, 94, 0]},
        "car4": {"start": [6, 168, 0.4], "end": [5.5, 100, 0]}},
        "num_vehicles": 0, "num_pedestrians": 10, "weather_distribution": [0], "max_steps": 2000}

    BIG_TOWN3_C = {"map": "Town03", "actors": {
        "car1": {"start": [170.5, 80, 0.4], "end": [144, 59, 0]},
        "car2": {"start": [195, 59, 0.4], "end": [167, 75.7, 0.13]},
        "car3": {"start": [140.6, 62.6, 0.4], "end": [191.2, 62.7, 0]},
        "car4": {"start": [188, 59, 0.4], "end": [139.2, 59, 0]},
        "car5": {"start": [70, -132.8, 8], "end": [130, -132, 8]},
        "car6": {"start": [84.3, -118, 9], "end": [125, -132, 8]},
        "car7": {"start": [43, -133, 4], "end": [120, -132, 8]},
        "car8": {"start": [84.3, -112, 9], "end": [115, -132, 8]},
        "car9": {"start": [-6.4, 92, 0.4], "end": [26.2, 134.4, 0]},
        "car10": {"start": [-6.4, 85, 0.4], "end": [20.2, 134.4, 0]},
        "car11": {"start": [6, 162, 0.4], "end": [5.5, 94, 0]},
        "car12": {"start": [6, 168, 0.4], "end": [5.5, 100, 0]}},
        "num_vehicles": 0, "num_pedestrians": 30, "weather_distribution": [0], "max_steps": 2000}
    local_map = locals()
