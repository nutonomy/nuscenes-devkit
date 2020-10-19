from nuscenes import NuScenes


class LidarsegChallengeMappings:
    def __init__(self, nusc: NuScenes):
        self.rawname2mergedname_mapping = self.get_raw2merged()
        self.mergedname2mergedidx_mapping = self.get_merged2idx()

        self.check_mapping()

        self.rawidx2mergedidx_mapping = dict()
        for rawname, rawidx in nusc.lidarseg_name2idx_mapping.items():
            self.rawidx2mergedidx_mapping[rawidx] = self.mergedname2mergedidx_mapping[
                self.rawname2mergedname_mapping[rawname]]

        print(self.rawidx2mergedidx_mapping)

    @staticmethod
    def get_raw2merged():
        return {'noise': 'void_ignore',
                'human.pedestrian.adult': 'pedestrian',
                'human.pedestrian.child': 'pedestrian',
                'human.pedestrian.wheelchair': 'void_ignore',
                'human.pedestrian.stroller': 'void_ignore',
                'human.pedestrian.personal_mobility': 'void_ignore',
                'human.pedestrian.police_officer': 'pedestrian',
                'human.pedestrian.construction_worker': 'pedestrian',
                'animal': 'void_ignore',
                'vehicle.car': 'car',
                'vehicle.motorcycle': 'motorcycle',
                'vehicle.bicycle': 'bicycle',
                'vehicle.bus.bendy': 'bus',
                'vehicle.bus.rigid': 'bus',
                'vehicle.truck': 'truck',
                'vehicle.construction': 'construction_vehicle',
                'vehicle.emergency.ambulance': 'void_ignore',
                'vehicle.emergency.police': 'void_ignore',
                'vehicle.trailer': 'trailer',
                'movable_object.barrier': 'barrier',
                'movable_object.trafficcone': 'traffic_cone',
                'movable_object.pushable_pullable': 'void_ignore',
                'movable_object.debris': 'void_ignore',
                'static_object.bicycle_rack': 'void_ignore',
                'flat.driveable_surface': 'driveable_surface',
                'flat.sidewalk': 'sidewalk',
                'flat.terrain': 'terrain',
                'flat.other': 'other_flat',
                'static.manmade': 'manmade',
                'static.vegetation': 'vegetation',
                'static.other': 'void_ignore',
                'vehicle.ego': 'void_ignore'}

    @staticmethod
    def get_merged2idx():
        """
        """
        return {'void_ignore': 0,
                'barrier': 1,
                'bicycle': 2,
                'bus': 3,
                'car': 4,
                'construction_vehicle': 5,
                'motorcycle': 6,
                'pedestrian': 7,
                'traffic_cone': 8,
                'trailer': 9,
                'truck': 10,
                'driveable_surface': 11,
                'other_flat': 12,
                'sidewalk': 13,
                'terrain': 14,
                'manmade': 15,
                'vegetation': 16}

    def check_mapping(self):
        merged_set = set()
        for raw_name, merged_name in self.rawname2mergedname_mapping.items():
            merged_set.add(merged_name)

        assert len(merged_set) == len(self.mergedname2mergedidx_mapping), 'Error: Number of merged classes is not ' \
                                                                          'the same as the number of merged indices.'


if __name__ == '__main__':
    nusc_ = NuScenes(version='v1.0-mini', dataroot='/data/sets/nuscenes', verbose=True)
    a = LidarsegChallengeMappings(nusc_)