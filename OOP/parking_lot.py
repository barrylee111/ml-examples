from typing import List

class Spot:
    def __init__(self, initial_value: int) -> None:
        self.num_available = initial_value
    
    def get_num_available(self) -> int: return self.num_available

    def dec_num_available(self) -> None: self.num_available -= 1

class SmallSpot(Spot):
    def __init__(self, initial_value: int) -> None:
        super().__init__(initial_value)

class MediumSpot(Spot):
    def __init__(self, initial_value: int) -> None:
        super().__init__(initial_value)

class LargeSpot(Spot):
    def __init__(self, initial_value: int) -> None:
        super().__init__(initial_value)
    

class ParkingLot:
    def __init__(self, spots: List[int]) -> None:
        self.initial = sum(spots)
        self.spots = [SmallSpot(spots[0]), MediumSpot(spots[1]), LargeSpot(spots[-1])]
        self.used_spots_by_type = {
            'motorcycle': 0,
            'car': 0,
            'van': 0
        }

    def add_vehicle(self, vehicle_type: str) -> None:
        valid_spot_types = self.get_valid_spot_types(vehicle_type)
        
        self.update_spot_counts(vehicle_type, valid_spot_types)

    def get_num_avail_by_spot_type(self):
        keys = ['SmallSpot', 'MediumSpot', 'LargeSpot']
        output = {}
        for idx, spot_type in enumerate(self.spots):
            output[keys[idx]] = spot_type.get_num_available()

        return output

    def get_num_avail_by_vehicle_type(self, vehicle_type: str):
        valid_spot_types = self.get_valid_spot_types(vehicle_type)

        output = 0
        for spot_type in valid_spot_types:
            output += spot_type.get_num_available()

        return output
    
    def get_num_used_spots(self): return sum(self.used_spots_by_type.values())

    def get_total_avail_spots(self): return self.initial - sum(self.used_spots_by_type.values())

    def get_valid_spot_types(self, vehicle_type: str) -> List[SmallSpot | MediumSpot | LargeSpot]:
        valid_spot_types = self.spots
        
        if vehicle_type == 'car': valid_spot_types = self.spots[1:]
        if vehicle_type == 'van': valid_spot_types = [self.spots[-1]]

        return valid_spot_types
    
    def update_spot_counts(self, vehicle_type: str, valid_spot_types) -> None:
        updated = False
        for spot_type in valid_spot_types:
            if spot_type.get_num_available() > 0:
                self.used_spots_by_type[vehicle_type] += 1
                spot_type.dec_num_available()
                updated = True
                break
        
        response_msg = f'{vehicle_type} added...' if updated else f'No available spots for your {vehicle_type}...'
        print(response_msg)
        
    
pl = ParkingLot([10, 10, 1])
count = pl.get_total_avail_spots()
print(f'Get Starting count: {count}')
print()
pl.add_vehicle('car')
count = pl.get_total_avail_spots()
print(f'Total Available Spots: {count}')
print()
pl.add_vehicle('motorcycle')
count = pl.get_total_avail_spots()
print(f'Total Available Spots: {count}')
print()
pl.add_vehicle('van')
count = pl.get_total_avail_spots()
print(f'Total Available Spots: {count}')
print()
pl.add_vehicle('van')
count = pl.get_total_avail_spots()
print(f'Total Available Spots: {count}')
print()
pl.add_vehicle('van')
count = pl.get_total_avail_spots()
print(f'Total Available Spots: {count}')
print()
print(f'Spots used by type: {pl.used_spots_by_type}')
print(f'Num used spots: {pl.get_num_used_spots()}')
print(f'Num available by vehicle type - car: {pl.get_num_avail_by_vehicle_type("car")}, van: {pl.get_num_avail_by_vehicle_type("van")}, motorcycle: {pl.get_num_avail_by_vehicle_type("motorcycle")}')
print(f'Num available by spot_type: {pl.get_num_avail_by_spot_type()}')