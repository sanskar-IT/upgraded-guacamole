from models import TrackSection, Station, Junction, SectionType, Train
from typing import List, Dict


class TrackConfiguration:
    """Configuration for the 40km railway network with 4 stations and 2 bypass sidings"""
    
    def __init__(self):
        self.total_length_km = 40.0
        self.track_sections = self._create_track_sections()
        self.stations = self._create_stations()
        self.junctions = self._create_junctions()
        self.initial_trains = self._create_initial_trains()
    
    def _create_track_sections(self) -> List[TrackSection]:
        """Create all track sections for the 40km network with bypass sidings"""
        sections = []
        
        # SEC-01: Mainline track from 0.0 km to 5.0 km
        sections.append(TrackSection(
            section_id="SEC-01",
            length_km=5.0,
            section_type=SectionType.MAINLINE,
            start_position_km=0.0,
            end_position_km=5.0,
            max_speed_kmh=120.0
        ))
        
        # SEC-02-MAIN: First mainline segment with stations (5.0 km to 18.0 km)
        sections.append(TrackSection(
            section_id="SEC-02-MAIN",
            length_km=13.0,
            section_type=SectionType.MAINLINE,
            start_position_km=5.0,
            end_position_km=18.0,
            max_speed_kmh=100.0
        ))
        
        # SEC-02-BYPASS: First siding track parallel to SEC-02-MAIN
        sections.append(TrackSection(
            section_id="SEC-02-BYPASS",
            length_km=13.0,
            section_type=SectionType.SIDING,
            start_position_km=5.0,
            end_position_km=18.0,
            max_speed_kmh=80.0
        ))
        
        # SEC-03: Connecting mainline track from 18.0 km to 22.0 km
        sections.append(TrackSection(
            section_id="SEC-03",
            length_km=4.0,
            section_type=SectionType.MAINLINE,
            start_position_km=18.0,
            end_position_km=22.0,
            max_speed_kmh=120.0
        ))
        
        # SEC-04-MAIN: Second mainline segment with station (22.0 km to 28.0 km)
        sections.append(TrackSection(
            section_id="SEC-04-MAIN",
            length_km=6.0,
            section_type=SectionType.MAINLINE,
            start_position_km=22.0,
            end_position_km=28.0,
            max_speed_kmh=100.0
        ))
        
        # SEC-04-BYPASS: Second siding track parallel to SEC-04-MAIN
        sections.append(TrackSection(
            section_id="SEC-04-BYPASS",
            length_km=6.0,
            section_type=SectionType.SIDING,
            start_position_km=22.0,
            end_position_km=28.0,
            max_speed_kmh=80.0
        ))
        
        # SEC-05: Final mainline track from 28.0 km to 40.0 km
        sections.append(TrackSection(
            section_id="SEC-05",
            length_km=12.0,
            section_type=SectionType.MAINLINE,
            start_position_km=28.0,
            end_position_km=40.0,
            max_speed_kmh=120.0
        ))
        
        return sections
    
    def _create_stations(self) -> List[Station]:
        """Create the 4 stations positioned on the mainline sections"""
        return [
            Station(
                station_id="STATION_01",
                station_name="First Station",
                position_km=8.0,  # On SEC-02-MAIN
                platform_sections=["SEC-02-MAIN"],
                is_junction=False
            ),
            Station(
                station_id="STATION_02",
                station_name="Second Station",
                position_km=15.0,  # On SEC-02-MAIN
                platform_sections=["SEC-02-MAIN"],
                is_junction=False
            ),
            Station(
                station_id="STATION_03",
                station_name="Third Station",
                position_km=25.0,  # On SEC-04-MAIN
                platform_sections=["SEC-04-MAIN"],
                is_junction=False
            ),
            Station(
                station_id="STATION_04",
                station_name="Terminal Station",
                position_km=38.0,  # On SEC-05
                platform_sections=["SEC-05"],
                is_junction=False
            )
        ]
    
    def _create_junctions(self) -> List[Junction]:
        """Create the 4 junctions for bypass siding connections"""
        return [
            Junction(
                junction_id="JUNCTION_01",
                position_km=5.0,
                connected_sections=["SEC-01", "SEC-02-MAIN", "SEC-02-BYPASS"],
                current_route="SEC-02-MAIN"
            ),
            Junction(
                junction_id="JUNCTION_02",
                position_km=18.0,
                connected_sections=["SEC-02-MAIN", "SEC-02-BYPASS", "SEC-03"],
                current_route="SEC-03"
            ),
            Junction(
                junction_id="JUNCTION_03",
                position_km=22.0,
                connected_sections=["SEC-03", "SEC-04-MAIN", "SEC-04-BYPASS"],
                current_route="SEC-04-MAIN"
            ),
            Junction(
                junction_id="JUNCTION_04",
                position_km=28.0,
                connected_sections=["SEC-04-MAIN", "SEC-04-BYPASS", "SEC-05"],
                current_route="SEC-05"
            )
        ]
    
    def _create_initial_trains(self) -> List[Train]:
        """Create the 4 initial trains positioned on the new track layout"""
        return [
            Train(
                train_id="TRAIN_A",
                train_name="Express Alpha",
                priority=1,  # Highest priority
                current_speed_kmh=0.0,
                position_km=2.0,  # On SEC-01
                current_track_section_id="SEC-01",
                destination_station="STATION_04",
                max_speed_kmh=120.0,
                length_meters=250.0
            ),
            Train(
                train_id="TRAIN_B",
                train_name="Inter-City Beta",
                priority=2,
                current_speed_kmh=0.0,
                position_km=10.0,  # On SEC-02-MAIN
                current_track_section_id="SEC-02-MAIN",
                destination_station="STATION_03",
                max_speed_kmh=110.0,
                length_meters=200.0
            ),
            Train(
                train_id="TRAIN_C",
                train_name="Regional Charlie",
                priority=3,
                current_speed_kmh=0.0,
                position_km=20.0,  # On SEC-03
                current_track_section_id="SEC-03",
                destination_station="STATION_04",
                max_speed_kmh=100.0,
                length_meters=180.0
            ),
            Train(
                train_id="TRAIN_D",
                train_name="Local Delta",
                priority=4,  # Lowest priority
                current_speed_kmh=0.0,
                position_km=35.0,  # On SEC-05
                current_track_section_id="SEC-05",
                destination_station="STATION_01",
                max_speed_kmh=80.0,
                length_meters=150.0
            )
        ]
    
    def get_section_by_position(self, position_km: float) -> TrackSection:
        """Get the track section at a given position"""
        for section in self.track_sections:
            if section.start_position_km <= position_km <= section.end_position_km:
                return section
        raise ValueError(f"No section found at position {position_km} km")
    
    def get_next_sections(self, current_section_id: str) -> List[TrackSection]:
        """Get possible next sections from the current section (handles bypass sidings)"""
        current_section = next((s for s in self.track_sections if s.section_id == current_section_id), None)
        if not current_section:
            return []
        
        # Find sections that start where this one ends
        next_sections = []
        for section in self.track_sections:
            if abs(section.start_position_km - current_section.end_position_km) < 0.1:
                next_sections.append(section)
        
        return next_sections
    
    def get_section_by_id(self, section_id: str) -> TrackSection:
        """Get a track section by its ID"""
        section = next((s for s in self.track_sections if s.section_id == section_id), None)
        if not section:
            raise ValueError(f"Section {section_id} not found")
        return section
