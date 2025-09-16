from models import TrackSection, Station, Junction, SectionType, Train
from typing import List, Dict


class TrackConfiguration:
    """Configuration for the 40km railway network with 4 stations and 2 junctions"""
    
    def __init__(self):
        self.total_length_km = 40.0
        self.track_sections = self._create_track_sections()
        self.stations = self._create_stations()
        self.junctions = self._create_junctions()
        self.initial_trains = self._create_initial_trains()
    
    def _create_track_sections(self) -> List[TrackSection]:
        """Create all track sections for the 40km network"""
        sections = []
        
        # Section 1: Start to Station 1 (0-8 km)
        sections.append(TrackSection(
            section_id="SEC-01",
            length_km=8.0,
            section_type=SectionType.MAINLINE,
            start_position_km=0.0,
            end_position_km=8.0,
            max_speed_kmh=120.0
        ))
        
        # Station 1 platform sections (8-10 km)
        sections.append(TrackSection(
            section_id="STN-01-PLATFORM",
            length_km=2.0,
            section_type=SectionType.STATION,
            start_position_km=8.0,
            end_position_km=10.0,
            max_speed_kmh=30.0
        ))
        
        # Section 2: Station 1 to Junction 1 (10-15 km)
        sections.append(TrackSection(
            section_id="SEC-02",
            length_km=5.0,
            section_type=SectionType.MAINLINE,
            start_position_km=10.0,
            end_position_km=15.0,
            max_speed_kmh=100.0
        ))
        
        # Junction 1 sections (15-16 km)
        sections.append(TrackSection(
            section_id="JCT-01-MAIN",
            length_km=0.5,
            section_type=SectionType.MAINLINE,
            start_position_km=15.0,
            end_position_km=15.5,
            max_speed_kmh=40.0
        ))
        
        sections.append(TrackSection(
            section_id="JCT-01-SIDING",
            length_km=0.5,
            section_type=SectionType.SIDING,
            start_position_km=15.0,
            end_position_km=15.5,
            max_speed_kmh=40.0
        ))
        
        # Section 3: Junction 1 to Station 2 (15.5-20 km)
        sections.append(TrackSection(
            section_id="SEC-03",
            length_km=4.5,
            section_type=SectionType.MAINLINE,
            start_position_km=15.5,
            end_position_km=20.0,
            max_speed_kmh=110.0
        ))
        
        # Station 2 platform sections (20-22 km)
        sections.append(TrackSection(
            section_id="STN-02-PLATFORM",
            length_km=2.0,
            section_type=SectionType.STATION,
            start_position_km=20.0,
            end_position_km=22.0,
            max_speed_kmh=30.0
        ))
        
        # Section 4: Station 2 to Station 3 (22-30 km)
        sections.append(TrackSection(
            section_id="SEC-04",
            length_km=8.0,
            section_type=SectionType.MAINLINE,
            start_position_km=22.0,
            end_position_km=30.0,
            max_speed_kmh=120.0
        ))
        
        # Station 3 platform sections (30-32 km)
        sections.append(TrackSection(
            section_id="STN-03-PLATFORM",
            length_km=2.0,
            section_type=SectionType.STATION,
            start_position_km=30.0,
            end_position_km=32.0,
            max_speed_kmh=30.0
        ))
        
        # Section 5: Station 3 to Junction 2 (32-35 km)
        sections.append(TrackSection(
            section_id="SEC-05",
            length_km=3.0,
            section_type=SectionType.MAINLINE,
            start_position_km=32.0,
            end_position_km=35.0,
            max_speed_kmh=100.0
        ))
        
        # Junction 2 sections (35-36 km)
        sections.append(TrackSection(
            section_id="JCT-02-MAIN",
            length_km=0.5,
            section_type=SectionType.MAINLINE,
            start_position_km=35.0,
            end_position_km=35.5,
            max_speed_kmh=40.0
        ))
        
        sections.append(TrackSection(
            section_id="JCT-02-SIDING",
            length_km=0.5,
            section_type=SectionType.SIDING,
            start_position_km=35.0,
            end_position_km=35.5,
            max_speed_kmh=40.0
        ))
        
        # Section 6: Junction 2 to Station 4 (35.5-38 km)
        sections.append(TrackSection(
            section_id="SEC-06",
            length_km=2.5,
            section_type=SectionType.MAINLINE,
            start_position_km=35.5,
            end_position_km=38.0,
            max_speed_kmh=100.0
        ))
        
        # Station 4 platform sections (38-40 km)
        sections.append(TrackSection(
            section_id="STN-04-PLATFORM",
            length_km=2.0,
            section_type=SectionType.STATION,
            start_position_km=38.0,
            end_position_km=40.0,
            max_speed_kmh=30.0
        ))
        
        return sections
    
    def _create_stations(self) -> List[Station]:
        """Create the 4 stations"""
        return [
            Station(
                station_id="STATION_01",
                station_name="Central Junction",
                position_km=9.0,
                platform_sections=["STN-01-PLATFORM"],
                is_junction=False
            ),
            Station(
                station_id="STATION_02",
                station_name="Midway Station",
                position_km=21.0,
                platform_sections=["STN-02-PLATFORM"],
                is_junction=False
            ),
            Station(
                station_id="STATION_03",
                station_name="Industrial Hub",
                position_km=31.0,
                platform_sections=["STN-03-PLATFORM"],
                is_junction=False
            ),
            Station(
                station_id="STATION_04",
                station_name="Terminal Station",
                position_km=39.0,
                platform_sections=["STN-04-PLATFORM"],
                is_junction=False
            )
        ]
    
    def _create_junctions(self) -> List[Junction]:
        """Create the 2 junctions"""
        return [
            Junction(
                junction_id="JUNCTION_01",
                position_km=15.25,
                connected_sections=["SEC-02", "JCT-01-MAIN", "JCT-01-SIDING", "SEC-03"],
                current_route="JCT-01-MAIN"
            ),
            Junction(
                junction_id="JUNCTION_02",
                position_km=35.25,
                connected_sections=["SEC-05", "JCT-02-MAIN", "JCT-02-SIDING", "SEC-06"],
                current_route="JCT-02-MAIN"
            )
        ]
    
    def _create_initial_trains(self) -> List[Train]:
        """Create the 4 initial trains with different priorities"""
        return [
            Train(
                train_id="TRAIN_A",
                train_name="Express Alpha",
                priority=1,  # Highest priority
                current_speed_kmh=0.0,
                position_km=0.5,
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
                position_km=5.0,
                current_track_section_id="SEC-01",
                destination_station="STATION_03",
                max_speed_kmh=110.0,
                length_meters=200.0
            ),
            Train(
                train_id="TRAIN_C",
                train_name="Regional Charlie",
                priority=3,
                current_speed_kmh=0.0,
                position_km=12.0,
                current_track_section_id="SEC-02",
                destination_station="STATION_04",
                max_speed_kmh=100.0,
                length_meters=180.0
            ),
            Train(
                train_id="TRAIN_D",
                train_name="Local Delta",
                priority=4,  # Lowest priority
                current_speed_kmh=0.0,
                position_km=25.0,
                current_track_section_id="SEC-04",
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
        """Get possible next sections from the current section"""
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
