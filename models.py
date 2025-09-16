from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum
import uuid


class SectionType(str, Enum):
    """Enumeration for track section types"""
    MAINLINE = "mainline"
    SIDING = "siding"
    STATION = "station"


class Train(BaseModel):
    """Model representing a train in the traffic control system"""
    train_id: str = Field(..., description="Unique identifier for the train (e.g., 'RJD-12345')")
    train_name: str = Field(..., description="Display name of the train (e.g., 'Rajdhani Express')")
    priority: int = Field(..., ge=1, le=5, description="Train priority: 1 (highest) to 5 (lowest)")
    current_speed_kmh: float = Field(..., ge=0, description="Current speed in kilometers per hour")
    position_km: float = Field(..., ge=0, description="Linear position on the overall track in kilometers")
    current_track_section_id: str = Field(..., description="ID of the current track section")
    destination_station: Optional[str] = Field(None, description="Target destination station")
    max_speed_kmh: float = Field(default=120.0, description="Maximum allowed speed for this train")
    length_meters: float = Field(default=200.0, description="Length of the train in meters")
    
    class Config:
        json_schema_extra = {
            "example": {
                "train_id": "RJD-12345",
                "train_name": "Rajdhani Express",
                "priority": 1,
                "current_speed_kmh": 80.5,
                "position_km": 15.2,
                "current_track_section_id": "SEC-01A",
                "destination_station": "STATION_04",
                "max_speed_kmh": 120.0,
                "length_meters": 250.0
            }
        }


class TrackSection(BaseModel):
    """Model representing a section of track"""
    section_id: str = Field(..., description="Unique identifier for the track section (e.g., 'SEC-01A')")
    length_km: float = Field(..., gt=0, description="Length of the section in kilometers")
    section_type: SectionType = Field(..., description="Type of track section")
    is_occupied: bool = Field(default=False, description="Whether the section is currently occupied")
    occupying_train_id: Optional[str] = Field(None, description="ID of the train occupying this section")
    start_position_km: float = Field(..., ge=0, description="Starting position of this section on the overall track")
    end_position_km: float = Field(..., ge=0, description="Ending position of this section on the overall track")
    max_speed_kmh: float = Field(default=100.0, description="Maximum allowed speed in this section")
    
    class Config:
        json_schema_extra = {
            "example": {
                "section_id": "SEC-01A",
                "length_km": 2.5,
                "section_type": "mainline",
                "is_occupied": True,
                "occupying_train_id": "RJD-12345",
                "start_position_km": 0.0,
                "end_position_km": 2.5,
                "max_speed_kmh": 120.0
            }
        }


class Station(BaseModel):
    """Model representing a railway station"""
    station_id: str = Field(..., description="Unique identifier for the station")
    station_name: str = Field(..., description="Display name of the station")
    position_km: float = Field(..., ge=0, description="Position of the station on the track")
    platform_sections: List[str] = Field(default=[], description="List of platform section IDs")
    is_junction: bool = Field(default=False, description="Whether this station is a junction")


class Junction(BaseModel):
    """Model representing a railway junction"""
    junction_id: str = Field(..., description="Unique identifier for the junction")
    position_km: float = Field(..., ge=0, description="Position of the junction on the track")
    connected_sections: List[str] = Field(..., description="List of connected track section IDs")
    current_route: Optional[str] = Field(None, description="Currently active route through the junction")


class TrafficControlDecision(BaseModel):
    """Model representing an AI traffic control decision"""
    train_id: str = Field(..., description="ID of the train this decision applies to")
    decision_type: str = Field(..., description="Type of decision: 'speed_change', 'route_change', 'stop', 'proceed'")
    target_speed_kmh: Optional[float] = Field(None, description="Target speed for speed change decisions")
    reason: str = Field(..., description="Human-readable reason for this decision")
    priority_override: bool = Field(default=False, description="Whether this decision overrides normal priority rules")


class SimulationState(BaseModel):
    """Model representing the complete state of the simulation"""
    timestamp: datetime = Field(default_factory=datetime.now, description="Current simulation timestamp")
    trains: List[Train] = Field(default=[], description="List of all trains in the simulation")
    track_sections: List[TrackSection] = Field(default=[], description="List of all track sections")
    stations: List[Station] = Field(default=[], description="List of all stations")
    junctions: List[Junction] = Field(default=[], description="List of all junctions")
    ai_recommendation: Optional[str] = Field(None, description="Current AI recommendation for traffic control")
    active_decisions: List[TrafficControlDecision] = Field(default=[], description="Currently active traffic control decisions")
    timetable: List["TimetableEntry"] = Field(default=[], description="List of timetable entries for the simulation")
    simulation_speed: float = Field(default=1.0, description="Simulation speed multiplier")
    is_running: bool = Field(default=False, description="Whether the simulation is currently running")
    
    # Enhanced features
    active_emergencies: List[EmergencyEvent] = Field(default=[], description="Active emergency events")
    system_mode: SystemMode = Field(default=SystemMode.NORMAL, description="Current system operation mode")
    weather_conditions: Optional[WeatherCondition] = Field(None, description="Current weather conditions")
    available_routes: List[RouteAlternative] = Field(default=[], description="Available alternative routes")
    cascading_events: List[CascadingEvent] = Field(default=[], description="Active cascading events")
    prediction_models: Dict[str, PredictionModel] = Field(default={}, description="Active prediction models")
    backup_power_active: bool = Field(default=False, description="Whether backup power is active")
    external_data_feeds: Dict[str, Any] = Field(default={}, description="External data feeds (weather, news, etc.)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-01-15T10:30:00",
                "trains": [],
                "track_sections": [],
                "stations": [],
                "junctions": [],
                "ai_recommendation": "Train A should reduce speed to 60 km/h to avoid conflict with Train B at Station 2",
                "active_decisions": [],
                "timetable": [],
                "simulation_speed": 1.0,
                "is_running": True
            }
        }


class ConflictPrediction(BaseModel):
    """Model representing a predicted conflict between trains"""
    conflict_id: str = Field(..., description="Unique identifier for this conflict")
    train1_id: str = Field(..., description="ID of the first train involved")
    train2_id: str = Field(..., description="ID of the second train involved")
    conflict_location_km: float = Field(..., description="Position where conflict will occur")
    estimated_time_to_conflict: float = Field(..., description="Estimated time to conflict in minutes")
    severity: str = Field(..., description="Severity level: 'low', 'medium', 'high', 'critical'")
    recommended_action: str = Field(..., description="Recommended action to resolve the conflict")


# Additional utility models for API responses
class SimulationStatus(BaseModel):
    """Status information about the simulation"""
    is_running: bool
    current_time: datetime
    total_trains: int
    active_conflicts: int
    simulation_speed: float


class TrainUpdate(BaseModel):
    """Model for updating train parameters"""
    train_id: str
    speed_kmh: Optional[float] = None
    destination_station: Optional[str] = None
    priority: Optional[int] = None


class TimetableEntry(BaseModel):
    """Represents a scheduled stop for a train at a station"""
    train_id: str = Field(..., description="Train ID this entry applies to")
    station_id: str = Field(..., description="Station ID for the stop")
    arrival_time: Optional[datetime] = Field(None, description="Planned arrival time (ISO 8601)")
    departure_time: Optional[datetime] = Field(None, description="Planned departure time (ISO 8601)")


class TimetableLoadRequest(BaseModel):
    """Payload for loading timetable entries via API"""
    entries: List[TimetableEntry] = Field(default_factory=list, description="List of timetable entries")


# Emergency Management Models

class EmergencyType(str, Enum):
    """Types of emergency situations"""
    SIGNAL_FAILURE = "signal_failure"
    TRACK_BLOCKAGE = "track_blockage"
    WEATHER_DISRUPTION = "weather_disruption"
    EQUIPMENT_FAILURE = "equipment_failure"
    POWER_OUTAGE = "power_outage"
    MEDICAL_EMERGENCY = "medical_emergency"
    SECURITY_INCIDENT = "security_incident"


class EmergencySeverity(str, Enum):
    """Emergency severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EmergencyEvent(BaseModel):
    """Model representing an emergency event"""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique emergency event ID")
    event_type: EmergencyType = Field(..., description="Type of emergency")
    severity: EmergencySeverity = Field(..., description="Severity level")
    location_km: float = Field(..., description="Location of emergency on track")
    affected_sections: List[str] = Field(default=[], description="List of affected track section IDs")
    affected_trains: List[str] = Field(default=[], description="List of affected train IDs")
    description: str = Field(..., description="Description of the emergency")
    start_time: datetime = Field(default_factory=datetime.now, description="When emergency started")
    estimated_duration_minutes: Optional[int] = Field(None, description="Estimated duration in minutes")
    is_active: bool = Field(default=True, description="Whether emergency is still active")
    fallback_procedures: List[str] = Field(default=[], description="Active fallback procedures")
    external_factors: Dict[str, Any] = Field(default={}, description="External factors like weather data")


class WeatherCondition(BaseModel):
    """Weather condition affecting operations"""
    condition_type: str = Field(..., description="Type of weather condition")
    severity: int = Field(..., ge=1, le=5, description="Severity from 1 (mild) to 5 (extreme)")
    visibility_km: Optional[float] = Field(None, description="Visibility in kilometers")
    wind_speed_kmh: Optional[float] = Field(None, description="Wind speed in km/h")
    temperature_celsius: Optional[float] = Field(None, description="Temperature in Celsius")
    precipitation_mm: Optional[float] = Field(None, description="Precipitation in mm/hour")
    speed_reduction_factor: float = Field(default=1.0, description="Speed reduction factor (0.0-1.0)")


class RouteAlternative(BaseModel):
    """Alternative route for trains"""
    route_id: str = Field(..., description="Unique route identifier")
    section_sequence: List[str] = Field(..., description="Sequence of section IDs")
    total_distance_km: float = Field(..., description="Total route distance")
    estimated_time_minutes: float = Field(..., description="Estimated travel time")
    capacity_trains: int = Field(..., description="Maximum trains this route can handle")
    current_load: int = Field(default=0, description="Current number of trains on route")
    is_available: bool = Field(default=True, description="Whether route is available")
    priority_score: float = Field(..., description="Route priority score (higher is better)")


class SystemMode(str, Enum):
    """System operation modes"""
    NORMAL = "normal"
    DEGRADED = "degraded"
    EMERGENCY = "emergency"
    BACKUP_POWER = "backup_power"
    MANUAL_OVERRIDE = "manual_override"


class PredictionModel(BaseModel):
    """Advanced prediction model configuration"""
    model_type: str = Field(..., description="Type of prediction model")
    confidence_threshold: float = Field(default=0.8, description="Minimum confidence for predictions")
    prediction_horizon_minutes: int = Field(default=30, description="Prediction time horizon")
    factors: Dict[str, float] = Field(default={}, description="Weighting factors for different variables")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last model update time")


class CascadingEvent(BaseModel):
    """Model for cascading event analysis"""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique event ID")
    trigger_event_id: str = Field(..., description="ID of the triggering event")
    affected_trains: List[str] = Field(..., description="List of affected train IDs")
    propagation_path: List[str] = Field(..., description="Path of event propagation")
    estimated_delay_minutes: Dict[str, float] = Field(..., description="Estimated delays per train")
    mitigation_actions: List[str] = Field(default=[], description="Actions to mitigate cascading effects")
    confidence_score: float = Field(..., description="Confidence in cascade prediction")


