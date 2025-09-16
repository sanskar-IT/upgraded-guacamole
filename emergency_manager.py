import asyncio
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Set
import math

from models import (
    EmergencyEvent, EmergencyType, EmergencySeverity, WeatherCondition,
    CascadingEvent, Train, TrackSection, SystemMode, RouteAlternative,
    TrafficControlDecision
)


class EmergencyManager:
    """Advanced emergency management system for train traffic control"""
    
    def __init__(self):
        self.active_emergencies: List[EmergencyEvent] = []
        self.emergency_protocols: Dict[EmergencyType, Dict] = self._initialize_protocols()
        self.cascading_analyzer = CascadingEventAnalyzer()
        self.route_planner = DynamicRoutePlanner()
        
    def _initialize_protocols(self) -> Dict[EmergencyType, Dict]:
        """Initialize emergency response protocols"""
        return {
            EmergencyType.SIGNAL_FAILURE: {
                "fallback_procedures": ["manual_control", "reduced_speed", "visual_signals"],
                "speed_reduction": 0.5,
                "safe_distance_multiplier": 2.0,
                "requires_manual_override": True
            },
            EmergencyType.TRACK_BLOCKAGE: {
                "fallback_procedures": ["route_diversion", "emergency_stop", "manual_clearance"],
                "speed_reduction": 0.0,
                "safe_distance_multiplier": 3.0,
                "requires_manual_override": True
            },
            EmergencyType.WEATHER_DISRUPTION: {
                "fallback_procedures": ["speed_reduction", "increased_spacing", "visibility_protocols"],
                "speed_reduction": 0.7,
                "safe_distance_multiplier": 1.5,
                "requires_manual_override": False
            },
            EmergencyType.EQUIPMENT_FAILURE: {
                "fallback_procedures": ["backup_systems", "degraded_operation", "maintenance_mode"],
                "speed_reduction": 0.6,
                "safe_distance_multiplier": 1.8,
                "requires_manual_override": True
            },
            EmergencyType.POWER_OUTAGE: {
                "fallback_procedures": ["backup_power", "battery_operation", "emergency_stop"],
                "speed_reduction": 0.3,
                "safe_distance_multiplier": 2.5,
                "requires_manual_override": True
            }
        }
    
    def create_emergency(self, event_type: EmergencyType, location_km: float, 
                        severity: EmergencySeverity, description: str,
                        affected_sections: List[str] = None,
                        estimated_duration: int = None) -> EmergencyEvent:
        """Create a new emergency event"""
        emergency = EmergencyEvent(
            event_type=event_type,
            severity=severity,
            location_km=location_km,
            description=description,
            affected_sections=affected_sections or [],
            estimated_duration_minutes=estimated_duration
        )
        
        # Apply emergency protocols
        protocol = self.emergency_protocols.get(event_type, {})
        emergency.fallback_procedures = protocol.get("fallback_procedures", [])
        
        self.active_emergencies.append(emergency)
        return emergency
    
    def assess_emergency_impact(self, emergency: EmergencyEvent, trains: List[Train], 
                              track_sections: List[TrackSection]) -> Dict:
        """Assess the impact of an emergency on trains and operations"""
        impact_assessment = {
            "affected_trains": [],
            "blocked_sections": [],
            "required_diversions": [],
            "estimated_delays": {},
            "cascading_risks": [],
            "recommended_actions": []
        }
        
        # Find affected trains
        for train in trains:
            distance_to_emergency = abs(train.position_km - emergency.location_km)
            
            # Check if train is in danger zone
            if distance_to_emergency <= 5.0:  # 5km danger zone
                impact_assessment["affected_trains"].append(train.train_id)
                
                # Calculate estimated delay
                protocol = self.emergency_protocols.get(emergency.event_type, {})
                speed_reduction = protocol.get("speed_reduction", 0.8)
                
                if speed_reduction == 0.0:  # Complete stop required
                    delay_minutes = emergency.estimated_duration_minutes or 60
                else:
                    # Calculate delay based on speed reduction
                    original_time = distance_to_emergency / train.current_speed_kmh * 60
                    new_speed = train.current_speed_kmh * speed_reduction
                    new_time = distance_to_emergency / max(new_speed, 10) * 60
                    delay_minutes = new_time - original_time
                
                impact_assessment["estimated_delays"][train.train_id] = delay_minutes
        
        # Find blocked sections
        for section in track_sections:
            if (section.start_position_km <= emergency.location_km <= section.end_position_km or
                section.section_id in emergency.affected_sections):
                impact_assessment["blocked_sections"].append(section.section_id)
        
        # Generate recommended actions
        impact_assessment["recommended_actions"] = self._generate_emergency_actions(
            emergency, impact_assessment["affected_trains"], trains
        )
        
        # Analyze cascading risks
        cascading_events = self.cascading_analyzer.analyze_cascading_effects(
            emergency, trains, track_sections
        )
        impact_assessment["cascading_risks"] = cascading_events
        
        return impact_assessment
    
    def _generate_emergency_actions(self, emergency: EmergencyEvent, 
                                  affected_train_ids: List[str], 
                                  trains: List[Train]) -> List[TrafficControlDecision]:
        """Generate emergency response actions"""
        actions = []
        protocol = self.emergency_protocols.get(emergency.event_type, {})
        
        for train_id in affected_train_ids:
            train = next((t for t in trains if t.train_id == train_id), None)
            if not train:
                continue
            
            if emergency.event_type == EmergencyType.TRACK_BLOCKAGE:
                # Emergency stop for track blockage
                actions.append(TrafficControlDecision(
                    train_id=train_id,
                    decision_type="emergency_stop",
                    target_speed_kmh=0.0,
                    reason=f"Emergency stop due to {emergency.event_type.value} at {emergency.location_km}km",
                    priority_override=True
                ))
            
            elif emergency.event_type == EmergencyType.SIGNAL_FAILURE:
                # Reduce speed and increase following distance
                target_speed = train.current_speed_kmh * protocol.get("speed_reduction", 0.5)
                actions.append(TrafficControlDecision(
                    train_id=train_id,
                    decision_type="speed_change",
                    target_speed_kmh=target_speed,
                    reason=f"Speed reduction due to signal failure at {emergency.location_km}km",
                    priority_override=True
                ))
            
            elif emergency.event_type == EmergencyType.WEATHER_DISRUPTION:
                # Weather-based speed reduction
                target_speed = train.current_speed_kmh * protocol.get("speed_reduction", 0.7)
                actions.append(TrafficControlDecision(
                    train_id=train_id,
                    decision_type="speed_change",
                    target_speed_kmh=target_speed,
                    reason=f"Weather-related speed reduction at {emergency.location_km}km",
                    priority_override=False
                ))
        
        return actions
    
    def resolve_emergency(self, emergency_id: str) -> bool:
        """Mark an emergency as resolved"""
        for emergency in self.active_emergencies:
            if emergency.event_id == emergency_id:
                emergency.is_active = False
                self.active_emergencies.remove(emergency)
                return True
        return False
    
    def get_system_mode(self) -> SystemMode:
        """Determine current system mode based on active emergencies"""
        if not self.active_emergencies:
            return SystemMode.NORMAL
        
        critical_emergencies = [e for e in self.active_emergencies 
                              if e.severity == EmergencySeverity.CRITICAL]
        
        if critical_emergencies:
            # Check for power outages
            power_outages = [e for e in critical_emergencies 
                           if e.event_type == EmergencyType.POWER_OUTAGE]
            if power_outages:
                return SystemMode.BACKUP_POWER
            return SystemMode.EMERGENCY
        
        high_severity = [e for e in self.active_emergencies 
                        if e.severity == EmergencySeverity.HIGH]
        if high_severity:
            return SystemMode.DEGRADED
        
        return SystemMode.NORMAL


class CascadingEventAnalyzer:
    """Analyzes and predicts cascading effects of emergencies"""
    
    def analyze_cascading_effects(self, trigger_event: EmergencyEvent, 
                                trains: List[Train], 
                                track_sections: List[TrackSection]) -> List[CascadingEvent]:
        """Analyze potential cascading effects from an emergency"""
        cascading_events = []
        
        # Analyze delay propagation
        delay_cascade = self._analyze_delay_propagation(trigger_event, trains)
        if delay_cascade:
            cascading_events.append(delay_cascade)
        
        # Analyze resource contention
        resource_cascade = self._analyze_resource_contention(trigger_event, trains, track_sections)
        if resource_cascade:
            cascading_events.append(resource_cascade)
        
        # Analyze junction congestion
        junction_cascade = self._analyze_junction_effects(trigger_event, trains, track_sections)
        if junction_cascade:
            cascading_events.append(junction_cascade)
        
        return cascading_events
    
    def _analyze_delay_propagation(self, trigger_event: EmergencyEvent, 
                                 trains: List[Train]) -> Optional[CascadingEvent]:
        """Analyze how delays propagate through the network"""
        affected_trains = []
        delay_estimates = {}
        propagation_path = []
        
        # Sort trains by position
        sorted_trains = sorted(trains, key=lambda t: t.position_km)
        
        # Find initial affected trains
        initial_affected = []
        for train in sorted_trains:
            if abs(train.position_km - trigger_event.location_km) <= 5.0:
                initial_affected.append(train)
        
        if not initial_affected:
            return None
        
        # Propagate delays through following trains
        base_delay = 30.0  # Base delay in minutes
        
        for i, train in enumerate(sorted_trains):
            if train in initial_affected:
                delay_estimates[train.train_id] = base_delay
                affected_trains.append(train.train_id)
                propagation_path.append(train.train_id)
                
                # Propagate to following trains
                for j in range(i + 1, len(sorted_trains)):
                    following_train = sorted_trains[j]
                    distance_gap = following_train.position_km - train.position_km
                    
                    if distance_gap <= 10.0:  # Within influence range
                        propagated_delay = base_delay * (1 - distance_gap / 10.0) * 0.6
                        if propagated_delay > 5.0:  # Significant delay
                            delay_estimates[following_train.train_id] = propagated_delay
                            affected_trains.append(following_train.train_id)
                            propagation_path.append(following_train.train_id)
        
        if len(affected_trains) > 1:
            return CascadingEvent(
                trigger_event_id=trigger_event.event_id,
                affected_trains=affected_trains,
                propagation_path=propagation_path,
                estimated_delay_minutes=delay_estimates,
                mitigation_actions=[
                    "Increase train spacing",
                    "Implement dynamic speed control",
                    "Activate alternative routes"
                ],
                confidence_score=0.8
            )
        
        return None
    
    def _analyze_resource_contention(self, trigger_event: EmergencyEvent,
                                   trains: List[Train], 
                                   track_sections: List[TrackSection]) -> Optional[CascadingEvent]:
        """Analyze resource contention effects"""
        # Find bottleneck sections
        bottlenecks = []
        for section in track_sections:
            if section.section_type.value in ["station", "siding"]:
                # Count trains approaching this section
                approaching_trains = 0
                for train in trains:
                    distance_to_section = abs(train.position_km - section.start_position_km)
                    if distance_to_section <= 15.0 and train.current_speed_kmh > 0:
                        approaching_trains += 1
                
                if approaching_trains > 2:  # Potential bottleneck
                    bottlenecks.append(section.section_id)
        
        if bottlenecks:
            affected_trains = []
            for train in trains:
                for section_id in bottlenecks:
                    section = next((s for s in track_sections if s.section_id == section_id), None)
                    if section and abs(train.position_km - section.start_position_km) <= 15.0:
                        affected_trains.append(train.train_id)
            
            if len(affected_trains) > 2:
                return CascadingEvent(
                    trigger_event_id=trigger_event.event_id,
                    affected_trains=list(set(affected_trains)),
                    propagation_path=bottlenecks,
                    estimated_delay_minutes={train_id: 15.0 for train_id in affected_trains},
                    mitigation_actions=[
                        "Implement queue management",
                        "Activate bypass routes",
                        "Coordinate train priorities"
                    ],
                    confidence_score=0.7
                )
        
        return None
    
    def _analyze_junction_effects(self, trigger_event: EmergencyEvent,
                                trains: List[Train], 
                                track_sections: List[TrackSection]) -> Optional[CascadingEvent]:
        """Analyze junction congestion effects"""
        junction_sections = [s for s in track_sections if "JCT" in s.section_id]
        
        for junction_section in junction_sections:
            trains_near_junction = []
            for train in trains:
                distance = abs(train.position_km - junction_section.start_position_km)
                if distance <= 8.0:  # Within junction influence
                    trains_near_junction.append(train)
            
            if len(trains_near_junction) > 1:
                # Potential junction congestion
                affected_train_ids = [t.train_id for t in trains_near_junction]
                
                return CascadingEvent(
                    trigger_event_id=trigger_event.event_id,
                    affected_trains=affected_train_ids,
                    propagation_path=[junction_section.section_id],
                    estimated_delay_minutes={
                        train_id: 10.0 + random.uniform(0, 15) 
                        for train_id in affected_train_ids
                    },
                    mitigation_actions=[
                        "Implement junction priority control",
                        "Coordinate train sequencing",
                        "Use alternative junction routes"
                    ],
                    confidence_score=0.75
                )
        
        return None


class DynamicRoutePlanner:
    """Plans alternative routes during emergencies"""
    
    def __init__(self):
        self.route_cache: Dict[str, List[RouteAlternative]] = {}
    
    def find_alternative_routes(self, blocked_sections: List[str], 
                              track_sections: List[TrackSection],
                              origin_km: float, destination_km: float) -> List[RouteAlternative]:
        """Find alternative routes avoiding blocked sections"""
        alternatives = []
        
        # Create main route avoiding blocked sections
        available_sections = [s for s in track_sections 
                            if s.section_id not in blocked_sections]
        
        # Sort sections by position
        available_sections.sort(key=lambda s: s.start_position_km)
        
        # Build route sequence
        route_sections = []
        current_pos = origin_km
        
        for section in available_sections:
            if section.start_position_km >= current_pos and section.end_position_km <= destination_km:
                route_sections.append(section.section_id)
                current_pos = section.end_position_km
        
        if route_sections:
            total_distance = sum(
                next(s.length_km for s in track_sections if s.section_id == sid)
                for sid in route_sections
            )
            
            # Calculate priority score based on distance and section types
            priority_score = 100.0 - (total_distance * 2)  # Prefer shorter routes
            
            # Bonus for mainline sections
            mainline_bonus = sum(
                10 if next(s.section_type.value for s in track_sections 
                          if s.section_id == sid) == "mainline" else 0
                for sid in route_sections
            )
            priority_score += mainline_bonus
            
            alternative = RouteAlternative(
                route_id=f"ALT_{len(alternatives) + 1}",
                section_sequence=route_sections,
                total_distance_km=total_distance,
                estimated_time_minutes=total_distance / 60.0 * 60,  # Assume 60 km/h average
                capacity_trains=3,  # Conservative estimate
                priority_score=priority_score
            )
            
            alternatives.append(alternative)
        
        # Try siding routes if available
        siding_sections = [s for s in available_sections 
                          if s.section_type.value == "siding"]
        
        if siding_sections:
            for siding in siding_sections:
                if origin_km <= siding.start_position_km <= destination_km:
                    siding_route = RouteAlternative(
                        route_id=f"SIDING_{siding.section_id}",
                        section_sequence=[siding.section_id],
                        total_distance_km=siding.length_km,
                        estimated_time_minutes=siding.length_km / 40.0 * 60,  # Slower on sidings
                        capacity_trains=1,  # Limited capacity
                        priority_score=50.0  # Lower priority
                    )
                    alternatives.append(siding_route)
        
        return sorted(alternatives, key=lambda r: r.priority_score, reverse=True)
    
    def optimize_route_assignment(self, trains: List[Train], 
                                routes: List[RouteAlternative]) -> Dict[str, str]:
        """Optimize assignment of trains to alternative routes"""
        assignments = {}
        
        # Sort trains by priority (lower number = higher priority)
        sorted_trains = sorted(trains, key=lambda t: t.priority)
        
        # Sort routes by priority score
        sorted_routes = sorted(routes, key=lambda r: r.priority_score, reverse=True)
        
        for train in sorted_trains:
            for route in sorted_routes:
                if route.current_load < route.capacity_trains and route.is_available:
                    assignments[train.train_id] = route.route_id
                    route.current_load += 1
                    break
        
        return assignments


class WeatherIntegration:
    """Integrates weather data and environmental factors"""
    
    def __init__(self):
        self.weather_impacts = {
            "heavy_rain": {"speed_factor": 0.7, "visibility_factor": 0.6},
            "snow": {"speed_factor": 0.5, "visibility_factor": 0.4},
            "fog": {"speed_factor": 0.8, "visibility_factor": 0.3},
            "high_wind": {"speed_factor": 0.6, "visibility_factor": 0.9},
            "ice": {"speed_factor": 0.3, "visibility_factor": 0.8}
        }
    
    def create_weather_condition(self, condition_type: str, severity: int) -> WeatherCondition:
        """Create weather condition with appropriate impacts"""
        impacts = self.weather_impacts.get(condition_type, {"speed_factor": 1.0, "visibility_factor": 1.0})
        
        # Adjust impacts based on severity
        severity_multiplier = severity / 3.0  # Normalize to 0-1.67 range
        speed_reduction = 1.0 - ((1.0 - impacts["speed_factor"]) * severity_multiplier)
        
        return WeatherCondition(
            condition_type=condition_type,
            severity=severity,
            speed_reduction_factor=max(0.2, speed_reduction),  # Minimum 20% speed
            visibility_km=max(0.5, 10.0 * impacts["visibility_factor"] / severity_multiplier),
            wind_speed_kmh=severity * 15.0 if condition_type == "high_wind" else None,
            precipitation_mm=severity * 10.0 if condition_type in ["heavy_rain", "snow"] else None
        )
    
    def assess_weather_impact(self, weather: WeatherCondition, trains: List[Train]) -> List[TrafficControlDecision]:
        """Generate traffic decisions based on weather conditions"""
        decisions = []
        
        for train in trains:
            # Apply weather-based speed reduction
            target_speed = train.current_speed_kmh * weather.speed_reduction_factor
            
            decisions.append(TrafficControlDecision(
                train_id=train.train_id,
                decision_type="speed_change",
                target_speed_kmh=target_speed,
                reason=f"Weather-based speed reduction: {weather.condition_type} (severity {weather.severity})",
                priority_override=False
            ))
        
        return decisions
