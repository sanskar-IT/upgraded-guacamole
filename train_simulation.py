import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import math

from models import (
    Train, TrackSection, SimulationState, Station, Junction,
    ConflictPrediction, TrafficControlDecision, SectionType, TimetableEntry
)
from track_config import TrackConfiguration


class TrafficController:
    """AI-powered traffic controller for train simulation"""
    
    def __init__(self, track_config: TrackConfiguration):
        self.track_config = track_config
        self.active_decisions: List[TrafficControlDecision] = []
        self.conflict_predictions: List[ConflictPrediction] = []
        
    def predict_conflicts(self, trains: List[Train], prediction_horizon_minutes: int = 30) -> List[ConflictPrediction]:
        """Predict potential conflicts between trains within the given time horizon"""
        conflicts = []
        
        for i, train1 in enumerate(trains):
            for j, train2 in enumerate(trains[i+1:], start=i+1):
                conflict = self._check_train_conflict(train1, train2, prediction_horizon_minutes)
                if conflict:
                    conflicts.append(conflict)
        
        return conflicts
    
    def _check_train_conflict(self, train1: Train, train2: Train, horizon_minutes: int) -> Optional[ConflictPrediction]:
        """Check if two trains will conflict within the prediction horizon"""
        # Simple linear prediction - in reality this would be more sophisticated
        time_horizon_hours = horizon_minutes / 60.0
        
        # Calculate future positions
        future_pos1 = train1.position_km + (train1.current_speed_kmh * time_horizon_hours)
        future_pos2 = train2.position_km + (train2.current_speed_kmh * time_horizon_hours)
        
        # Check if trains will be in the same location (within safe distance)
        safe_distance_km = 0.5  # 500 meters minimum separation
        
        if abs(future_pos1 - future_pos2) < safe_distance_km:
            # Calculate when conflict will occur
            relative_speed = abs(train1.current_speed_kmh - train2.current_speed_kmh)
            if relative_speed > 0:
                time_to_conflict = abs(train1.position_km - train2.position_km) / relative_speed * 60  # in minutes
            else:
                time_to_conflict = 0
            
            if time_to_conflict <= horizon_minutes:
                conflict_location = (train1.position_km + train2.position_km) / 2
                severity = self._calculate_conflict_severity(train1, train2, time_to_conflict)
                
                return ConflictPrediction(
                    conflict_id=f"CONFLICT_{train1.train_id}_{train2.train_id}_{int(time.time())}",
                    train1_id=train1.train_id,
                    train2_id=train2.train_id,
                    conflict_location_km=conflict_location,
                    estimated_time_to_conflict=time_to_conflict,
                    severity=severity,
                    recommended_action=self._generate_conflict_resolution(train1, train2)
                )
        
        return None
    
    def _calculate_conflict_severity(self, train1: Train, train2: Train, time_to_conflict: float) -> str:
        """Calculate the severity of a predicted conflict"""
        if time_to_conflict < 5:
            return "critical"
        elif time_to_conflict < 10:
            return "high"
        elif time_to_conflict < 20:
            return "medium"
        else:
            return "low"
    
    def _generate_conflict_resolution(self, train1: Train, train2: Train) -> str:
        """Generate recommended action to resolve conflict"""
        if train1.priority < train2.priority:
            return f"Train {train2.train_id} should reduce speed to allow Train {train1.train_id} to pass"
        elif train2.priority < train1.priority:
            return f"Train {train1.train_id} should reduce speed to allow Train {train2.train_id} to pass"
        else:
            return f"Train with lower position should reduce speed"
    
    def generate_traffic_decisions(self, trains: List[Train], track_sections: List[TrackSection], timetable: List[TimetableEntry] = None) -> List[TrafficControlDecision]:
        """Generate AI-powered traffic control decisions"""
        decisions = []
        conflicts = self.predict_conflicts(trains, 15)  # 15-minute horizon
        
        # Handle conflict resolution
        for conflict in conflicts:
            train1 = next((t for t in trains if t.train_id == conflict.train1_id), None)
            train2 = next((t for t in trains if t.train_id == conflict.train2_id), None)
            
            if train1 and train2:
                # Priority-based decision making
                if train1.priority < train2.priority:  # Lower number = higher priority
                    # Train 2 should slow down
                    target_speed = max(30.0, train2.current_speed_kmh * 0.7)
                    decisions.append(TrafficControlDecision(
                        train_id=train2.train_id,
                        decision_type="speed_change",
                        target_speed_kmh=target_speed,
                        reason=f"Priority conflict with {train1.train_id}. Reducing speed to maintain safe separation.",
                        priority_override=False
                    ))
                elif train2.priority < train1.priority:
                    # Train 1 should slow down
                    target_speed = max(30.0, train1.current_speed_kmh * 0.7)
                    decisions.append(TrafficControlDecision(
                        train_id=train1.train_id,
                        decision_type="speed_change",
                        target_speed_kmh=target_speed,
                        reason=f"Priority conflict with {train2.train_id}. Reducing speed to maintain safe separation.",
                        priority_override=False
                    ))
                else:
                    # Same priority - train behind should slow down
                    slower_train = train2 if train2.position_km < train1.position_km else train1
                    target_speed = max(30.0, slower_train.current_speed_kmh * 0.6)
                    decisions.append(TrafficControlDecision(
                        train_id=slower_train.train_id,
                        decision_type="speed_change",
                        target_speed_kmh=target_speed,
                        reason="Same priority conflict. Trailing train reducing speed to maintain separation.",
                        priority_override=False
                    ))
        
        # Check for timetable adherence
        if timetable:
            timetable_decisions = self._optimize_for_timetable(trains, timetable, track_sections)
            decisions.extend(timetable_decisions)
        
        # Check for station approach optimizations
        for train in trains:
            station_decisions = self._optimize_station_approach(train, track_sections)
            decisions.extend(station_decisions)
        
        return decisions
    
    def _optimize_station_approach(self, train: Train, track_sections: List[TrackSection]) -> List[TrafficControlDecision]:
        """Optimize train approach to stations"""
        decisions = []
        
        # Find upcoming station sections
        current_section = next((s for s in track_sections if s.section_id == train.current_track_section_id), None)
        if not current_section:
            return decisions
        
        # Look ahead for station sections
        upcoming_sections = self._get_upcoming_sections(train, track_sections, 5.0)  # 5km lookahead
        
        for section in upcoming_sections:
            if section.section_type == SectionType.STATION:
                distance_to_station = section.start_position_km - train.position_km
                
                # Calculate optimal speed for smooth station entry
                if distance_to_station > 0:
                    # If approaching too fast for station speed limit
                    required_deceleration_distance = self._calculate_stopping_distance(
                        train.current_speed_kmh, section.max_speed_kmh
                    )
                    
                    if distance_to_station <= required_deceleration_distance * 1.5:  # Start slowing 1.5x earlier
                        optimal_speed = min(
                            section.max_speed_kmh * 1.2,  # Slightly above station speed
                            train.current_speed_kmh * 0.8  # Gradual reduction
                        )
                        
                        decisions.append(TrafficControlDecision(
                            train_id=train.train_id,
                            decision_type="speed_change",
                            target_speed_kmh=optimal_speed,
                            reason=f"Optimizing approach to station section {section.section_id}",
                            priority_override=False
                        ))
                        break
        
        return decisions
    
    def _get_upcoming_sections(self, train: Train, track_sections: List[TrackSection], distance_km: float) -> List[TrackSection]:
        """Get track sections within the specified distance ahead of the train"""
        upcoming = []
        current_position = train.position_km
        max_position = current_position + distance_km
        
        for section in track_sections:
            if (section.start_position_km >= current_position and 
                section.start_position_km <= max_position):
                upcoming.append(section)
        
        return sorted(upcoming, key=lambda s: s.start_position_km)
    
    def _calculate_stopping_distance(self, current_speed_kmh: float, target_speed_kmh: float) -> float:
        """Calculate distance needed to decelerate from current speed to target speed"""
        if current_speed_kmh <= target_speed_kmh:
            return 0.0
        
        # Simple physics: v² = u² + 2as, assuming comfortable deceleration of 1 m/s²
        deceleration_ms2 = 1.0  # m/s²
        current_speed_ms = current_speed_kmh / 3.6
        target_speed_ms = target_speed_kmh / 3.6
        
        distance_m = (current_speed_ms ** 2 - target_speed_ms ** 2) / (2 * deceleration_ms2)
        return max(0.0, distance_m / 1000.0)  # Convert to km
    
    def _optimize_for_timetable(self, trains: List[Train], timetable: List[TimetableEntry], track_sections: List[TrackSection]) -> List[TrafficControlDecision]:
        """Generate decisions to optimize timetable adherence"""
        decisions = []
        current_time = datetime.now()
        
        for train in trains:
            # Find upcoming timetable entries for this train
            train_schedule = [
                entry for entry in timetable 
                if entry.train_id == train.train_id and 
                (entry.arrival_time is None or entry.arrival_time > current_time)
            ]
            
            if not train_schedule:
                continue
                
            # Sort by time
            train_schedule.sort(key=lambda x: x.arrival_time or datetime.max)
            next_stop = train_schedule[0]
            
            # Find the station for this stop
            from track_config import TrackConfiguration  # Import here to avoid circular dependency
            track_config = TrackConfiguration()
            try:
                station = next(s for s in track_config.stations if s.station_id == next_stop.station_id)
            except StopIteration:
                continue
                
            # Calculate time and distance to station
            distance_to_station = station.position_km - train.position_km
            
            if distance_to_station <= 0:
                continue  # Train has passed this station
                
            if next_stop.arrival_time:
                time_to_arrival = (next_stop.arrival_time - current_time).total_seconds() / 3600  # hours
                
                if time_to_arrival > 0:
                    # Calculate required average speed to meet timetable
                    required_avg_speed = distance_to_station / time_to_arrival
                    
                    # Apply speed adjustments based on schedule adherence
                    if required_avg_speed > train.max_speed_kmh * 1.1:
                        # Train is likely to be late - increase speed if safe
                        target_speed = min(train.max_speed_kmh, train.current_speed_kmh * 1.1)
                        decisions.append(TrafficControlDecision(
                            train_id=train.train_id,
                            decision_type="speed_change",
                            target_speed_kmh=target_speed,
                            reason=f"Increasing speed to meet scheduled arrival at {next_stop.station_id}",
                            priority_override=False
                        ))
                    elif required_avg_speed < train.current_speed_kmh * 0.8:
                        # Train is ahead of schedule - can reduce speed for efficiency
                        target_speed = max(40.0, required_avg_speed * 1.1)
                        decisions.append(TrafficControlDecision(
                            train_id=train.train_id,
                            decision_type="speed_change",
                            target_speed_kmh=target_speed,
                            reason=f"Reducing speed - ahead of schedule for {next_stop.station_id}",
                            priority_override=False
                        ))
                    elif time_to_arrival < 0.1:  # Less than 6 minutes
                        # Approaching scheduled time - prepare for station entry
                        target_speed = 50.0  # Moderate speed for station approach
                        decisions.append(TrafficControlDecision(
                            train_id=train.train_id,
                            decision_type="speed_change",
                            target_speed_kmh=target_speed,
                            reason=f"Preparing for scheduled arrival at {next_stop.station_id}",
                            priority_override=False
                        ))
        
        return decisions


class TrainSimulation:
    """Main simulation engine for train traffic control"""
    
    def __init__(self):
        self.track_config = TrackConfiguration()
        self.traffic_controller = TrafficController(self.track_config)
        self.current_state = SimulationState(
            trains=self.track_config.initial_trains.copy(),
            track_sections=self.track_config.track_sections.copy(),
            stations=self.track_config.stations.copy(),
            junctions=self.track_config.junctions.copy(),
            simulation_speed=1.0,
            is_running=False
        )
        self.last_update_time = time.time()
        self._update_track_occupancy()
    
    def _update_track_occupancy(self):
        """Update track section occupancy based on current train positions"""
        # Clear all occupancy
        for section in self.current_state.track_sections:
            section.is_occupied = False
            section.occupying_train_id = None
        
        # Set occupancy based on train positions
        for train in self.current_state.trains:
            try:
                section = self.track_config.get_section_by_position(train.position_km)
                section_in_state = next(
                    (s for s in self.current_state.track_sections if s.section_id == section.section_id), 
                    None
                )
                if section_in_state:
                    section_in_state.is_occupied = True
                    section_in_state.occupying_train_id = train.train_id
                    train.current_track_section_id = section.section_id
            except ValueError:
                continue  # Train position might be out of bounds
    
    async def update_simulation(self, dt_seconds: float):
        """Update simulation state by one time step"""
        if not self.current_state.is_running:
            return
        
        # Generate AI traffic control decisions
        decisions = self.traffic_controller.generate_traffic_decisions(
            self.current_state.trains, 
            self.current_state.track_sections,
            self.current_state.timetable
        )
        self.current_state.active_decisions = decisions
        
        # Apply traffic control decisions
        self._apply_traffic_decisions(decisions)
        
        # Update train positions and speeds
        for train in self.current_state.trains:
            await self._update_train(train, dt_seconds)
        
        # Update track occupancy
        self._update_track_occupancy()
        
        # Update simulation timestamp
        self.current_state.timestamp = datetime.now()
        
        # Generate AI recommendation
        self.current_state.ai_recommendation = self._generate_ai_recommendation()
    
    def _apply_traffic_decisions(self, decisions: List[TrafficControlDecision]):
        """Apply traffic control decisions to trains"""
        for decision in decisions:
            train = next((t for t in self.current_state.trains if t.train_id == decision.train_id), None)
            if train:
                if decision.decision_type == "speed_change" and decision.target_speed_kmh is not None:
                    # Gradual speed change for realism
                    speed_diff = decision.target_speed_kmh - train.current_speed_kmh
                    train.current_speed_kmh += speed_diff * 0.1  # 10% adjustment per update
                elif decision.decision_type == "stop":
                    train.current_speed_kmh = max(0.0, train.current_speed_kmh - 10.0)
    
    async def _update_train(self, train: Train, dt_seconds: float):
        """Update individual train position and speed"""
        if train.current_speed_kmh <= 0:
            return
        
        # Get current section and its speed limit
        try:
            current_section = self.track_config.get_section_by_position(train.position_km)
            max_section_speed = min(current_section.max_speed_kmh, train.max_speed_kmh)
        except ValueError:
            max_section_speed = train.max_speed_kmh
        
        # Enforce speed limits
        train.current_speed_kmh = min(train.current_speed_kmh, max_section_speed)
        
        # Update position
        distance_traveled_km = (train.current_speed_kmh / 3600.0) * dt_seconds * self.current_state.simulation_speed
        train.position_km += distance_traveled_km
        
        # Keep train within track bounds
        train.position_km = max(0.0, min(train.position_km, self.track_config.total_length_km))
        
        # Check if train reached destination
        if train.destination_station:
            station = next((s for s in self.current_state.stations if s.station_id == train.destination_station), None)
            if station and abs(train.position_km - station.position_km) < 0.1:
                train.current_speed_kmh = 0.0  # Stop at destination
    
    def _generate_ai_recommendation(self) -> str:
        """Generate human-readable AI recommendation"""
        conflicts = self.traffic_controller.predict_conflicts(self.current_state.trains, 10)
        
        if not conflicts:
            return "All trains operating normally. No conflicts detected."
        
        # Focus on the most critical conflict
        critical_conflicts = [c for c in conflicts if c.severity in ["critical", "high"]]
        if critical_conflicts:
            conflict = critical_conflicts[0]
            return f"URGENT: {conflict.recommended_action}. Time to conflict: {conflict.estimated_time_to_conflict:.1f} minutes."
        
        moderate_conflicts = [c for c in conflicts if c.severity == "medium"]
        if moderate_conflicts:
            conflict = moderate_conflicts[0]
            return f"ADVISORY: {conflict.recommended_action}. Time to conflict: {conflict.estimated_time_to_conflict:.1f} minutes."
        
        return f"Monitoring {len(conflicts)} potential conflicts. All situations under control."
    
    def start_simulation(self):
        """Start the simulation"""
        self.current_state.is_running = True
        self.last_update_time = time.time()
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.current_state.is_running = False
    
    def reset_simulation(self):
        """Reset simulation to initial state"""
        self.current_state = SimulationState(
            trains=self.track_config.initial_trains.copy(),
            track_sections=self.track_config.track_sections.copy(),
            stations=self.track_config.stations.copy(),
            junctions=self.track_config.junctions.copy(),
            simulation_speed=1.0,
            is_running=False
        )
        self._update_track_occupancy()
    
    def set_simulation_speed(self, speed_multiplier: float):
        """Set simulation speed multiplier"""
        self.current_state.simulation_speed = max(0.1, min(10.0, speed_multiplier))
    
    def update_train_speed(self, train_id: str, new_speed_kmh: float):
        """Manually update train speed"""
        train = next((t for t in self.current_state.trains if t.train_id == train_id), None)
        if train:
            train.current_speed_kmh = max(0.0, min(new_speed_kmh, train.max_speed_kmh))
    
    def get_simulation_status(self):
        """Get current simulation status"""
        return {
            "is_running": self.current_state.is_running,
            "current_time": self.current_state.timestamp,
            "total_trains": len(self.current_state.trains),
            "active_conflicts": len(self.traffic_controller.predict_conflicts(self.current_state.trains, 15)),
            "simulation_speed": self.current_state.simulation_speed
        }
