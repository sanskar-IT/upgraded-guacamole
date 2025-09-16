import asyncio
import numpy as np
import math
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import json

from models import (
    Train, TrackSection, ConflictPrediction, PredictionModel,
    EmergencyEvent, WeatherCondition, CascadingEvent
)


@dataclass
class PredictionFeatures:
    """Features used for advanced prediction models"""
    train_speed: float
    train_position: float
    train_priority: int
    section_occupancy: float
    weather_factor: float
    time_of_day: float
    historical_delay: float
    junction_congestion: float
    emergency_proximity: float


class NonLinearPredictor:
    """Advanced non-linear prediction system for train behavior"""
    
    def __init__(self):
        self.models: Dict[str, PredictionModel] = {}
        self.historical_data: List[Dict] = []
        self.feature_weights = {
            "speed_variance": 0.15,
            "human_factor": 0.10,
            "weather_impact": 0.20,
            "congestion_effect": 0.25,
            "emergency_influence": 0.30
        }
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize prediction models"""
        self.models = {
            "conflict_prediction": PredictionModel(
                model_type="neural_network",
                confidence_threshold=0.85,
                prediction_horizon_minutes=45,
                factors={
                    "speed_differential": 0.3,
                    "distance_factor": 0.25,
                    "priority_influence": 0.2,
                    "environmental_factor": 0.15,
                    "historical_pattern": 0.1
                }
            ),
            "delay_propagation": PredictionModel(
                model_type="cascade_model",
                confidence_threshold=0.75,
                prediction_horizon_minutes=60,
                factors={
                    "network_density": 0.4,
                    "bottleneck_proximity": 0.3,
                    "weather_conditions": 0.2,
                    "emergency_events": 0.1
                }
            ),
            "human_behavior": PredictionModel(
                model_type="behavioral_model",
                confidence_threshold=0.70,
                prediction_horizon_minutes=30,
                factors={
                    "operator_fatigue": 0.25,
                    "stress_level": 0.20,
                    "experience_factor": 0.20,
                    "time_pressure": 0.15,
                    "environmental_stress": 0.20
                }
            )
        }
    
    def extract_features(self, train: Train, trains: List[Train], 
                        track_sections: List[TrackSection],
                        weather: Optional[WeatherCondition] = None,
                        emergencies: List[EmergencyEvent] = None) -> PredictionFeatures:
        """Extract features for prediction models"""
        
        # Calculate section occupancy
        current_section = next(
            (s for s in track_sections if s.section_id == train.current_track_section_id), 
            None
        )
        section_occupancy = 0.0
        if current_section:
            nearby_trains = sum(
                1 for t in trains 
                if abs(t.position_km - train.position_km) <= 5.0 and t.train_id != train.train_id
            )
            section_occupancy = min(1.0, nearby_trains / 3.0)  # Normalize to 0-1
        
        # Weather factor
        weather_factor = 1.0
        if weather:
            weather_factor = weather.speed_reduction_factor
        
        # Time of day factor (affects human behavior)
        current_hour = datetime.now().hour
        time_of_day = math.sin(2 * math.pi * current_hour / 24)  # Sinusoidal pattern
        
        # Historical delay (simplified)
        historical_delay = 0.0  # Would be calculated from historical data
        
        # Junction congestion
        junction_congestion = 0.0
        for section in track_sections:
            if "JCT" in section.section_id and abs(section.start_position_km - train.position_km) <= 10.0:
                nearby_trains = sum(
                    1 for t in trains 
                    if abs(t.position_km - section.start_position_km) <= 5.0
                )
                junction_congestion = max(junction_congestion, min(1.0, nearby_trains / 2.0))
        
        # Emergency proximity
        emergency_proximity = 0.0
        if emergencies:
            for emergency in emergencies:
                distance = abs(emergency.location_km - train.position_km)
                if distance <= 15.0:
                    proximity_factor = 1.0 - (distance / 15.0)
                    severity_multiplier = {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}
                    emergency_proximity = max(
                        emergency_proximity, 
                        proximity_factor * severity_multiplier.get(emergency.severity.value, 0.5)
                    )
        
        return PredictionFeatures(
            train_speed=train.current_speed_kmh,
            train_position=train.position_km,
            train_priority=train.priority,
            section_occupancy=section_occupancy,
            weather_factor=weather_factor,
            time_of_day=time_of_day,
            historical_delay=historical_delay,
            junction_congestion=junction_congestion,
            emergency_proximity=emergency_proximity
        )
    
    def predict_advanced_conflicts(self, trains: List[Train], 
                                 track_sections: List[TrackSection],
                                 weather: Optional[WeatherCondition] = None,
                                 emergencies: List[EmergencyEvent] = None,
                                 horizon_minutes: int = 45) -> List[ConflictPrediction]:
        """Advanced conflict prediction using non-linear models"""
        conflicts = []
        model = self.models["conflict_prediction"]
        
        for i, train1 in enumerate(trains):
            for j, train2 in enumerate(trains[i+1:], start=i+1):
                conflict = self._predict_nonlinear_conflict(
                    train1, train2, trains, track_sections, 
                    weather, emergencies, horizon_minutes
                )
                if conflict:
                    conflicts.append(conflict)
        
        return conflicts
    
    def _predict_nonlinear_conflict(self, train1: Train, train2: Train,
                                  all_trains: List[Train], track_sections: List[TrackSection],
                                  weather: Optional[WeatherCondition],
                                  emergencies: List[EmergencyEvent],
                                  horizon_minutes: int) -> Optional[ConflictPrediction]:
        """Predict conflict using non-linear behavior models"""
        
        # Extract features for both trains
        features1 = self.extract_features(train1, all_trains, track_sections, weather, emergencies)
        features2 = self.extract_features(train2, all_trains, track_sections, weather, emergencies)
        
        # Non-linear speed prediction
        predicted_speed1 = self._predict_nonlinear_speed(train1, features1)
        predicted_speed2 = self._predict_nonlinear_speed(train2, features2)
        
        # Calculate future positions with non-linear behavior
        time_horizon_hours = horizon_minutes / 60.0
        
        # Account for acceleration/deceleration patterns
        future_pos1 = self._calculate_nonlinear_position(
            train1, predicted_speed1, time_horizon_hours, features1
        )
        future_pos2 = self._calculate_nonlinear_position(
            train2, predicted_speed2, time_horizon_hours, features2
        )
        
        # Dynamic safe distance based on conditions
        base_safe_distance = 0.5  # 500m base
        weather_multiplier = 1.0 / (weather.speed_reduction_factor if weather else 1.0)
        emergency_multiplier = 1.0 + features1.emergency_proximity + features2.emergency_proximity
        human_factor_multiplier = 1.0 + abs(features1.time_of_day) * 0.2  # Higher at night
        
        safe_distance = base_safe_distance * weather_multiplier * emergency_multiplier * human_factor_multiplier
        
        if abs(future_pos1 - future_pos2) < safe_distance:
            # Calculate more accurate time to conflict
            time_to_conflict = self._calculate_nonlinear_conflict_time(
                train1, train2, predicted_speed1, predicted_speed2, features1, features2
            )
            
            if time_to_conflict <= horizon_minutes:
                # Calculate confidence based on model factors
                confidence = self._calculate_prediction_confidence(features1, features2)
                
                severity = self._calculate_enhanced_severity(
                    time_to_conflict, features1, features2, weather, emergencies
                )
                
                conflict_location = self._estimate_conflict_location(
                    train1, train2, predicted_speed1, predicted_speed2, time_to_conflict
                )
                
                return ConflictPrediction(
                    conflict_id=f"NL_CONFLICT_{train1.train_id}_{train2.train_id}_{int(datetime.now().timestamp())}",
                    train1_id=train1.train_id,
                    train2_id=train2.train_id,
                    conflict_location_km=conflict_location,
                    estimated_time_to_conflict=time_to_conflict,
                    severity=severity,
                    recommended_action=self._generate_advanced_resolution(
                        train1, train2, features1, features2, weather, emergencies
                    )
                )
        
        return None
    
    def _predict_nonlinear_speed(self, train: Train, features: PredictionFeatures) -> float:
        """Predict train speed using non-linear behavior model"""
        base_speed = train.current_speed_kmh
        
        # Human factor variations
        human_variance = 1.0 + (features.time_of_day * 0.1)  # Night operations slower
        
        # Weather impact
        weather_impact = features.weather_factor
        
        # Congestion response (non-linear)
        congestion_factor = 1.0 - (features.section_occupancy ** 1.5) * 0.3
        
        # Emergency response (exponential decay with distance)
        emergency_factor = 1.0 - (features.emergency_proximity ** 0.8) * 0.4
        
        # Junction approach behavior
        junction_factor = 1.0 - (features.junction_congestion ** 1.2) * 0.25
        
        predicted_speed = (base_speed * human_variance * weather_impact * 
                          congestion_factor * emergency_factor * junction_factor)
        
        return max(0.0, min(predicted_speed, train.max_speed_kmh))
    
    def _calculate_nonlinear_position(self, train: Train, predicted_speed: float,
                                    time_hours: float, features: PredictionFeatures) -> float:
        """Calculate future position with non-linear behavior"""
        base_distance = predicted_speed * time_hours
        
        # Add acceleration/deceleration curves
        if predicted_speed > train.current_speed_kmh:
            # Acceleration curve (logarithmic approach to target speed)
            accel_factor = 1.0 - math.exp(-time_hours * 2)  # Exponential approach
            actual_distance = base_distance * accel_factor
        elif predicted_speed < train.current_speed_kmh:
            # Deceleration curve
            decel_factor = 1.0 - math.exp(-time_hours * 1.5)
            actual_distance = base_distance * (1.0 + (1.0 - decel_factor) * 0.3)
        else:
            actual_distance = base_distance
        
        # Add stochastic variations based on human factors
        human_variation = 1.0 + (features.time_of_day * 0.05 * math.sin(time_hours * math.pi))
        
        return train.position_km + (actual_distance * human_variation)
    
    def _calculate_nonlinear_conflict_time(self, train1: Train, train2: Train,
                                         speed1: float, speed2: float,
                                         features1: PredictionFeatures,
                                         features2: PredictionFeatures) -> float:
        """Calculate time to conflict with non-linear behavior"""
        
        # Basic relative motion calculation
        relative_speed = abs(speed1 - speed2)
        distance_gap = abs(train1.position_km - train2.position_km)
        
        if relative_speed <= 0.1:  # Nearly same speed
            return float('inf')
        
        base_time = distance_gap / relative_speed * 60  # Convert to minutes
        
        # Adjust for non-linear behavior
        # Trains tend to adjust speeds as they approach each other
        behavioral_adjustment = 1.0 + (features1.emergency_proximity + features2.emergency_proximity) * 0.2
        
        # Weather delays
        weather_delay = (2.0 - features1.weather_factor) * 0.1
        
        # Junction delays
        junction_delay = max(features1.junction_congestion, features2.junction_congestion) * 5.0
        
        adjusted_time = base_time * behavioral_adjustment + weather_delay + junction_delay
        
        return max(0.0, adjusted_time)
    
    def _calculate_prediction_confidence(self, features1: PredictionFeatures,
                                       features2: PredictionFeatures) -> float:
        """Calculate confidence in conflict prediction"""
        base_confidence = 0.8
        
        # Reduce confidence for high uncertainty conditions
        weather_uncertainty = 1.0 - abs(1.0 - features1.weather_factor) * 0.3
        emergency_uncertainty = 1.0 - max(features1.emergency_proximity, features2.emergency_proximity) * 0.2
        congestion_uncertainty = 1.0 - max(features1.section_occupancy, features2.section_occupancy) * 0.15
        
        confidence = base_confidence * weather_uncertainty * emergency_uncertainty * congestion_uncertainty
        
        return max(0.5, min(1.0, confidence))
    
    def _calculate_enhanced_severity(self, time_to_conflict: float,
                                   features1: PredictionFeatures, features2: PredictionFeatures,
                                   weather: Optional[WeatherCondition],
                                   emergencies: List[EmergencyEvent]) -> str:
        """Calculate enhanced severity considering multiple factors"""
        
        # Base severity from time
        if time_to_conflict < 3:
            base_severity = "critical"
        elif time_to_conflict < 8:
            base_severity = "high"
        elif time_to_conflict < 15:
            base_severity = "medium"
        else:
            base_severity = "low"
        
        # Escalate based on conditions
        escalation_factors = 0
        
        # Weather escalation
        if weather and weather.speed_reduction_factor < 0.7:
            escalation_factors += 1
        
        # Emergency proximity escalation
        if max(features1.emergency_proximity, features2.emergency_proximity) > 0.5:
            escalation_factors += 1
        
        # High congestion escalation
        if max(features1.section_occupancy, features2.section_occupancy) > 0.7:
            escalation_factors += 1
        
        # Junction congestion escalation
        if max(features1.junction_congestion, features2.junction_congestion) > 0.6:
            escalation_factors += 1
        
        # Escalate severity
        severity_levels = ["low", "medium", "high", "critical"]
        current_index = severity_levels.index(base_severity)
        escalated_index = min(len(severity_levels) - 1, current_index + escalation_factors)
        
        return severity_levels[escalated_index]
    
    def _estimate_conflict_location(self, train1: Train, train2: Train,
                                  speed1: float, speed2: float, time_minutes: float) -> float:
        """Estimate where conflict will occur"""
        time_hours = time_minutes / 60.0
        
        future_pos1 = train1.position_km + speed1 * time_hours
        future_pos2 = train2.position_km + speed2 * time_hours
        
        return (future_pos1 + future_pos2) / 2.0
    
    def _generate_advanced_resolution(self, train1: Train, train2: Train,
                                    features1: PredictionFeatures, features2: PredictionFeatures,
                                    weather: Optional[WeatherCondition],
                                    emergencies: List[EmergencyEvent]) -> str:
        """Generate advanced conflict resolution recommendations"""
        
        # Analyze priority and conditions
        priority_diff = abs(train1.priority - train2.priority)
        
        if priority_diff > 1:
            # Clear priority difference
            higher_priority_train = train1 if train1.priority < train2.priority else train2
            lower_priority_train = train2 if train1.priority < train2.priority else train1
            
            action = f"Train {lower_priority_train.train_id} should reduce speed to 70% and increase following distance"
            
            # Add weather considerations
            if weather and weather.speed_reduction_factor < 0.8:
                action += f" (weather conditions require extra caution: {weather.condition_type})"
            
            # Add emergency considerations
            emergency_proximity = max(features1.emergency_proximity, features2.emergency_proximity)
            if emergency_proximity > 0.3:
                action += " and implement emergency spacing protocols"
            
            return action
        
        else:
            # Similar priority - use advanced logic
            congestion1 = features1.section_occupancy + features1.junction_congestion
            congestion2 = features2.section_occupancy + features2.junction_congestion
            
            if congestion1 > congestion2:
                action = f"Train {train1.train_id} should use alternative route or reduce speed"
            elif congestion2 > congestion1:
                action = f"Train {train2.train_id} should use alternative route or reduce speed"
            else:
                # Position-based decision
                trailing_train = train1 if train1.position_km < train2.position_km else train2
                action = f"Train {trailing_train.train_id} should implement adaptive speed control"
            
            # Add contextual information
            if weather:
                action += f" considering {weather.condition_type} conditions"
            
            return action
    
    def predict_cascading_delays(self, initial_delay: float, affected_trains: List[Train],
                               track_sections: List[TrackSection]) -> Dict[str, float]:
        """Predict cascading delay propagation using non-linear models"""
        delay_predictions = {}
        
        # Sort trains by position
        sorted_trains = sorted(affected_trains, key=lambda t: t.position_km)
        
        current_delay = initial_delay
        
        for i, train in enumerate(sorted_trains):
            # Base delay propagation
            delay_predictions[train.train_id] = current_delay
            
            # Calculate delay propagation to next train
            if i < len(sorted_trains) - 1:
                next_train = sorted_trains[i + 1]
                distance_gap = next_train.position_km - train.position_km
                
                # Non-linear delay propagation
                propagation_factor = math.exp(-distance_gap / 10.0)  # Exponential decay
                
                # Adjust for train priorities
                priority_factor = 1.0 + (next_train.priority - train.priority) * 0.1
                
                # Network congestion effect
                congestion_multiplier = 1.0 + len(affected_trains) / 10.0
                
                current_delay = current_delay * propagation_factor * priority_factor * congestion_multiplier
                current_delay = max(1.0, current_delay)  # Minimum 1 minute delay
        
        return delay_predictions
    
    def update_model_with_feedback(self, model_name: str, prediction: Dict, actual_outcome: Dict):
        """Update prediction models based on actual outcomes"""
        if model_name not in self.models:
            return
        
        # Store historical data for model improvement
        feedback_data = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "prediction": prediction,
            "actual": actual_outcome,
            "accuracy": self._calculate_accuracy(prediction, actual_outcome)
        }
        
        self.historical_data.append(feedback_data)
        
        # Update model confidence based on recent accuracy
        recent_feedback = [d for d in self.historical_data[-50:] if d["model"] == model_name]
        if recent_feedback:
            avg_accuracy = sum(d["accuracy"] for d in recent_feedback) / len(recent_feedback)
            self.models[model_name].confidence_threshold = max(0.5, min(0.95, avg_accuracy))
    
    def _calculate_accuracy(self, prediction: Dict, actual: Dict) -> float:
        """Calculate prediction accuracy"""
        # Simplified accuracy calculation
        if "time_to_conflict" in prediction and "actual_time" in actual:
            predicted_time = prediction["time_to_conflict"]
            actual_time = actual["actual_time"]
            
            if actual_time == 0:  # No conflict occurred
                return 1.0 if predicted_time > 60 else 0.0
            
            error_ratio = abs(predicted_time - actual_time) / actual_time
            return max(0.0, 1.0 - error_ratio)
        
        return 0.5  # Default accuracy for unknown cases


class ExternalDataIntegrator:
    """Integrates external data feeds for enhanced prediction"""
    
    def __init__(self):
        self.data_sources = {
            "weather_api": "https://api.weather.com/v1/current",
            "traffic_feed": "https://api.traffic.gov/incidents",
            "news_feed": "https://api.news.com/transport"
        }
        self.cache = {}
        self.last_update = {}
    
    async def fetch_weather_data(self, location: str) -> Optional[WeatherCondition]:
        """Fetch real-time weather data"""
        # Simulate weather API call
        # In real implementation, would use actual weather API
        
        cache_key = f"weather_{location}"
        if (cache_key in self.cache and 
            datetime.now() - self.last_update.get(cache_key, datetime.min) < timedelta(minutes=15)):
            return self.cache[cache_key]
        
        # Simulate weather conditions
        import random
        conditions = ["clear", "light_rain", "heavy_rain", "fog", "snow", "high_wind"]
        condition_type = random.choice(conditions)
        severity = random.randint(1, 4)
        
        weather = WeatherCondition(
            condition_type=condition_type,
            severity=severity,
            speed_reduction_factor=max(0.3, 1.0 - (severity * 0.15)),
            visibility_km=max(0.5, 10.0 - severity * 2),
            wind_speed_kmh=severity * 10 if condition_type == "high_wind" else None,
            precipitation_mm=severity * 5 if "rain" in condition_type else None
        )
        
        self.cache[cache_key] = weather
        self.last_update[cache_key] = datetime.now()
        
        return weather
    
    async def fetch_traffic_incidents(self) -> List[Dict]:
        """Fetch traffic and transport incidents"""
        # Simulate traffic incident API
        incidents = []
        
        # Generate random incidents for simulation
        import random
        for _ in range(random.randint(0, 3)):
            incidents.append({
                "id": f"INC_{random.randint(1000, 9999)}",
                "type": random.choice(["accident", "maintenance", "signal_failure", "medical"]),
                "location_km": random.uniform(0, 40),
                "severity": random.choice(["low", "medium", "high"]),
                "estimated_duration": random.randint(15, 120)
            })
        
        return incidents
    
    async def fetch_news_alerts(self) -> List[Dict]:
        """Fetch transport-related news and alerts"""
        # Simulate news feed
        alerts = []
        
        import random
        news_types = ["strike_alert", "weather_warning", "infrastructure_update", "security_alert"]
        
        for _ in range(random.randint(0, 2)):
            alerts.append({
                "id": f"NEWS_{random.randint(1000, 9999)}",
                "type": random.choice(news_types),
                "priority": random.choice(["low", "medium", "high"]),
                "content": "Simulated news alert content",
                "timestamp": datetime.now().isoformat(),
                "affects_operations": random.choice([True, False])
            })
        
        return alerts
    
    def integrate_external_factors(self, weather: Optional[WeatherCondition],
                                 incidents: List[Dict], news: List[Dict]) -> Dict[str, Any]:
        """Integrate all external factors into operational parameters"""
        
        integrated_factors = {
            "weather_impact": 1.0,
            "incident_impact": 1.0,
            "news_impact": 1.0,
            "overall_risk_level": "low",
            "recommended_adjustments": []
        }
        
        # Weather impact
        if weather:
            integrated_factors["weather_impact"] = weather.speed_reduction_factor
            if weather.severity >= 3:
                integrated_factors["recommended_adjustments"].append(
                    f"Implement weather protocols for {weather.condition_type}"
                )
        
        # Incident impact
        high_severity_incidents = [i for i in incidents if i.get("severity") == "high"]
        if high_severity_incidents:
            integrated_factors["incident_impact"] = 0.7
            integrated_factors["recommended_adjustments"].append(
                "Multiple high-severity incidents detected - implement contingency protocols"
            )
        
        # News impact
        operational_news = [n for n in news if n.get("affects_operations")]
        if operational_news:
            high_priority_news = [n for n in operational_news if n.get("priority") == "high"]
            if high_priority_news:
                integrated_factors["news_impact"] = 0.8
                integrated_factors["recommended_adjustments"].append(
                    "High-priority operational alerts - review emergency procedures"
                )
        
        # Calculate overall risk level
        risk_factors = [
            integrated_factors["weather_impact"],
            integrated_factors["incident_impact"],
            integrated_factors["news_impact"]
        ]
        
        avg_risk = sum(risk_factors) / len(risk_factors)
        
        if avg_risk < 0.6:
            integrated_factors["overall_risk_level"] = "high"
        elif avg_risk < 0.8:
            integrated_factors["overall_risk_level"] = "medium"
        else:
            integrated_factors["overall_risk_level"] = "low"
        
        return integrated_factors
