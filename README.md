# Advanced Train Traffic Control System

A comprehensive train traffic control simulation system with advanced AI-powered emergency management, cascading event prediction, dynamic routing, and real-time environmental integration.

## 🚀 Enhanced Features

### 🚨 Emergency Management System
- **Signal Failures**: Automatic fallback to manual control with reduced speed protocols
- **Track Blockages**: Dynamic route diversion and emergency stop procedures
- **Weather Disruptions**: Adaptive speed control based on visibility and precipitation
- **Equipment Failures**: Degraded mode operation with backup system activation
- **Power Outages**: Backup power protocols and battery operation modes
- **Medical/Security Emergencies**: Priority response and resource allocation

### 🔮 Advanced Prediction Engine
- **Non-Linear Behavior Modeling**: Accounts for human factors, fatigue, and experience levels
- **Cascading Event Analysis**: Predicts chain reactions and system-wide delay propagation
- **Machine Learning Integration**: Adaptive models that improve with historical data
- **Multi-Factor Conflict Prediction**: Weather, congestion, emergency proximity analysis
- **Confidence Scoring**: Reliability metrics for all predictions

### 🛤️ Dynamic Route Management
- **Alternative Route Planning**: Real-time calculation of optimal diversions
- **Junction Switching Logic**: Intelligent traffic flow optimization
- **Load Balancing**: Distributes trains across available routes
- **Capacity Management**: Monitors and manages route utilization
- **Priority-Based Routing**: High-priority trains get optimal paths

### 🌦️ Environmental Integration
- **Real-Time Weather Data**: API integration for current conditions
- **Speed Adaptation**: Automatic adjustments for rain, snow, fog, wind
- **Visibility Management**: Enhanced safety protocols for low visibility
- **Temperature Compensation**: Adjustments for extreme weather conditions

### 📊 External Data Integration
- **Weather APIs**: Live meteorological data integration
- **Traffic Incident Feeds**: Real-time incident monitoring
- **News Alert Processing**: Transport-related news and strike alerts
- **Risk Assessment**: Combined analysis of all external factors

## 📁 Project Structure

```
train-traffic-control-minimal/
├── main.py                    # FastAPI server with enhanced endpoints
├── models.py                  # Enhanced data models with emergency types
├── train_simulation.py        # Core simulation with advanced traffic controller
├── track_config.py           # Track and station configuration
├── emergency_manager.py      # Emergency response and cascading analysis
├── advanced_prediction.py    # Non-linear prediction and ML models
├── requirements.txt          # Python dependencies
└── README.md                # This documentation
```

## 🛠️ Installation & Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Server**
   ```bash
   python main.py
   ```

3. **Access the Dashboard**
   - Web Interface: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

## 🎮 System Operation Modes

### Normal Mode
- Standard operations with full AI assistance
- Predictive conflict resolution
- Optimal speed and routing decisions

### Degraded Mode
- Reduced functionality due to equipment issues
- Increased safety margins
- Manual override capabilities

### Emergency Mode
- Critical situation response
- Emergency stop protocols
- Priority resource allocation

### Backup Power Mode
- Limited functionality on backup systems
- Essential operations only
- Battery conservation protocols

## 🔧 API Endpoints

### Core Simulation
- `GET /simulation/state` - Current system state
- `POST /simulation/start` - Start simulation
- `POST /simulation/stop` - Stop simulation
- `POST /simulation/reset` - Reset to initial state

### Emergency Management
- `POST /emergency/create` - Create emergency event
- `GET /emergency/active` - List active emergencies
- `POST /emergency/resolve/{id}` - Resolve emergency
- `GET /emergency/impact/{id}` - Assess emergency impact

### Advanced Prediction
- `GET /prediction/conflicts` - Advanced conflict predictions
- `GET /prediction/cascading` - Cascading event analysis
- `POST /prediction/scenario` - What-if scenario analysis
- `GET /prediction/confidence` - Model confidence metrics

### Route Management
- `GET /routes/alternatives` - Available alternative routes
- `POST /routes/optimize` - Optimize route assignments
- `GET /routes/capacity` - Route capacity analysis

### External Data
- `GET /external/weather` - Current weather conditions
- `GET /external/incidents` - Traffic incidents
- `GET /external/news` - Transport news alerts
- `GET /external/risk-assessment` - Combined risk analysis

## 🧠 AI Traffic Controller Features

### Conflict Resolution
- **Priority-Based Decisions**: Respects train hierarchy
- **Environmental Adaptation**: Weather and emergency considerations
- **Predictive Spacing**: Maintains optimal train separation
- **Junction Optimization**: Efficient traffic flow through junctions

### Emergency Response
- **Automatic Detection**: Identifies potential emergency situations
- **Rapid Response**: Immediate safety protocol activation
- **Resource Coordination**: Optimal allocation of available resources
- **Communication**: Clear instructions and status updates

### Learning Capabilities
- **Historical Analysis**: Learns from past incidents and decisions
- **Pattern Recognition**: Identifies recurring operational patterns
- **Performance Optimization**: Continuously improves decision quality
- **Feedback Integration**: Incorporates operator feedback

## 📈 Performance Metrics

### Safety Metrics
- Conflict prevention rate
- Emergency response time
- Safety protocol compliance
- Near-miss incidents

### Efficiency Metrics
- Average train speed
- On-time performance
- Route utilization
- Energy consumption

### System Metrics
- Prediction accuracy
- Response time
- System availability
- Resource utilization

## 🔍 Monitoring & Analytics

### Real-Time Dashboard
- Live train positions and speeds
- Active conflicts and resolutions
- Emergency status and responses
- Weather and environmental conditions

### Historical Analysis
- Performance trends
- Incident patterns
- Efficiency improvements
- Predictive model accuracy

### Reporting
- Daily operational summaries
- Emergency response reports
- Performance benchmarks
- Compliance documentation

## 🚨 Emergency Scenarios Handled

1. **Signal System Failures**
   - Automatic fallback to manual control
   - Reduced speed protocols
   - Visual signal procedures

2. **Track Obstructions**
   - Immediate train stops
   - Alternative route calculation
   - Emergency clearance coordination

3. **Severe Weather Events**
   - Speed reductions based on conditions
   - Enhanced spacing requirements
   - Visibility-based protocols

4. **Equipment Malfunctions**
   - Degraded mode operation
   - Backup system activation
   - Maintenance coordination

5. **Power System Failures**
   - Battery backup activation
   - Essential systems prioritization
   - Emergency power protocols

## 🔧 Configuration

### System Parameters
- Prediction horizons (15-60 minutes)
- Safety margins (0.5-3.0 km)
- Speed reduction factors (0.3-1.0)
- Confidence thresholds (0.5-0.95)

### Emergency Protocols
- Response time targets
- Escalation procedures
- Communication protocols
- Resource allocation rules

### External Integrations
- Weather API endpoints
- Incident feed URLs
- News service connections
- Alert notification systems

## 🤝 Contributing

The system is designed for extensibility:
- Add new emergency types in `models.py`
- Implement custom prediction algorithms in `advanced_prediction.py`
- Extend route planning logic in `emergency_manager.py`
- Integrate additional external data sources

## 📝 License

This project is designed for educational and research purposes in railway traffic management and AI-powered transportation systems.

---

**Note**: This advanced system now handles complex emergency scenarios, multi-train cascading events, dynamic route changes, and advanced prediction capabilities that were previously unsupported.
