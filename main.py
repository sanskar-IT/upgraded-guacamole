import asyncio
import json
from contextlib import asynccontextmanager
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

from models import (
    SimulationState, Train, TrackSection, Station, Junction,
    SimulationStatus, TrainUpdate, ConflictPrediction, TrafficControlDecision,
    TimetableEntry, TimetableLoadRequest
)
from train_simulation import TrainSimulation


# Global instances
simulation = TrainSimulation()
websocket_connections: List[WebSocket] = []

# Background task to update simulation
async def simulation_loop():
    """Main simulation loop running in the background"""
    while True:
        if simulation.current_state.is_running:
            await simulation.update_simulation(1.0)  # 1 second time step
        await asyncio.sleep(1.0 / simulation.current_state.simulation_speed)  # Adjustable speed

# Lifespan manager to start background tasks
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the simulation loop
    task = asyncio.create_task(simulation_loop())
    
    # Set initial speeds for trains
    for train in simulation.current_state.trains:
        train.current_speed_kmh = 60.0  # Start with moderate speed
    
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

# FastAPI app initialization
app = FastAPI(
    title="Train Traffic Control Simulation API",
    description="Train traffic control simulation with AI-powered conflict resolution",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                self.active_connections.remove(connection)

websocket_manager = WebSocketManager()

# API Routes

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve a simple HTML dashboard"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Train Traffic Control Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .control-panel { background: #f5f5f5; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
            .status { background: #e7f3ff; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
            .train { background: #fff; border: 1px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 3px; }
            .button { padding: 10px 20px; margin: 5px; background: #007bff; color: white; border: none; border-radius: 3px; cursor: pointer; }
            .button:hover { background: #0056b3; }
            .stop { background: #dc3545; }
            .stop:hover { background: #c82333; }
        </style>
    </head>
    <body>
        <h1>🚂 Train Traffic Control Simulation</h1>
        
        <div class="control-panel">
            <h3>Simulation Controls</h3>
            <button class="button" onclick="startSimulation()">▶️ Start</button>
            <button class="button stop" onclick="stopSimulation()">⏹️ Stop</button>
            <button class="button" onclick="resetSimulation()">🔄 Reset</button>
            <label for="speed">Speed: </label>
            <input type="range" id="speed" min="0.1" max="5" step="0.1" value="1" onchange="setSpeed(this.value)">
            <span id="speedValue">1.0x</span>
        </div>
        
        <div id="status" class="status">
            <h3>System Status</h3>
            <p id="statusText">Loading...</p>
        </div>
        
        <div id="trains">
            <h3>Train Status</h3>
            <div id="trainList">Loading trains...</div>
        </div>
        
        <div id="ai-recommendation">
            <h3>🤖 AI Traffic Controller</h3>
            <p id="aiText">Analyzing traffic...</p>
        </div>
        
        <script>
            const API_BASE = '';
            
            async function startSimulation() {
                await fetch(API_BASE + '/simulation/start', {method: 'POST'});
                updateStatus();
            }
            
            async function stopSimulation() {
                await fetch(API_BASE + '/simulation/stop', {method: 'POST'});
                updateStatus();
            }
            
            async function resetSimulation() {
                await fetch(API_BASE + '/simulation/reset', {method: 'POST'});
                updateStatus();
            }
            
            async function setSpeed(value) {
                document.getElementById('speedValue').textContent = value + 'x';
                await fetch(API_BASE + '/simulation/speed', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({speed: parseFloat(value)})
                });
            }
            
            async function updateStatus() {
                try {
                    const response = await fetch(API_BASE + '/simulation/state');
                    const data = await response.json();
                    
                    document.getElementById('statusText').innerHTML = `
                        <strong>Running:</strong> ${data.is_running ? '✅ Yes' : '❌ No'}<br>
                        <strong>Trains:</strong> ${data.trains.length}<br>
                        <strong>Speed:</strong> ${data.simulation_speed}x<br>
                        <strong>Time:</strong> ${new Date(data.timestamp).toLocaleTimeString()}
                    `;
                    
                    const trainList = document.getElementById('trainList');
                    trainList.innerHTML = data.trains.map(train => `
                        <div class="train">
                            <strong>${train.train_name}</strong> (${train.train_id})<br>
                            Position: ${train.position_km.toFixed(2)} km | 
                            Speed: ${train.current_speed_kmh.toFixed(1)} km/h | 
                            Priority: ${train.priority} |
                            Destination: ${train.destination_station || 'None'}
                        </div>
                    `).join('');
                    
                    document.getElementById('aiText').textContent = data.ai_recommendation || 'No recommendations';
                    
                } catch (error) {
                    console.error('Error updating status:', error);
                }
            }
            
            // Update status every 2 seconds
            setInterval(updateStatus, 2000);
            updateStatus();
        </script>
    </body>
    </html>
    """
    return html_content

@app.get("/simulation/state", response_model=SimulationState)
async def get_simulation_state():
    """Get the current simulation state"""
    return simulation.current_state

@app.get("/simulation/status", response_model=dict)
async def get_simulation_status():
    """Get simulation status summary"""
    return simulation.get_simulation_status()

@app.post("/simulation/start")
async def start_simulation():
    """Start the simulation"""
    simulation.start_simulation()
    await websocket_manager.broadcast({"type": "simulation_started", "status": "running"})
    return {"message": "Simulation started"}

@app.post("/simulation/stop")
async def stop_simulation():
    """Stop the simulation"""
    simulation.stop_simulation()
    await websocket_manager.broadcast({"type": "simulation_stopped", "status": "stopped"})
    return {"message": "Simulation stopped"}

@app.post("/simulation/reset")
async def reset_simulation():
    """Reset simulation to initial state"""
    simulation.reset_simulation()
    # Set initial speeds again
    for train in simulation.current_state.trains:
        train.current_speed_kmh = 60.0
    await websocket_manager.broadcast({"type": "simulation_reset", "status": "reset"})
    return {"message": "Simulation reset"}

@app.post("/simulation/speed")
async def set_simulation_speed(speed_data: dict):
    """Set simulation speed multiplier"""
    speed = speed_data.get("speed", 1.0)
    simulation.set_simulation_speed(speed)
    await websocket_manager.broadcast({"type": "speed_changed", "speed": speed})
    return {"message": f"Simulation speed set to {speed}x"}

@app.get("/trains", response_model=List[Train])
async def get_all_trains():
    """Get all trains in the simulation"""
    return simulation.current_state.trains

@app.get("/trains/{train_id}", response_model=Train)
async def get_train(train_id: str):
    """Get specific train by ID"""
    train = next((t for t in simulation.current_state.trains if t.train_id == train_id), None)
    if not train:
        raise HTTPException(status_code=404, detail="Train not found")
    return train

@app.post("/trains/{train_id}/speed")
async def update_train_speed(train_id: str, speed_data: dict):
    """Update train speed manually"""
    new_speed = speed_data.get("speed_kmh", 0.0)
    train = next((t for t in simulation.current_state.trains if t.train_id == train_id), None)
    if not train:
        raise HTTPException(status_code=404, detail="Train not found")
    
    simulation.update_train_speed(train_id, new_speed)
    await websocket_manager.broadcast({
        "type": "train_speed_updated", 
        "train_id": train_id, 
        "new_speed": new_speed
    })
    return {"message": f"Train {train_id} speed updated to {new_speed} km/h"}

@app.get("/tracks", response_model=List[TrackSection])
async def get_all_track_sections():
    """Get all track sections"""
    return simulation.current_state.track_sections

@app.get("/stations", response_model=List[Station])
async def get_all_stations():
    """Get all stations"""
    return simulation.current_state.stations

@app.get("/junctions", response_model=List[Junction])
async def get_all_junctions():
    """Get all junctions"""
    return simulation.current_state.junctions

@app.get("/conflicts", response_model=List[ConflictPrediction])
async def get_predicted_conflicts(horizon_minutes: int = 15):
    """Get predicted conflicts within the specified time horizon"""
    return simulation.traffic_controller.predict_conflicts(
        simulation.current_state.trains, 
        horizon_minutes
    )

@app.get("/decisions", response_model=List[TrafficControlDecision])
async def get_active_decisions():
    """Get currently active traffic control decisions"""
    return simulation.current_state.active_decisions

@app.post("/emergency/stop_all")
async def emergency_stop_all():
    """Emergency stop all trains"""
    for train in simulation.current_state.trains:
        train.current_speed_kmh = 0.0
    
    await websocket_manager.broadcast({"type": "emergency_stop", "message": "All trains stopped"})
    return {"message": "Emergency stop activated - all trains stopped"}

@app.get("/analytics/summary")
async def get_analytics_summary():
    """Get simulation analytics and statistics"""
    state = simulation.current_state
    conflicts = simulation.traffic_controller.predict_conflicts(state.trains, 30)
    
    # Calculate various metrics
    total_trains = len(state.trains)
    running_trains = len([t for t in state.trains if t.current_speed_kmh > 0])
    avg_speed = sum(t.current_speed_kmh for t in state.trains) / total_trains if total_trains > 0 else 0
    occupied_sections = len([s for s in state.track_sections if s.is_occupied])
    
    # Priority distribution
    priority_distribution = {}
    for train in state.trains:
        priority_distribution[train.priority] = priority_distribution.get(train.priority, 0) + 1
    
    return {
        "total_trains": total_trains,
        "running_trains": running_trains,
        "stopped_trains": total_trains - running_trains,
        "average_speed_kmh": round(avg_speed, 2),
        "occupied_track_sections": occupied_sections,
        "total_track_sections": len(state.track_sections),
        "track_utilization_percent": round((occupied_sections / len(state.track_sections)) * 100, 1),
        "predicted_conflicts": len(conflicts),
        "critical_conflicts": len([c for c in conflicts if c.severity == "critical"]),
        "priority_distribution": priority_distribution,
        "simulation_speed": state.simulation_speed,
        "is_running": state.is_running
    }

# Basic API endpoints matching your specification

@app.get("/train_status", response_model=List[Train])
async def get_train_status():
    """GET /train_status → positions, ETAs, priorities"""
    return simulation.current_state.trains

@app.post("/resolve_conflict")
async def resolve_conflict(conflict_data: dict):
    """POST /resolve_conflict → get AI decision + reason"""
    conflicts = simulation.traffic_controller.predict_conflicts(simulation.current_state.trains, 15)
    
    if not conflicts:
        return {
            "message": "No conflicts detected",
            "ai_decisions": []
        }
    
    # Generate basic recommendations for conflicts
    decisions = []
    for conflict in conflicts:
        train1 = next((t for t in simulation.current_state.trains if t.train_id == conflict.train1_id), None)
        train2 = next((t for t in simulation.current_state.trains if t.train_id == conflict.train2_id), None)
        
        if train1 and train2:
            # Simple priority-based decision
            if train1.priority < train2.priority:  # Lower number = higher priority
                decisions.append({
                    "train_id": train2.train_id,
                    "recommended_action": "slow_down",
                    "reason": f"Higher priority train {train1.train_id} approaching",
                    "confidence_score": 0.85
                })
            else:
                decisions.append({
                    "train_id": train1.train_id,
                    "recommended_action": "slow_down", 
                    "reason": f"Higher priority train {train2.train_id} approaching",
                    "confidence_score": 0.85
                })
    
    return {
        "conflicts_analyzed": len(conflicts),
        "ai_decisions": decisions
    }

@app.post("/simulate_scenario")
async def simulate_scenario(scenario_data: dict):
    """POST /simulate_scenario → run what-if, return KPIs"""
    # Simple scenario simulation
    train_id = scenario_data.get("delay_train")
    delay_factor = scenario_data.get("delay_factor", 0.8)
    
    original_kpis = await get_analytics_summary()
    
    # Apply temporary scenario
    if train_id:
        train = next((t for t in simulation.current_state.trains if t.train_id == train_id), None)
        if train:
            original_speed = train.current_speed_kmh
            train.current_speed_kmh *= delay_factor
            
            # Get new KPIs
            modified_kpis = await get_analytics_summary()
            
            # Restore original speed
            train.current_speed_kmh = original_speed
            
            return {
                "scenario": f"Delayed {train_id} by factor {delay_factor}",
                "original_kpis": original_kpis,
                "modified_kpis": modified_kpis,
                "impact": "Temporary speed reduction applied"
            }
    
    return {
        "scenario": "No changes applied",
        "original_kpis": original_kpis
    }

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time simulation updates"""
    await websocket_manager.connect(websocket)
    try:
        while True:
            # Send current state every 2 seconds
            state_data = {
                "type": "state_update",
                "data": simulation.current_state.model_dump(),
                "timestamp": simulation.current_state.timestamp.isoformat()
            }
            await websocket.send_text(json.dumps(state_data, default=str))
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "simulation_running": simulation.current_state.is_running,
        "total_trains": len(simulation.current_state.trains),
        "timestamp": simulation.current_state.timestamp
    }

if __name__ == "__main__":
    print("🚂 Starting Train Traffic Control Simulation API...")
    print("Dashboard will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
