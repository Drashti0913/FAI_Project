"""
server.py — Flask backend for the delivery simulation demo.

Connects delivery_env.py + assignment.py to the live web UI.

Run:
    pip install flask
    python3 server.py

Then open:  http://localhost:5000
"""

import json
import random
from flask import Flask, jsonify, request, render_template_string

from delivery_env import DeliveryEnvironment, Order

try:
    from assignment import assign_order_to_driver, locally_optimize_queue
    USE_ASSIGNMENT = True
    print("[OK] Assignment heuristic loaded")
except ImportError:
    USE_ASSIGNMENT = False
    print("[WARN] assignment.py not found — using simple nearest-driver assignment")

app = Flask(__name__)

# ── Global simulation state ──────────────────────────────────────────
env   = None
stats = {"completed": 0, "ontime": 0, "total_wait": 0}

def make_env():
    global env, stats
    env   = DeliveryEnvironment(num_drivers=3, grid_size=5,
                                 traffic_pattern='rush_hour',
                                 order_arrival_rate=0.3)
    env.reset()
    stats = {"completed": 0, "ontime": 0, "total_wait": 0}

make_env()

# ── Helper: get edges with their current congestion status ───────────
def get_edge_data():
    edges = []
    hour  = (env.current_time // 60) % 24
    seen  = set()
    for (n1, n2) in env.graph.edges.keys():
        key = (min(n1,n2), max(n1,n2))
        if key in seen:
            continue
        seen.add(key)
        mult = env.graph.traffic_multipliers.get((n1,n2), {}).get(hour, 1.0)
        if mult == 1.0:
            mult = env.graph.traffic_multipliers.get((n2,n1), {}).get(hour, 1.0)
        base = env.graph.edges.get((n1,n2), env.graph.edges.get((n2,n1), 10))
        learned_key = (n1, n2)
        r_key = (n2, n1)
        learned_data = env.learned_traffic.get(learned_key) or env.learned_traffic.get(r_key)
        learned_time = None
        if learned_data and hour in learned_data and learned_data[hour]:
            learned_time = round(sum(learned_data[hour]) / len(learned_data[hour]), 1)
        edges.append({
            "from":         n1,
            "to":           n2,
            "base_time":    base,
            "multiplier":   mult,
            "congested":    mult > 1.0,
            "learned_time": learned_time,
        })
    return edges

# ── Helper: build full state snapshot ────────────────────────────────
def get_state():
    hour = (env.current_time // 60) % 24
    rush = hour in [8, 9, 10, 17, 18, 19]

    drivers = []
    for d in env.drivers:
        dest = None
        if d.order_queue:
            o = env.orders.get(d.order_queue[0])
            dest = o.destination if o else None
        drivers.append({
            "id":         d.id,
            "node":       d.current_node,
            "queue_len":  len(d.order_queue),
            "queue":      list(d.order_queue),
            "status":     d.status,
            "dest":       dest,
            "route":      list(d.route),
        })

    orders = []
    for o in env.orders.values():
        if not o.delivered:
            orders.append({
                "id":           o.id,
                "dest":         o.destination,
                "arrival":      o.arrival_time,
                "deadline":     o.deadline,
                "time_left":    o.deadline - env.current_time,
                "assigned_to":  o.assigned_driver,
                "urgent":       (o.deadline - env.current_time) < 10,
            })

    on_time_pct = 0
    avg_wait    = 0
    if stats["completed"] > 0:
        on_time_pct = round(stats["ontime"] / stats["completed"] * 100, 1)
        avg_wait    = round(stats["total_wait"] / stats["completed"], 1)

    return {
        "time":        env.current_time,
        "hour":        hour,
        "rush_hour":   rush,
        "hour_str":    f"{hour}:{env.current_time % 60:02d}",
        "drivers":     drivers,
        "orders":      orders,
        "pending":     len(env.pending_orders),
        "completed":   stats["completed"],
        "on_time_pct": on_time_pct,
        "avg_wait":    avg_wait,
        "edges":       get_edge_data(),
    }

# ── Step the simulation one tick ─────────────────────────────────────
def do_step():
    """
    Advances env one timestep using your real backend code.
    Generates orders via env internals, moves drivers, records deliveries.
    """
    prev_nodes = [d.current_node for d in env.drivers]
    events     = []

    # Generate new order
    if random.random() < env.order_arrival_rate:
        order = Order(
            id           = env.order_counter,
            destination  = random.randint(0, env.graph.num_nodes - 1),
            arrival_time = env.current_time,
            deadline     = env.current_time + 30,
        )
        env.orders[order.id] = order
        env.order_counter += 1

        # Use assignment heuristic if available, otherwise nearest driver
        if USE_ASSIGNMENT:
            driver_id, pos = assign_order_to_driver(order, env)
            order.assigned_driver = driver_id
            env.drivers[driver_id].order_queue.insert(pos, order.id)
            locally_optimize_queue(env.drivers[driver_id], env)
            events.append(f"Order {order.id} → Driver {driver_id} (heuristic, pos {pos})")
        else:
            def mdist(a, b): return abs(a//5-b//5)+abs(a%5-b%5)
            best = min(env.drivers,
                       key=lambda d: mdist(d.current_node, order.destination) + len(d.order_queue)*10)
            best.add_order(order.id)
            order.assigned_driver = best.id
            events.append(f"Order {order.id} → Driver {best.id} (nearest)")

        # Recompute route for assigned driver
        d = env.drivers[order.assigned_driver]
        _recompute_route(d)

    # Move drivers
    for driver in env.drivers:
        if driver.route and driver.status == 'delivering':
            next_node = driver.route.pop(0)
            hour      = (env.current_time // 60) % 24
            edge      = (driver.current_node, next_node)
            actual    = env.graph.get_edge_cost(driver.current_node, next_node, hour)
            env.learned_traffic[edge][hour].append(actual)
            driver.current_node = next_node

            # Check delivery
            if driver.order_queue:
                first = env.orders.get(driver.order_queue[0])
                if first and driver.current_node == first.destination:
                    oid = driver.order_queue.pop(0)
                    o   = env.orders[oid]
                    o.delivered     = True
                    o.delivery_time = env.current_time
                    wait = env.current_time - o.arrival_time
                    late = env.current_time > o.deadline
                    stats["completed"]  += 1
                    stats["total_wait"] += wait
                    if not late:
                        stats["ontime"] += 1
                    env.completed_orders.append(oid)
                    events.append(
                        f"Driver {driver.id} delivered order {oid} "
                        f"(wait {wait}min, {'LATE' if late else 'on-time'})"
                    )
                    _recompute_route(driver)

    env.current_time += 1
    return events

def _recompute_route(driver):
    """Simple BFS route to first queued order's destination."""
    if not driver.order_queue:
        driver.route  = []
        driver.status = 'idle'
        return
    dest = env.orders[driver.order_queue[0]].destination
    path = _bfs(driver.current_node, dest)
    driver.route  = path[1:]
    driver.status = 'delivering' if driver.route else 'idle'

def _bfs(start, goal):
    if start == goal:
        return [start]
    from collections import deque
    vis = {start: None}
    q   = deque([start])
    while q:
        n = q.popleft()
        for nb in env.graph.get_neighbors(n):
            if nb not in vis:
                vis[nb] = n
                if nb == goal:
                    path, c = [], goal
                    while c is not None:
                        path.append(c); c = vis[c]
                    return path[::-1]
                q.append(nb)
    return [start]

# ── Routes ────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/api/state')
def api_state():
    return jsonify(get_state())

@app.route('/api/step', methods=['POST'])
def api_step():
    n     = int(request.json.get('n', 1))
    events = []
    for _ in range(n):
        events.extend(do_step())
        if env.current_time >= 480 or len(env.completed_orders) >= 50:
            break
    return jsonify({**get_state(), "events": events[-10:]})

@app.route('/api/reset', methods=['POST'])
def api_reset():
    make_env()
    return jsonify(get_state())

@app.route('/api/set_time', methods=['POST'])
def api_set_time():
    """Jump to a specific hour for rush-hour demo."""
    hour = int(request.json.get('hour', 9))
    env.current_time = hour * 60
    return jsonify(get_state())

# ── HTML UI ───────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dynamic Delivery Optimization — Live Demo</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f5f5f5;color:#1a1a1a;font-size:14px}
header{background:#fff;border-bottom:1px solid #e0e0e0;padding:14px 24px;display:flex;align-items:center;gap:16px}
header h1{font-size:16px;font-weight:600;color:#1a1a1a}
header p{font-size:13px;color:#666;margin-top:1px}
.badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:500}
.badge-green{background:#e8f5e9;color:#2e7d32}
.badge-red{background:#ffebee;color:#c62828}
.badge-blue{background:#e3f2fd;color:#1565c0}
main{display:grid;grid-template-columns:1fr 280px;gap:16px;padding:16px 24px;max-width:1100px;margin:0 auto}
.grid-card{background:#fff;border-radius:12px;border:1px solid #e0e0e0;padding:16px}
.card{background:#fff;border-radius:12px;border:1px solid #e0e0e0;padding:14px;margin-bottom:12px}
.card h3{font-size:13px;font-weight:600;color:#555;margin-bottom:10px;text-transform:uppercase;letter-spacing:.4px}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:16px}
.stat{background:#fafafa;border:1px solid #eee;border-radius:8px;padding:10px 12px;text-align:center}
.stat-val{font-size:22px;font-weight:600;color:#1a1a1a}
.stat-lbl{font-size:11px;color:#888;margin-top:2px}
canvas{display:block;width:100%;border-radius:8px}
.controls{display:flex;align-items:center;gap:10px;margin-top:12px;flex-wrap:wrap}
button{padding:7px 16px;border-radius:8px;border:1px solid #ddd;background:#fff;cursor:pointer;font-size:13px;font-weight:500}
button:hover{background:#f5f5f5}
button.primary{background:#1976d2;border-color:#1565c0;color:#fff}
button.primary:hover{background:#1565c0}
button.danger{background:#e53935;border-color:#c62828;color:#fff}
input[type=range]{flex:1;accent-color:#1976d2}
label{font-size:12px;color:#666;white-space:nowrap}
.log{height:130px;overflow-y:auto;font-size:12px;color:#555;line-height:1.7}
.log div{border-bottom:1px solid #f0f0f0;padding:1px 0}
.log .late{color:#c62828;font-weight:500}
.log .delivered{color:#2e7d32}
.driver-row{display:flex;align-items:center;gap:8px;padding:4px 0;border-bottom:1px solid #f5f5f5;font-size:12px}
.driver-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0}
.order-row{display:flex;align-items:center;justify-content:space-between;padding:3px 0;font-size:12px;border-bottom:1px solid #f5f5f5}
.time-btns{display:flex;gap:6px;flex-wrap:wrap}
.time-btns button{padding:4px 10px;font-size:11px}
.rush-info{background:#fff8e1;border:1px solid #ffe082;border-radius:8px;padding:8px 12px;font-size:12px;color:#f57f17;margin-bottom:10px}
</style>
</head>
<body>

<header>
  <div>
    <h1>Dynamic Delivery Optimization — Live Demo</h1>
    <p>CS 4100 · Q-learning dispatch + simulation-based assignment heuristic + A* routing</p>
  </div>
  <span id="rush-badge" class="badge badge-green">Off-peak</span>
  <span id="time-badge" class="badge badge-blue">t = 0</span>
</header>

<main>
  <div>
    <div class="stats">
      <div class="stat"><div class="stat-val" id="s-completed">0</div><div class="stat-lbl">Completed</div></div>
      <div class="stat"><div class="stat-val" id="s-ontime">—</div><div class="stat-lbl">On-time %</div></div>
      <div class="stat"><div class="stat-val" id="s-wait">—</div><div class="stat-lbl">Avg wait (min)</div></div>
      <div class="stat"><div class="stat-val" id="s-pending">0</div><div class="stat-lbl">Pending</div></div>
    </div>

    <div class="grid-card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
        <span style="font-size:13px;font-weight:600;color:#555">5×5 City Grid</span>
        <div style="display:flex;gap:10px;font-size:12px;color:#888">
          <span>● Driver &nbsp; ■ Order &nbsp; <span style="color:#ef9f27">■</span> Urgent &nbsp; <span style="color:#ef5350">──</span> Rush edge</span>
        </div>
      </div>
      <canvas id="grid" width="520" height="520"></canvas>
      <div class="controls">
        <button class="primary" id="btn-play" onclick="togglePlay()">▶ Play</button>
        <button onclick="doReset()">↺ Reset</button>
        <label>Speed</label>
        <input type="range" id="speed" min="1" max="10" value="5" step="1">
        <label>Order rate</label>
        <input type="range" id="rate" min="1" max="9" value="3" step="1">
      </div>
    </div>
  </div>

  <div>
    <div class="card">
      <h3>Jump to hour</h3>
      <div id="rush-info" class="rush-info" style="display:none">⚠ Rush hour active — center edges 2× slower</div>
      <div class="time-btns">
        <button onclick="jumpTo(0)">12am</button>
        <button onclick="jumpTo(8)" style="background:#fff8e1">8am rush</button>
        <button onclick="jumpTo(12)">12pm</button>
        <button onclick="jumpTo(17)" style="background:#fff8e1">5pm rush</button>
        <button onclick="jumpTo(20)">8pm</button>
      </div>
    </div>

    <div class="card">
      <h3>Drivers</h3>
      <div id="driver-list"></div>
    </div>

    <div class="card">
      <h3>Active orders</h3>
      <div id="order-list" style="max-height:140px;overflow-y:auto"></div>
    </div>

    <div class="card">
      <h3>Traffic learning</h3>
      <div id="traffic-info" style="font-size:12px;color:#555;line-height:1.8"></div>
    </div>

    <div class="card">
      <h3>Event log</h3>
      <div class="log" id="log"></div>
    </div>
  </div>
</main>

<script>
const COLS=['#1976d2','#2e7d32','#bf360c'];
const GRID_SIZE=5;
let playing=false,timer=null,state=null;

async function fetchState(){
  const r=await fetch('/api/state');
  state=await r.json();
  render();
}

async function doStep(n=1){
  const r=await fetch('/api/step',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({n})});
  state=await r.json();
  render();
  if(state.events) state.events.forEach(e=>addLog(e));
}

async function doReset(){
  stopPlay();
  const r=await fetch('/api/reset',{method:'POST'});
  state=await r.json();
  render();
  addLog('Simulation reset.');
}

async function jumpTo(hour){
  stopPlay();
  const r=await fetch('/api/set_time',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({hour})});
  state=await r.json();
  render();
  addLog(`Jumped to ${hour}:00`);
}

function togglePlay(){
  if(playing){stopPlay();}
  else{startPlay();}
}

function startPlay(){
  playing=true;
  document.getElementById('btn-play').textContent='⏸ Pause';
  document.getElementById('btn-play').className='primary danger';
  const spd=parseInt(document.getElementById('speed').value);
  const delay=Math.round(1000/spd);
  timer=setInterval(()=>doStep(1),delay);
}

function stopPlay(){
  playing=false;
  document.getElementById('btn-play').textContent='▶ Play';
  document.getElementById('btn-play').className='primary';
  clearInterval(timer);
}

document.getElementById('speed').oninput=function(){
  if(playing){stopPlay();startPlay();}
};

function render(){
  if(!state)return;
  document.getElementById('s-completed').textContent=state.completed;
  document.getElementById('s-ontime').textContent=state.on_time_pct>0?state.on_time_pct+'%':'—';
  document.getElementById('s-wait').textContent=state.avg_wait>0?state.avg_wait:'—';
  document.getElementById('s-pending').textContent=state.pending;
  document.getElementById('time-badge').textContent=`${state.hour_str}  (t=${state.time})`;

  const isRush=state.rush_hour;
  document.getElementById('rush-badge').textContent=isRush?'Rush hour!':'Off-peak';
  document.getElementById('rush-badge').className='badge '+(isRush?'badge-red':'badge-green');
  document.getElementById('rush-info').style.display=isRush?'block':'none';

  renderDriverList();
  renderOrderList();
  renderTraffic();
  drawGrid();
}

function renderDriverList(){
  const el=document.getElementById('driver-list');
  el.innerHTML='';
  if(!state)return;
  for(const d of state.drivers){
    const div=document.createElement('div');
    div.className='driver-row';
    const status=d.status==='delivering'?`→ node ${d.dest}`:'idle';
    div.innerHTML=`<div class="driver-dot" style="background:${COLS[d.id]}"></div>
      <strong>D${d.id}</strong>
      <span style="color:#888">node ${d.node}</span>
      <span style="color:#555">${status}</span>
      <span style="margin-left:auto;background:#f5f5f5;padding:1px 7px;border-radius:10px">${d.queue_len} orders</span>`;
    el.appendChild(div);
  }
}

function renderOrderList(){
  const el=document.getElementById('order-list');
  el.innerHTML='';
  if(!state)return;
  const orders=state.orders.slice(0,8);
  if(orders.length===0){el.innerHTML='<div style="color:#aaa;font-size:12px;padding:4px 0">No active orders</div>';return;}
  for(const o of orders){
    const div=document.createElement('div');div.className='order-row';
    const col=o.urgent?'#c62828':o.time_left<20?'#e65100':'#1a1a1a';
    div.innerHTML=`<span>Order ${o.id} → node ${o.dest}</span>
      <span style="color:${col};font-weight:500">${o.time_left}min left</span>
      <span style="color:#aaa">D${o.assigned_to??'?'}</span>`;
    el.appendChild(div);
  }
}

function renderTraffic(){
  const el=document.getElementById('traffic-info');
  if(!state)return;
  const congested=state.edges.filter(e=>e.congested);
  const learned=state.edges.filter(e=>e.learned_time!==null);
  el.innerHTML=`
    <div>Congested edges: <strong>${congested.length}</strong> of ${state.edges.length}</div>
    <div>Edges with learned data: <strong>${learned.length}</strong></div>
    <div style="margin-top:6px;color:#888">Rush hours: 8–10am, 5–7pm (2× slower)</div>
    <div style="color:#888">Center nodes: 6,7,11,12,13,17,18</div>
  `;
}

function drawGrid(){
  if(!state)return;
  const cv=document.getElementById('grid');
  const ctx=cv.getContext('2d');
  const W=cv.width,H=cv.height,C=W/GRID_SIZE;
  ctx.clearRect(0,0,W,H);

  function pos(n){return{x:(n%GRID_SIZE)*C+C/2,y:Math.floor(n/GRID_SIZE)*C+C/2};}

  for(const e of state.edges){
    const p=pos(e.from),q=pos(e.to);
    ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);
    if(e.congested){ctx.strokeStyle='#ef5350';ctx.lineWidth=3;}
    else{ctx.strokeStyle='#e0e0e0';ctx.lineWidth=1;}
    ctx.stroke();
  }

  for(const o of state.orders){
    const p=pos(o.dest);
    ctx.fillStyle=o.urgent?'#ef5350':'#ef9f27';
    ctx.fillRect(p.x-7,p.y-7,14,14);
  }

  for(let n=0;n<25;n++){
    const p=pos(n);
    ctx.beginPath();ctx.arc(p.x,p.y,C*0.22,0,Math.PI*2);
    ctx.fillStyle='#fff';ctx.fill();
    ctx.strokeStyle='#ccc';ctx.lineWidth=1;ctx.stroke();
    ctx.fillStyle='#999';ctx.font=`${Math.round(C*0.18)}px sans-serif`;
    ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(n,p.x,p.y);
  }

  for(const d of state.drivers){
    const p=pos(d.node);
    if(d.dest!==null){
      const q=pos(d.dest);
      ctx.beginPath();ctx.setLineDash([5,5]);ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);
      ctx.strokeStyle=COLS[d.id];ctx.lineWidth=1.5;ctx.stroke();ctx.setLineDash([]);
    }
    ctx.beginPath();ctx.arc(p.x,p.y,C*0.3,0,Math.PI*2);
    ctx.fillStyle=COLS[d.id];ctx.fill();
    ctx.fillStyle='#fff';ctx.font=`bold ${Math.round(C*0.22)}px sans-serif`;
    ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText('D'+d.id,p.x,p.y);
  }
}

function addLog(msg){
  const el=document.getElementById('log');
  const div=document.createElement('div');
  const late=msg.includes('LATE');
  const delivered=msg.includes('delivered');
  if(late)div.className='late';
  else if(delivered)div.className='delivered';
  div.textContent=msg;
  el.prepend(div);
  if(el.children.length>60)el.removeChild(el.lastChild);
}

fetchState();
</script>
</body>
</html>
"""

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  Delivery Optimization Demo Server")
    print("="*50)
    print("  Open in browser: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=False, port=5000)