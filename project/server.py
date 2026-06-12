"""
server_ql.py — 1 Driver Q-learning demo.

Put in SAME folder as: delivery_env.py  rl_agent.py  assignment.py

Run:
    pip install flask
    python3 server_ql.py

Open: http://localhost:8080
"""

import random, os
from collections import deque
from flask import Flask, jsonify, request, render_template_string
from delivery_env import DeliveryEnvironment, Order

# ── Q-learning agent (1 driver version) ──────────────────────────────
import numpy as np, pickle

class SingleDriverQLAgent:
    """
    Q-learning agent for exactly 1 driver.
    State: (current_node, next_dest, queue_len, hour)
    Action: index into neighbors list (0-3) or 4=stay
    """
    def __init__(self, alpha=0.1, gamma=0.95,
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.9995):
        self.Q            = {}
        self.alpha        = alpha
        self.gamma        = gamma
        self.epsilon      = epsilon_start
        self.epsilon_end  = epsilon_end
        self.epsilon_decay= epsilon_decay

    def discretize_state(self, env):
        d    = env.drivers[0]
        hour = (env.current_time // 60) % 24
        if d.order_queue:
            dest  = env.orders[d.order_queue[0]].destination
            qlen  = min(len(d.order_queue), 5)
        else:
            dest, qlen = -1, 0
        return (d.current_node, dest, qlen, hour)

    def select_action(self, state, env, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        neighbors = env.graph.get_neighbors(env.drivers[0].current_node)
        n_actions = len(neighbors) + 1   # neighbors + stay
        if random.random() < epsilon or state not in self.Q:
            return random.randint(0, n_actions - 1)
        q = self.Q[state]
        # only consider valid actions for this state
        valid_q = q[:n_actions]
        return int(np.argmax(valid_q))

    def update(self, state, action, reward, next_state, env, done):
        n_act = len(env.graph.get_neighbors(env.drivers[0].current_node)) + 1
        if state not in self.Q:
            self.Q[state] = np.zeros(5)   # max 4 neighbors + stay
        if next_state not in self.Q:
            self.Q[next_state] = np.zeros(5)
        old_q  = self.Q[state][action]
        target = reward if done else reward + self.gamma * np.max(self.Q[next_state][:n_act])
        self.Q[state][action] = old_q + self.alpha * (target - old_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'Q': self.Q, 'epsilon': self.epsilon}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.Q       = data['Q']
            self.epsilon = data.get('epsilon', self.epsilon_end)

# ── Setup ─────────────────────────────────────────────────────────────
agent      = SingleDriverQLAgent()
AGENT_MODE = "random"

if os.path.exists("Q_table_1driver.pkl"):
    agent.load("Q_table_1driver.pkl")
    agent.epsilon = 0.0
    AGENT_MODE    = "qlearning"
    print(f"[OK] Q-table loaded — {len(agent.Q)} states")
else:
    print("[INFO] No Q_table_1driver.pkl found — will train live or use random")

app         = Flask(__name__)
env         = None
stats       = {}
events      = deque(maxlen=60)
reward_hist = []
ep_rewards  = []   # rewards within current episode

def make_env():
    global env, stats, reward_hist, ep_rewards
    env = DeliveryEnvironment(
        num_drivers=1, grid_size=5,
        traffic_pattern="rush_hour",
        order_arrival_rate=0.25,
        use_progress_shaping=False,
    )
    env.reset()
    stats       = {"completed":0,"ontime":0,"total_wait":0,"missed":0,"ep":0}
    reward_hist = []
    ep_rewards  = []
    events.clear()
    events.appendleft("Simulation ready. 1 driver, Q-learning.")

make_env()

# ── Helpers ───────────────────────────────────────────────────────────
def get_edges():
    hour = (env.current_time // 60) % 24
    seen, out = set(), []
    for (n1, n2) in env.graph.edges.keys():
        key = (min(n1,n2), max(n1,n2))
        if key in seen: continue
        seen.add(key)
        m = env.graph.traffic_multipliers.get((n1,n2),{}).get(hour,1.0)
        if m == 1.0:
            m = env.graph.traffic_multipliers.get((n2,n1),{}).get(hour,1.0)
        base = env.graph.edges.get((n1,n2), 10)
        lrn  = env.learned_traffic.get((n1,n2)) or env.learned_traffic.get((n2,n1))
        lt   = None
        if lrn and hour in lrn and lrn[hour]:
            lt = round(sum(lrn[hour])/len(lrn[hour]), 1)
        out.append({"from":n1,"to":n2,"base":base,
                    "mult":m,"congested":m>1.0,"learned":lt})
    return out

def snapshot():
    hour  = (env.current_time // 60) % 24
    rush  = hour in {8,9,10,17,18,19}
    d     = env.drivers[0]
    dest  = None
    if d.order_queue and d.order_queue[0] in env.orders:
        dest = env.orders[d.order_queue[0]].destination

    # Q-table info
    state     = agent.discretize_state(env)
    q_arr     = agent.Q.get(state, None)
    q_known   = q_arr is not None
    neighbors = env.graph.get_neighbors(d.current_node)
    q_actions = []
    if q_arr is not None:
        for i, nb in enumerate(neighbors):
            q_actions.append({"node": nb, "q": round(float(q_arr[i]), 2)})
        # stay action
        stay_idx = len(neighbors)
        if stay_idx < len(q_arr):
            q_actions.append({"node": "stay", "q": round(float(q_arr[stay_idx]), 2)})

    orders = []
    for o in env.orders.values():
        if not o.delivered:
            tl = o.deadline - env.current_time
            if tl > 0:
              orders.append({"id":o.id,"dest":o.destination,
                   "time_left":tl,"urgent":tl<10,"driver":o.assigned_driver})

    otp = round(stats["ontime"]/stats["completed"]*100,1) if stats["completed"]>0 else 0
    aw  = round(stats["total_wait"]/stats["completed"],1) if stats["completed"]>0 else 0

    return {
        "time":env.current_time, "hour":hour,
        "hour_str":f"{hour}:{env.current_time%60:02d}",
        "rush":rush,
        "driver":{"node":d.current_node,"dest":dest,
                  "queue":list(d.order_queue),"queue_len":len(d.order_queue),
                  "status":d.status},
        "orders":orders, "edges":get_edges(),
        "completed":stats["completed"], "missed":stats["missed"],
        "on_time_pct":otp, "avg_wait":aw,
        "agent_mode":AGENT_MODE,
        "q_known":q_known, "q_states":len(agent.Q),
        "q_actions":q_actions, "epsilon":round(agent.epsilon,4),
        "reward_history":list(reward_hist[-80:]),
        "events":list(events)[:20],
    }

def do_step(n=1, train=False):
    global ep_rewards
    for _ in range(n):
        d         = env.drivers[0]
        state     = agent.discretize_state(env)
        action    = agent.select_action(state, env)
        neighbors = env.graph.get_neighbors(d.current_node)

        # convert action index to neighbor node
        if action < len(neighbors):
            move = [action]
        else:
            move = [len(neighbors)]   # stay (out-of-range = stay in env)

        prev      = len(env.completed_orders)
        _, reward, done = env.step(move)
        next_state = agent.discretize_state(env)
        ep_rewards.append(reward)

        if train:
            agent.update(state, action, reward, next_state, env, done)

        for oid in env.completed_orders[prev:]:
            o    = env.orders[oid]
            wait = o.delivery_time - o.arrival_time
            late = o.delivery_time > o.deadline
            stats["completed"]  += 1
            stats["total_wait"] += wait
            if not late: stats["ontime"] += 1
            else:        stats["missed"] += 1
            tag = "LATE ✗" if late else "on-time ✓"
            events.appendleft(
                f"t={env.current_time}: delivered #{oid} "
                f"(wait {wait}min, {tag})")

        if done:
            stats["ep"] += 1
            ep_total = sum(ep_rewards)
            reward_hist.append(round(ep_total, 1))
            ep_rewards = []
            if train:
                agent.decay_epsilon()
                if stats["ep"] % 100 == 0:
                    agent.save("Q_table_1driver.pkl")
                    events.appendleft(f"Q-table saved ({len(agent.Q)} states)")
            events.appendleft(
                f"--- Episode {stats['ep']} done | "
                f"completed {len(env.completed_orders)} | "
                f"reward {ep_total:.0f} ---")
            env.reset()
            break

# ── Routes ────────────────────────────────────────────────────────────
@app.route("/")
def index(): return render_template_string(HTML)

@app.route("/api/state")
def api_state(): return jsonify(snapshot())

@app.route("/api/step", methods=["POST"])
def api_step():
    n     = int(request.json.get("n", 1))
    train = bool(request.json.get("train", False))
    do_step(n, train=train)
    return jsonify(snapshot())

@app.route("/api/reset", methods=["POST"])
def api_reset():
    make_env(); return jsonify(snapshot())

@app.route("/api/jump", methods=["POST"])
def api_jump():
    h = int(request.json.get("hour", 9))
    env.current_time = h * 60
    events.appendleft(f"Jumped to {h}:00")
    return jsonify(snapshot())

# ── HTML UI ───────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Q-Learning Delivery — 1 Driver</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f0f2f5;color:#1a1a2e;font-size:13px}
header{background:#1a1a2e;color:#fff;padding:12px 20px;display:flex;align-items:center;justify-content:space-between;gap:12px}
header h1{font-size:15px;font-weight:600}
header p{font-size:11px;color:#aab;margin-top:2px}
.hb{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.badge{padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600}
.b-green{background:#27ae60;color:#fff}.b-red{background:#e74c3c;color:#fff}
.b-blue{background:#2980b9;color:#fff}.b-amber{background:#f39c12;color:#111}
.b-purple{background:#8e44ad;color:#fff}.b-gray{background:#7f8c8d;color:#fff}
main{display:grid;grid-template-columns:1fr 270px;gap:10px;padding:10px;max-width:1080px;margin:0 auto}
.card{background:#fff;border-radius:10px;border:1px solid #e0e0e0;padding:12px;margin-bottom:10px}
.card h3{font-size:10px;font-weight:700;color:#999;text-transform:uppercase;letter-spacing:.6px;margin-bottom:8px}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:10px}
.stat{background:#fff;border:1px solid #eee;border-radius:8px;padding:8px 4px;text-align:center}
.sv{font-size:22px;font-weight:700;color:#1a1a2e}.sl{font-size:10px;color:#aaa;margin-top:1px}
canvas{display:block;width:100%}
.ctrl{display:flex;align-items:center;gap:8px;margin-top:8px;flex-wrap:wrap}
.ctrl label{font-size:11px;color:#888;white-space:nowrap}
input[type=range]{flex:1;min-width:50px;accent-color:#2980b9}
button{padding:6px 12px;border-radius:6px;border:1px solid #ccc;background:#fff;cursor:pointer;font-size:12px;font-weight:500}
button:hover{background:#f0f2f5}
.btn-play{background:#2980b9;border-color:#2471a3;color:#fff}
.btn-stop{background:#e74c3c;border-color:#c0392b;color:#fff}
.btn-reset{background:#27ae60;border-color:#229954;color:#fff}
.btn-train{background:#8e44ad;border-color:#7d3c98;color:#fff}
.jb{display:flex;gap:6px;flex-wrap:wrap}
.jb button{padding:4px 8px;font-size:11px}
.rush-btn{background:#fff8e1!important;border-color:#f39c12!important;color:#e67e22!important;font-weight:600!important}
.o-row{display:flex;justify-content:space-between;align-items:center;padding:3px 0;border-bottom:1px solid #f5f5f5;font-size:11px}
.o-urgent{color:#e74c3c;font-weight:600}.o-ok{color:#27ae60}
.q-row{display:flex;align-items:center;gap:8px;padding:4px 0;border-bottom:1px solid #f5f5f5;font-size:12px}
.q-bar-bg{flex:1;background:#f0f2f5;border-radius:4px;height:10px;overflow:hidden}
.q-bar{height:10px;border-radius:4px;transition:width .3s}
.log{height:110px;overflow-y:auto;font-size:11px;color:#555;line-height:1.6}
.log div{border-bottom:1px solid #f5f5f5;padding:1px 0}
.ev-late{color:#e74c3c}.ev-ok{color:#27ae60}.ev-ep{color:#8e44ad;font-weight:500}
.d-card{background:#e8f4fd;border:1.5px solid #2980b9;border-radius:10px;padding:10px 12px;margin-bottom:10px}
.d-card h3{color:#1a5276;margin-bottom:6px;font-size:11px;font-weight:700;text-transform:uppercase}
.d-row{display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #aed6f1;font-size:12px}
.chart-wrap{height:75px}
.train-row{display:flex;align-items:center;gap:8px;margin-top:6px;flex-wrap:wrap}
.train-info{font-size:11px;color:#888}
</style>
</head>
<body>
<header>
  <div>
    <h1>Q-Learning Delivery Optimization — 1 Driver</h1>
    <p>CS 4100 · Tabular Q-learning navigation + simulation-based assignment heuristic (Person 2)</p>
  </div>
  <div class="hb">
    <span id="agent-badge" class="badge b-blue">Agent: Q-learning</span>
    <span id="rush-badge"  class="badge b-green">Off-peak</span>
    <span id="time-badge"  class="badge b-amber">t=0</span>
    <span id="state-badge" class="badge b-purple">States: 0</span>
    <span id="eps-badge"   class="badge b-gray">ε=1.0</span>
  </div>
</header>

<main>
<div>
  <div class="stats">
    <div class="stat"><div class="sv" id="s-comp">0</div><div class="sl">Completed</div></div>
    <div class="stat"><div class="sv" id="s-ontime">—</div><div class="sl">On-time %</div></div>
    <div class="stat"><div class="sv" id="s-wait">—</div><div class="sl">Avg wait (min)</div></div>
    <div class="stat"><div class="sv" id="s-miss">0</div><div class="sl">Missed</div></div>
  </div>

  <div class="card">
    <h3>5×5 City Grid
      <span style="font-weight:400;color:#bbb;text-transform:none">
        · Red = rush congestion · Orange square = order · Blue circle = driver
      </span>
    </h3>
    <canvas id="grid" width="520" height="520"></canvas>
    <div class="ctrl">
      <button class="btn-play" id="btn" onclick="togglePlay(false)">▶ Play</button>
      <button class="btn-train" id="btn-train" onclick="togglePlay(true)">▶ Train + Play</button>
      <button class="btn-reset" onclick="doReset()">↺ Reset</button>
      <label>Speed</label>
      <input type="range" id="spd" min="1" max="15" value="5">
      <span id="spd-lbl" style="font-size:11px;color:#888;min-width:24px">5×</span>
    </div>
    <div class="train-info" style="margin-top:6px">
      <b>Play</b> = use Q-table only &nbsp;|&nbsp;
      <b>Train + Play</b> = learn while running (Q-table grows, saves every 100 episodes)
    </div>
  </div>

  <div class="card">
    <h3>Q-values for current state
      <span id="q-lbl" style="font-weight:400;text-transform:none;color:#bbb"></span>
    </h3>
    <div id="q-viz"></div>
  </div>

  <div class="card">
    <h3>Episode reward history <span style="font-weight:400;color:#bbb;text-transform:none">(higher = better)</span></h3>
    <div class="chart-wrap"><canvas id="rchart" height="75"></canvas></div>
  </div>
</div>

<div>
  <div class="card">
    <h3>Jump to hour</h3>
    <div class="jb">
      <button onclick="jump(0)">12am</button>
      <button class="rush-btn" onclick="jump(8)">8am rush ⚡</button>
      <button onclick="jump(12)">12pm</button>
      <button class="rush-btn" onclick="jump(17)">5pm rush ⚡</button>
      <button onclick="jump(22)">10pm</button>
    </div>
  </div>

  <div class="d-card">
    <h3>🚗 Driver D0</h3>
    <div class="d-row"><span>Position</span><b id="d-node">—</b></div>
    <div class="d-row"><span>Destination</span><b id="d-dest">—</b></div>
    <div class="d-row"><span>Queue</span><b id="d-queue">0 orders</b></div>
    <div class="d-row"><span>Status</span><b id="d-status">idle</b></div>
    <div class="d-row" style="border:none"><span>State in Q-table?</span><span id="d-known">—</span></div>
  </div>

  <div class="card">
    <h3>Active orders</h3>
    <div id="order-list" style="max-height:130px;overflow-y:auto"></div>
  </div>

  <div class="card">
    <h3>Traffic learning</h3>
    <div id="traffic" style="font-size:12px;line-height:2;color:#555"></div>
  </div>

  <div class="card">
    <h3>Event log</h3>
    <div class="log" id="log"></div>
  </div>
</div>
</main>

<script>
const GS=5;
let playing=false,training=false,timer=null,state=null;

async function load(){const r=await fetch('/api/state');state=await r.json();render();}
async function step(n,train){
  const r=await fetch('/api/step',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({n:n||1,train:train||false})});
  state=await r.json();render();
}
async function doReset(){stop();const r=await fetch('/api/reset',{method:'POST'});state=await r.json();render();}
async function jump(h){stop();const r=await fetch('/api/jump',{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify({hour:h})});
  state=await r.json();render();}

function togglePlay(train){
  if(playing&&training===train){stop();return;}
  if(playing){stop();}
  start(train);
}
function start(train){
  playing=true;training=train;
  const b=document.getElementById('btn');
  const bt=document.getElementById('btn-train');
  if(train){bt.textContent='⏸ Stop Training';bt.className='btn-stop';}
  else{b.textContent='⏸ Pause';b.className='btn-stop';}
  const spd=parseInt(document.getElementById('spd').value);
  timer=setInterval(()=>step(1,train),Math.max(40,Math.round(700/spd)));
}
function stop(){
  playing=false;training=false;
  document.getElementById('btn').textContent='▶ Play';
  document.getElementById('btn').className='btn-play';
  document.getElementById('btn-train').textContent='▶ Train + Play';
  document.getElementById('btn-train').className='btn-train';
  clearInterval(timer);
}
document.getElementById('spd').oninput=function(){
  document.getElementById('spd-lbl').textContent=this.value+'×';
  if(playing){const t=training;stop();start(t);}
};

function render(){
  if(!state)return;
  document.getElementById('s-comp').textContent=state.completed;
  document.getElementById('s-ontime').textContent=state.on_time_pct>0?state.on_time_pct+'%':'—';
  document.getElementById('s-wait').textContent=state.avg_wait>0?state.avg_wait:'—';
  document.getElementById('s-miss').textContent=state.missed;
  document.getElementById('time-badge').textContent=state.hour_str+' (t='+state.time+')';
  document.getElementById('rush-badge').textContent=state.rush?'Rush hour ⚡':'Off-peak';
  document.getElementById('rush-badge').className='badge '+(state.rush?'b-red':'b-green');
  document.getElementById('state-badge').textContent='Q-states: '+state.q_states;
  document.getElementById('eps-badge').textContent='ε='+state.epsilon;
  const am=state.agent_mode;
  document.getElementById('agent-badge').textContent=am==='qlearning'?'Agent: Q-learning':'Agent: Random';
  document.getElementById('agent-badge').className='badge '+(am==='qlearning'?'b-blue':'b-red');
  const d=state.driver;
  document.getElementById('d-node').textContent='node '+d.node;
  document.getElementById('d-dest').textContent=d.dest!==null?'node '+d.dest:'none';
  document.getElementById('d-queue').textContent=d.queue_len+' order'+(d.queue_len!==1?'s':'');
  document.getElementById('d-status').textContent=d.status;
  document.getElementById('d-known').innerHTML=state.q_known?
    '<span style="color:#27ae60;font-weight:600">✓ Yes — using Q-table</span>':
    '<span style="color:#e74c3c">✗ No — random fallback</span>';
  renderQViz();renderOrders();renderTraffic();renderLog();drawGrid();drawChart();
}

function renderQViz(){
  const el=document.getElementById('q-viz');
  const lbl=document.getElementById('q-lbl');
  const qa=state.q_actions||[];
  if(!qa.length){
    lbl.textContent='— state not seen yet';
    el.innerHTML='<div style="color:#aaa;font-size:12px;padding:6px 0">No Q-values yet for this state. Agent explores randomly.</div>';
    return;
  }
  lbl.textContent=`— ${qa.length} actions available`;
  const nums=qa.map(a=>a.q);
  const mn=Math.min(...nums),mx=Math.max(...nums),rng=mx-mn||1;
  const best=qa.reduce((a,b)=>a.q>b.q?a:b);
  el.innerHTML='';
  for(const a of qa){
    const pct=Math.max(4,Math.round((a.q-mn)/rng*100));
    const isBest=a.node===best.node;
    const col=isBest?'#2980b9':a.q>=0?'#27ae60':'#e74c3c';
    const label=typeof a.node==='number'?`→ node ${a.node}`:a.node;
    const div=document.createElement('div');div.className='q-row';
    div.innerHTML=`
      <span style="min-width:90px;font-weight:${isBest?700:400}">
        ${label}${isBest?' ✓ best':''}
      </span>
      <div class="q-bar-bg">
        <div class="q-bar" style="width:${pct}%;background:${col}"></div>
      </div>
      <span style="min-width:55px;text-align:right;font-weight:${isBest?700:400};color:${col}">
        ${a.q.toFixed(2)}
      </span>`;
    el.appendChild(div);
  }
}

function renderOrders(){
  const el=document.getElementById('order-list');el.innerHTML='';
  const os=state.orders||[];
  if(!os.length){el.innerHTML='<div style="color:#aaa;padding:4px 0">No active orders</div>';return;}
  for(const o of os.slice(0,8)){
    const d=document.createElement('div');d.className='o-row';
    d.innerHTML=`<span>#${o.id} → node ${o.dest}</span>
      <span class="${o.urgent?'o-urgent':'o-ok'}">${o.time_left}min</span>`;
    el.appendChild(d);
  }
}

function renderTraffic(){
  const cong=(state.edges||[]).filter(e=>e.congested).length;
  const lrn=(state.edges||[]).filter(e=>e.learned!==null).length;
  document.getElementById('traffic').innerHTML=
    `Congested edges: <b>${cong}</b> / ${(state.edges||[]).length}<br>
     Learned traffic data: <b>${lrn}</b> edges<br>
     Rush hours: 8–10am, 5–7pm (2× slower)<br>
     Center nodes: 6, 7, 11–13, 17, 18`;
}

function renderLog(){
  const el=document.getElementById('log');el.innerHTML='';
  for(const ev of state.events||[]){
    const d=document.createElement('div');
    d.className=ev.includes('LATE')?'ev-late':ev.includes('on-time')?'ev-ok':ev.includes('---')?'ev-ep':'';
    d.textContent=ev;el.appendChild(d);
  }
}

function drawGrid(){
  if(!state)return;
  const cv=document.getElementById('grid'),ctx=cv.getContext('2d');
  const W=cv.width,H=cv.height,C=W/GS;
  ctx.clearRect(0,0,W,H);
  function pos(n){return{x:(n%GS)*C+C/2,y:Math.floor(n/GS)*C+C/2};}

  for(const e of state.edges||[]){
    const p=pos(e.from),q=pos(e.to);
    ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);
    ctx.strokeStyle=e.congested?'rgba(231,76,60,0.6)':'#e8e8e8';
    ctx.lineWidth=e.congested?4:1.2;ctx.stroke();
    if(e.congested&&e.learned){
      const mx=(p.x+q.x)/2,my=(p.y+q.y)/2;
      ctx.fillStyle='rgba(231,76,60,0.9)';
      ctx.font=`bold ${Math.round(C*0.13)}px sans-serif`;
      ctx.textAlign='center';ctx.textBaseline='middle';
      ctx.fillText(e.learned+'m',mx,my);
    }
  }

  // highlight Q-value neighbors
  const qa=state.q_actions||[];
  if(qa.length>0){
    const nums=qa.filter(a=>typeof a.node==='number').map(a=>a.q);
    const mn=Math.min(...nums),mx=Math.max(...nums),rng=mx-mn||1;
    const best=qa.filter(a=>typeof a.node==='number').reduce((a,b)=>a.q>b.q?a:b,{q:-Infinity,node:-1});
    for(const a of qa){
      if(typeof a.node!=='number')continue;
      const p=pos(a.node);
      const alpha=0.12+0.45*(a.q-mn)/rng;
      ctx.beginPath();ctx.arc(p.x,p.y,C*0.35,0,Math.PI*2);
      ctx.fillStyle=a.node===best.node?`rgba(41,128,185,${alpha})`:`rgba(189,189,189,${alpha})`;
      ctx.fill();
    }
  }

  for(const o of state.orders||[]){
    const p=pos(o.dest),sz=C*0.22;
    ctx.fillStyle=o.urgent?'#e74c3c':'#f39c12';
    ctx.fillRect(p.x-sz,p.y-sz,sz*2,sz*2);
    ctx.fillStyle='#fff';
    ctx.font=`bold ${Math.round(sz)}px sans-serif`;
    ctx.textAlign='center';ctx.textBaseline='middle';
    ctx.fillText(o.id,p.x,p.y);
  }

  for(let n=0;n<25;n++){
    const p=pos(n);
    ctx.beginPath();ctx.arc(p.x,p.y,C*0.17,0,Math.PI*2);
    ctx.fillStyle='#fff';ctx.fill();
    ctx.strokeStyle='#ccc';ctx.lineWidth=0.8;ctx.stroke();
    ctx.fillStyle='#aaa';
    ctx.font=`${Math.round(C*0.14)}px sans-serif`;
    ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(n,p.x,p.y);
  }

  const d=state.driver;
  if(d.dest!==null){
    const p=pos(d.node),q=pos(d.dest);
    ctx.beginPath();ctx.setLineDash([5,5]);
    ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);
    ctx.strokeStyle='#2980b9';ctx.lineWidth=2;ctx.stroke();ctx.setLineDash([]);
  }

  const p=pos(d.node);
  ctx.beginPath();ctx.arc(p.x,p.y,C*0.3,0,Math.PI*2);
  ctx.fillStyle='#2980b9';ctx.fill();
  ctx.strokeStyle='#fff';ctx.lineWidth=3;ctx.stroke();
  ctx.fillStyle='#fff';
  ctx.font=`bold ${Math.round(C*0.2)}px sans-serif`;
  ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText('D0',p.x,p.y);
  if(d.queue_len>0){
    const bx=p.x+C*0.22,by=p.y-C*0.22;
    ctx.beginPath();ctx.arc(bx,by,C*0.14,0,Math.PI*2);
    ctx.fillStyle='#f39c12';ctx.fill();
    ctx.fillStyle='#fff';
    ctx.font=`bold ${Math.round(C*0.13)}px sans-serif`;
    ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(d.queue_len,bx,by);
  }
}

function drawChart(){
  const cv=document.getElementById('rchart'),ctx=cv.getContext('2d');
  cv.width=cv.offsetWidth||520;cv.height=75;
  const W=cv.width,H=cv.height;
  ctx.clearRect(0,0,W,H);
  const hist=state.reward_history||[];
  if(hist.length<2)return;
  const mn=Math.min(...hist),mx=Math.max(...hist),rng=mx-mn||1;
  const pw=W/hist.length;
  ctx.beginPath();
  hist.forEach((v,i)=>{
    const x=i*pw,y=H-(v-mn)/rng*(H-10)-5;
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  });
  ctx.strokeStyle='#2980b9';ctx.lineWidth=1.5;ctx.stroke();
  ctx.fillStyle='rgba(41,128,185,0.1)';
  ctx.lineTo(hist.length*pw,H);ctx.lineTo(0,H);ctx.closePath();ctx.fill();
  const last=hist[hist.length-1];
  ctx.fillStyle='#888';ctx.font='10px sans-serif';ctx.textAlign='left';
  ctx.fillText(`${hist.length} episodes · last: ${last.toFixed(1)}`,4,12);
}

load();
</script>
</body>
</html>
"""

if __name__=="__main__":
    print("\n"+"="*50)
    print("  Q-Learning 1-Driver Delivery Demo")
    print(f"  Agent: {AGENT_MODE.upper()}")
    print("  Open: http://localhost:8080")
    print("="*50+"\n")
    app.run(debug=False, port=8080, host='0.0.0.0')
