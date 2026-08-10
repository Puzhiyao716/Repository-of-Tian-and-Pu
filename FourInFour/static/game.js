// ================================================================
// 四维四子棋 - 前端客户端脚本（槽位制）
// ================================================================

const SYM = { 1: "○", 2: "✕", 3: "△" };
const COL = { 1: "player-1", 2: "player-2", 3: "player-3" };

let mySlot = null, gameOver = false, started = false;
let currentTurn = 1, boardState = [], lastMoves = {};
let playerCount = 0, playerTypes = {}, playerStrategies = {};
let playerTimeLimits = {}, playerMaxIters = {};
let winningLine = [];

// DOM
const bg = document.getElementById("boardGrid");
const ti = document.getElementById("turnIndicator");
const le = document.getElementById("logEntries");
const bs = document.getElementById("btnStartGame");
const slotRows = {};
for (let i = 1; i <= 3; i++) slotRows[i] = document.getElementById("slot" + i);

// ---- 工具 ----
function idx(r, c) { const w=Math.floor(r/4),x=r%4,y=Math.floor(c/4),z=c%4; return w*64+x*16+y*4+z; }
function addLog(msg) {
    const d = document.createElement("div"); d.className = "log-entry";
    d.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    le.prepend(d); while (le.children.length > 50) le.lastChild.remove();
}

// ---- 棋盘（修正边框：每格仅一条红线，不重叠） ----
function buildBoard() {
    bg.innerHTML = "";
    for (let r = 0; r < 16; r++) for (let c = 0; c < 16; c++) {
        const cell = document.createElement("div"); cell.className = "cell";
        cell.dataset.row = r; cell.dataset.col = c;
        if (c === 3||c === 7||c === 11||c === 15) cell.classList.add("block-right");
        if (r === 3||r === 7||r === 11||r === 15) cell.classList.add("block-bottom");
        if (c === 0) cell.classList.add("block-left");
        if (r === 0) cell.classList.add("block-top");
        cell.addEventListener("click", () => onCellClick(r, c));
        bg.appendChild(cell);
    }
}
function renderBoard() {
    const cells = bg.children;
    for (let r = 0; r < 16; r++) for (let c = 0; c < 16; c++) {
        const i = idx(r, c), cell = cells[r*16+c];
        cell.className = "cell";
        if (c === 3||c === 7||c === 11||c === 15) cell.classList.add("block-right");
        if (r === 3||r === 7||r === 11||r === 15) cell.classList.add("block-bottom");
        if (c === 0) cell.classList.add("block-left");
        if (r === 0) cell.classList.add("block-top");
        const o = boardState[i] || 0;
        if (o) { cell.textContent = SYM[o]; cell.classList.add(COL[o]); }
        else cell.textContent = "";
    }
    for (const [pid, pos] of Object.entries(lastMoves)) {
        const [r, c] = pos; bg.children[r*16+c].classList.add("last-move");
    }
    // 高亮获胜四子
    for (const [r, c] of winningLine) {
        bg.children[r*16+c].classList.add("win-highlight");
    }
    // 回合提示：棋盘周围发光
    const wrapper = document.querySelector(".board-wrapper");
    if (!gameOver && started && mySlot !== null && mySlot === currentTurn) {
        wrapper.classList.add("my-turn");
    } else {
        wrapper.classList.remove("my-turn");
    }
    updateTurnUI();
}

function updateTurnUI() {
    if (!started) { ti.innerHTML = "等待玩家就绪…"; return; }
    if (gameOver) return;
    ti.innerHTML = `当前回合：<span class="symbol">${SYM[currentTurn]||"—"}</span> 玩家 ${currentTurn}`;
}
function handleGameOver(winner) {
    if (winner===0) { ti.innerHTML="🤝 <b>平局！</b>"; addLog("游戏结束：平局！"); }
    else { ti.innerHTML=`🏆 <b>玩家 ${winner} (${SYM[winner]}) 获胜！</b>`; addLog(`🎉 玩家 ${winner} 获胜！`); }
}

// ---- 槽位 UI ----
function updateSlots() {
    for (let s = 1; s <= 3; s++) {
        const row = slotRows[s];
        const isComp = playerTypes[s] === "computer";
        const strat = playerStrategies[s] || "";
        const occ = s in playerTypes;
        let html = "";
        if (occ) {
            if (isComp) html = `🤖 ${strat} <button class="btn-sm btn-del" data-slot="${s}" data-act="remove_robot">删除</button>`;
            else if (mySlot === s) html = `😎 你 <button class="btn-sm btn-stand" data-act="stand">站起</button>`;
            else html = "🧑 人类（已就位）";
        } else {
            html = `<button class="btn-sm btn-sit" data-slot="${s}" data-act="sit">坐下</button>
                    <button class="btn-sm btn-bot" data-slot="${s}" data-act="add_robot">+🤖</button>`;
        }
        row.innerHTML = html;
    }
    document.querySelectorAll(".btn-sm").forEach(btn => {
        btn.addEventListener("click", () => {
            const act = btn.dataset.act, slot = parseInt(btn.dataset.slot);
            if (act === "sit") send({ type: "sit", slot });
            else if (act === "stand") send({ type: "stand" });
            else if (act === "add_robot") send({
                type: "add_robot", slot,
                strategy: document.getElementById("selStrategy").value,
                time_limit: parseFloat(document.getElementById("inpTimeLimit").value) || 5.0,
                max_iters: parseInt(document.getElementById("inpMaxIters").value) || 30000,
            });
            else if (act === "remove_robot") send({ type: "remove_robot", slot });
        });
    });
}

function updateUI() {
    bs.style.display = (playerCount >= 3 && !started) ? "block" : "none";
    updateSlots(); updateTurnUI();
}

// ---- WebSocket ----
let ws = null;
function connectWS() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${proto}//${location.host}/ws`);
    ws.onopen = () => addLog("已连接（观战者，点击座位坐下）");
    ws.onmessage = (e) => { try { handleMsg(JSON.parse(e.data)); } catch(err) { console.error(err); } };
    ws.onclose = () => { addLog("断开，3秒后重连..."); setTimeout(connectWS, 3000); };
    ws.onerror = () => {};
}
function send(data) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(data)); }

function handleMsg(msg) {
    switch (msg.type) {
    case "welcome":
        mySlot = null;
        break;
    case "sit_confirm":
        mySlot = msg.slot;
        addLog(`你已坐在座位 ${mySlot} (${SYM[mySlot]})`);
        break;
    case "stand_confirm":
        mySlot = null;
        addLog("你已站起，现在是观战者");
        break;
    case "state":
        boardState = msg.board || [];
        currentTurn = msg.current_turn;
        gameOver = msg.game_over;
        lastMoves = msg.last_moves || {};
        playerCount = msg.player_count || 0;
        started = msg.started || false;
        playerTypes = msg.player_types || {};
        playerStrategies = msg.player_strategies || {};
        playerTimeLimits = msg.player_time_limits || {};
        playerMaxIters = msg.player_max_iters || {};
        winningLine = msg.winning_line || [];
        renderBoard(); updateUI();
        if (msg.game_over) handleGameOver(msg.winner);
        break;
    case "move_result":
        if (msg.success) {
            const m = msg.move;
            // 记录落子日志，使用1~4表示而不是0~3以符合玩家习惯
            let logMsg = `玩家${m.player}(${SYM[m.player]})落子[${m.row + 1},${m.col + 1}] | 4D:${m.w + 1},${m.x + 1},${m.y + 1},${m.z + 1}`;
            // 记录思考时间和迭代次数
            if (msg.thinking) {
                logMsg += `  |  思考 ${msg.thinking.time}s · ${msg.thinking.iters} 次迭代`;
            }
            addLog(logMsg);
            if (msg.game_over) handleGameOver(msg.winner);
        } else addLog(`⚠ ${msg.message}`);
        break;
    case "error": addLog(`❌ ${msg.message}`); break;
    case "chat": addLog(`💬 ${msg.message}`); break;
    case "pong": break;
    }
}

// ---- 交互 ----
function onCellClick(r, c) {
    if (!started) { addLog("游戏尚未开始"); return; }
    if (gameOver) { addLog("游戏已结束"); return; }
    if (mySlot === null) { addLog("请先坐下再落子"); return; }
    if (mySlot !== currentTurn) { addLog("不是你的回合"); return; }
    if (boardState[idx(r, c)] !== 0) { addLog("该位置已有棋子"); return; }
    send({ type: "move", row: r, col: c });
}

bs.addEventListener("click", () => send({ type: "start_game" }));
document.getElementById("btnReset").addEventListener("click", () => send({ type: "reset" }));

// ---- 启动 ----
buildBoard();
connectWS();
setInterval(() => send({ type: "ping" }), 30000);
addLog("欢迎！点击座位坐下或添加机器人…");
