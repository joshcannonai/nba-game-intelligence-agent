import { useEffect, useRef, useState } from "react";
import system from "./data/system.json";
import schedule from "./data/games.json";

// The page is served from Vercel but the model and the datasets live on the
// presenter's laptop, so the agent tab talks to a bridge on localhost. Vercel's
// own functions run in a cloud container and cannot reach it; the browser can,
// because during a screen share the browser IS the laptop.
// Same-origin when the bridge is serving this build (the demo path: the bridge
// mounts ui/web/dist, so http://localhost:8000 is the whole app and there is no
// cross-origin hop to fail). Falls back to the loopback address when the page came
// from Vercel, which is the review path -- teammates get Tools and System, and the
// Agent tab tells them plainly that it only runs on the presenter's machine.
const BRIDGE = location.port === "8000" ? "" : "http://localhost:8000";

type Health = {
	bridge: boolean;
	ollama: boolean;
	models: string[];
	source: string;
};
type Step = {
	kind: "call" | "back" | "err";
	name: string;
	detail: string;
	t: number;
};

type Game = { id: string; d: string; a: string; h: string; asOf: string };
const GAMES: Game[] = schedule.games;
const NAMES: Record<string, string> = schedule.names;

const DAY = {
	weekday: "short",
	month: "short",
	day: "numeric",
	year: "numeric",
} as const;
function labelFor(g: Game) {
  const when = new Date(g.d + "T12:00:00").toLocaleDateString(undefined, DAY);
  return `${NAMES[g.a] ?? g.a} at ${NAMES[g.h] ?? g.h}  ·  ${when}`;
}
export function matches(g: Game, q: string) {
  if (!q) return true;
  const hay = (g.a + " " + g.h + " " + (NAMES[g.a] ?? "") + " " + (NAMES[g.h] ?? "") + " " + g.d).toLowerCase();
  return q.toLowerCase().split(/\s+/).every((term) => hay.includes(term));
}

const ICON = {
	width: 14,
	height: 14,
	fill: "none",
	stroke: "currentColor",
	strokeWidth: 1.75,
	strokeLinecap: "round" as const,
	strokeLinejoin: "round" as const,
};

const ArrowOut = () => (
	<svg viewBox="0 0 16 16" {...ICON} aria-hidden="true">
		<path d="M2 8h11M9 4l4 4-4 4" />
	</svg>
);
const ArrowBack = () => (
	<svg viewBox="0 0 16 16" {...ICON} aria-hidden="true">
		<path d="M14 8H3M7 4L3 8l4 4" />
	</svg>
);
const Alert = () => (
	<svg viewBox="0 0 16 16" {...ICON} aria-hidden="true">
		<path d="M8 5v4M8 11.5v.01M8 2l6 11H2z" />
	</svg>
);
const Chevron = () => (
	<svg viewBox="0 0 16 16" {...ICON} width="12" height="12" aria-hidden="true">
		<path d="M6 3l5 5-5 5" />
	</svg>
);


// Every team's fixtures, built once. A team appears in a game as home or away, so
// each game lands in two buckets: the picker wants "this team's season", not
// "this team's home games".
const BY_TEAM: Record<string, Game[]> = (() => {
	const out: Record<string, Game[]> = {};
	for (const g of GAMES) {
		(out[g.h] ||= []).push(g);
		(out[g.a] ||= []).push(g);
	}
	for (const k of Object.keys(out)) out[k].sort((x, y) => x.d.localeCompare(y.d));
	return out;
})();
const TEAM_CODES = Object.keys(BY_TEAM).sort((a, b) =>
	(NAMES[a] ?? a).localeCompare(NAMES[b] ?? b),
);

function GamePicker({
	open,
	onClose,
	onPick,
	current,
}: {
	open: boolean;
	onClose: () => void;
	onPick: (g: Game) => void;
	current: Game;
}) {
	const [q, setQ] = useState("");
	const [expanded, setExpanded] = useState<string | null>(current.h);

	useEffect(() => {
		if (!open) return;
		const onKey = (e: KeyboardEvent) => {
			if (e.key === "Escape") onClose();
		};
		window.addEventListener("keydown", onKey);
		document.body.style.overflow = "hidden";
		return () => {
			window.removeEventListener("keydown", onKey);
			document.body.style.overflow = "";
		};
	}, [open, onClose]);

	if (!open) return null;

	const teams = TEAM_CODES.filter((code) =>
		!q ? true : (code + " " + (NAMES[code] ?? "")).toLowerCase().includes(q.toLowerCase()),
	);

	return (
		<div className="sheet" role="dialog" aria-modal="true" aria-label="Choose a game">
			<div className="sheet-head">
				<div className="sheet-title">
					<strong>Choose a game</strong>
					<span>
						{GAMES.length.toLocaleString()} games, {TEAM_CODES.length} teams, 2025-26
					</span>
				</div>
				<input
					type="search"
					autoFocus
					value={q}
					placeholder="Filter teams"
					onChange={(e) => setQ(e.target.value)}
				/>
				<button className="sheet-close" onClick={onClose} aria-label="Close">
					<svg viewBox="0 0 16 16" {...ICON} width="18" height="18" aria-hidden="true">
						<path d="M4 4l8 8M12 4l-8 8" />
					</svg>
				</button>
			</div>

			<div className="sheet-body">
				{teams.length === 0 && <p className="note">No team matches that.</p>}
				{teams.map((code) => {
					const isOpen = expanded === code;
					const games = BY_TEAM[code];
					return (
						<div className="team" key={code}>
							<button
								className="team-row"
								aria-expanded={isOpen}
								onClick={() => setExpanded(isOpen ? null : code)}
							>
								<span className="team-chev" data-open={isOpen}>
									<Chevron />
								</span>
								<span className="team-name">{NAMES[code] ?? code}</span>
								<span className="team-code">{code}</span>
								<span className="team-count">{games.length} games</span>
							</button>
							{isOpen && (
								<div className="team-games">
									{games.map((g) => {
										const opp = g.h === code ? g.a : g.h;
										const home = g.h === code;
										return (
											<button
												key={g.id}
												className={
													"game-row" + (g.id === current.id ? " is-current" : "")
												}
												onClick={() => {
													onPick(g);
													onClose();
												}}
											>
												<span className="game-date">
													{new Date(g.d + "T12:00:00").toLocaleDateString(
														undefined,
														DAY,
													)}
												</span>
												<span className="game-vs">
													{home ? "vs" : "at"} {NAMES[opp] ?? opp}
												</span>
												<span className="game-id">{g.id}</span>
											</button>
										);
									})}
								</div>
							)}
						</div>
					);
				})}
			</div>
		</div>
	);
}


const Sun = () => (
	<svg viewBox="0 0 16 16" {...ICON} width="15" height="15" aria-hidden="true">
		<circle cx="8" cy="8" r="3.2" />
		<path d="M8 1v1.6M8 13.4V15M15 8h-1.6M2.6 8H1M12.9 3.1l-1.1 1.1M4.2 11.8l-1.1 1.1M12.9 12.9l-1.1-1.1M4.2 4.2L3.1 3.1" />
	</svg>
);
const Moon = () => (
	<svg viewBox="0 0 16 16" {...ICON} width="15" height="15" aria-hidden="true">
		<path d="M13.5 9.5A5.8 5.8 0 016.5 2.5a5.8 5.8 0 107 7z" />
	</svg>
);

function ThemeToggle() {
	// Dark is the default because the use scene is a shared screen in a lit room.
	// The choice is remembered so a presenter who switches once is not switched
	// back by their OS preference mid-demo.
	const [theme, setTheme] = useState<"dark" | "light">(
		() => (localStorage.getItem("theme") as "dark" | "light") ?? "dark",
	);
	useEffect(() => {
		document.documentElement.setAttribute("data-theme", theme);
		localStorage.setItem("theme", theme);
	}, [theme]);
	return (
		<button
			className="theme-btn"
			onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
			aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
		>
			{theme === "dark" ? <Sun /> : <Moon />}
		</button>
	);
}

function Pill({ ok, label }: { ok: boolean | null; label: string }) {
	return (
		<span className="pill">
			<span className={"dot " + (ok === null ? "" : ok ? "on" : "off")} />
			{label}
		</span>
	);
}

function Row({ term, children }: { term: string; children: React.ReactNode }) {
	return (
		<div className="row">
			<dt>{term}</dt>
			<dd>{children}</dd>
		</div>
	);
}


function pctOf(n: unknown) {
	return typeof n === "number" ? Math.round(n * 100) : null;
}

/** Parse the agent's answer. It is asked for JSON and usually fences it. */
function parseReport(raw: string): Record<string, unknown> | null {
	const fenced = raw.match(/```(?:json)?\s*([\s\S]*?)```/);
	const body = (fenced ? fenced[1] : raw).trim();
	const start = body.indexOf("{");
	const end = body.lastIndexOf("}");
	if (start < 0 || end <= start) return null;
	try {
		return JSON.parse(body.slice(start, end + 1));
	} catch {
		return null;
	}
}

function Tldr({ raw, game }: { raw: string; game: Game }) {
	const r = parseReport(raw);
	if (!r) return null;
	const home = pctOf(r.home_win_prob);
	const away = pctOf(r.away_win_prob);
	if (home === null || away === null) return null;
	const favHome = home >= away;
	const favAbbr = favHome ? game.h : game.a;
	const favName = NAMES[favAbbr] ?? favAbbr;
	const favPct = favHome ? home : away;
	const factors = Array.isArray(r.key_factors) ? (r.key_factors as string[]) : [];
	const missing = Array.isArray(r.missing) ? (r.missing as string[]) : [];
	const confidence = favPct >= 70 ? "a clear favourite" : favPct >= 58 ? "a moderate edge" : "close to a coin flip";

	return (
		<>
			<h2 className="sec">In short</h2>
			<div className="tldr">
				<p className="verdict">
					<strong>{favName}</strong> to win, {favPct}%. The model calls that{" "}
					{confidence}.
				</p>

				<div className="odds" role="img" aria-label={`${NAMES[game.a] ?? game.a} ${away}%, ${NAMES[game.h] ?? game.h} ${home}%`}>
					<div className="bar">
						<span className="away" style={{ width: away + "%" }} />
						<span className="home" style={{ width: home + "%" }} />
					</div>
					<div className="ends">
						<span>
							{NAMES[game.a] ?? game.a} <b>{away}%</b>
						</span>
						<span>
							<b>{home}%</b> {NAMES[game.h] ?? game.h}
						</span>
					</div>
				</div>

				{typeof r.narrative === "string" && r.narrative && (
					<p className="narrative">{r.narrative}</p>
				)}

				{factors.length > 0 && (
					<div className="reasons">
						<h3>Why</h3>
						<ul>
							{factors.map((f, i) => (
								<li key={i}>{f}</li>
							))}
						</ul>
					</div>
				)}

				<div className="reasons">
					<h3>What it could not find</h3>
					{missing.length === 0 ? (
						<p className="none">
							Nothing. Every tool it called returned data.
						</p>
					) : (
						<ul>
							{missing.map((m, i) => (
								<li key={i}>{m}</li>
							))}
						</ul>
					)}
				</div>
			</div>
		</>
	);
}

function AgentTab({ health }: { health: Health | null }) {
	const [arm, setArm] = useState<"A" | "B" | "C">("C");
	const [pickerOpen, setPickerOpen] = useState(false);
	const [matchup, setMatchup] = useState<Game>(
		GAMES.find((g) => g.id === "CHI-ORL-2025-12-01") ?? GAMES[0],
	);
	const [steps, setSteps] = useState<Step[]>([]);
	const [final, setFinal] = useState("");
	const [running, setRunning] = useState(false);
	const [elapsed, setElapsed] = useState(0);
	const logRef = useRef<HTMLDivElement>(null);

	useEffect(() => {
		// Scroll the log box, never the page: yanking the whole window while the
		// presenter is talking over it is worse than a list that fills quietly.
		const box = logRef.current;
		if (box) box.scrollTop = box.scrollHeight;
	}, [steps.length]);

	async function run() {
		setSteps([]);
		setFinal("");
		setRunning(true);
		setElapsed(0);
		try {
			if (arm === "A") {
				const r = await fetch(BRIDGE + "/api/predict", {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({
						matchup_id: matchup.id,
						as_of_date: matchup.asOf,
					}),
				});
				setFinal(JSON.stringify(await r.json(), null, 2));
				setRunning(false);
				return;
			}
			const res = await fetch(BRIDGE + "/api/run", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					matchup_id: matchup.id,
					as_of_date: matchup.asOf,
					include_model: arm === "C",
					model_backend: "ollama",
				}),
			});
			if (!res.body) throw new Error("no response stream");
			const reader = res.body.getReader();
			const decoder = new TextDecoder();
			let buffer = "";
			for (;;) {
				const { done, value } = await reader.read();
				if (done) break;
				buffer += decoder.decode(value, { stream: true });
				const frames = buffer.split("\n\n");
				buffer = frames.pop() ?? "";
				for (const frame of frames) {
					const lines = frame.split("\n");
					const evLine = lines.find((l) => l.startsWith("event: "));
					const dataLine = lines.find((l) => l.startsWith("data: "));
					if (!evLine || !dataLine) continue;
					const ev = evLine.slice(7).trim();
					const data = JSON.parse(dataLine.slice(6));
					if (typeof data.elapsed === "number") setElapsed(data.elapsed);
					if (ev === "tool_call")
						setSteps((s) => [
							...s,
							{
								kind: "call",
								name: data.name,
								detail: JSON.stringify(data.args),
								t: data.elapsed,
							},
						]);
					else if (ev === "tool_result")
						setSteps((s) => [
							...s,
							{
								kind: "back",
								name: data.name,
								detail: data.content,
								t: data.elapsed,
							},
						]);
					else if (ev === "error")
						setSteps((s) => [
							...s,
							{
								kind: "err",
								name: "run failed",
								detail: data.message,
								t: data.elapsed,
							},
						]);
					else if (ev === "final") setFinal(data.content);
				}
			}
		} catch (err) {
			setSteps((s) => [
				...s,
				{
					kind: "err",
					name: "cannot reach the local bridge",
					detail: String(err) + ". Start it with: python -m ui.serve",
					t: 0,
				},
			]);
		} finally {
			setRunning(false);
		}
	}

	return (
		<div className="panel">
			{health !== null && !health.bridge && (
				<div className="banner">
					<strong>The agent is not reachable.</strong> It runs on the
					presenter's machine, not in the cloud, so this tab only works on that
					laptop. Start it with <code>python -m ui.serve</code> and reload.
					Tools and System work without it.
				</div>
			)}

			<div className="slab console">
				<div className="controls">
					<div className="field">
						<span>Which approach</span>
						<div className="segmented" role="group" aria-label="Which approach">
							{(
								[
									["A", "Model only"],
									["B", "Agent only"],
									["C", "Both"],
								] as const
							).map(([k, label]) => (
								<button
									key={k}
									type="button"
									aria-pressed={arm === k}
									onClick={() => setArm(k)}
								>
									{label}
								</button>
							))}
						</div>
					</div>
					<div className="field grow">
						<span>Game to predict</span>
						<button className="picker" onClick={() => setPickerOpen(true)}>
							<span>{labelFor(matchup)}</span>
							<svg viewBox="0 0 16 16" {...ICON} width="13" height="13" aria-hidden="true">
								<path d="M4 6l4 4 4-4" />
							</svg>
						</button>
					</div>
					<label className="field">
						<span>Knows nothing after</span>
						<input type="date" value={matchup.asOf} readOnly tabIndex={-1} />
					</label>
					<button className="run" onClick={run} disabled={running}>
						{running
							? "Thinking, " + elapsed.toFixed(0) + "s"
							: "Run the agent"}
					</button>
				</div>
				<p className="note tight" style={{ marginTop: 16 }}>
					{arm === "A"
						? "The fitted model on its own, no language model involved. Returns instantly, because scoring a logistic regression is a dot product. "
						: arm === "B"
							? "The agent without the model tool. It has to reason its own way to a probability from the retrieval tools alone. "
							: "The agent given the model's number as one input among several. "}
					Every
					call below is filtered to the as-of date before it returns, so the
					agent cannot see the result it is being asked to predict.
				</p>
			</div>

			<GamePicker
				open={pickerOpen}
				onClose={() => setPickerOpen(false)}
				onPick={setMatchup}
				current={matchup}
			/>

			{steps.length > 0 && (
				<>
					<h2 className="sec">What it did</h2>
					<p className="note">
						Each line is a real function call, in the order the model chose to
						make it. {steps.length} so far.
					</p>
					<div className="log" ref={logRef}>
						{steps.map((s, i) => (
							<div key={i} className={"entry " + s.kind}>
								<div className="t">{s.t.toFixed(1)}s</div>
								<div className="icon">
									{s.kind === "call" ? (
										<ArrowOut />
									) : s.kind === "back" ? (
										<ArrowBack />
									) : (
										<Alert />
									)}
								</div>
								<div>
									<div className="name">{s.name}</div>
									<div className="detail">
										{s.detail.length > 300
											? s.detail.slice(0, 300) + "..."
											: s.detail}
									</div>
								</div>
							</div>
						))}
					</div>
				</>
			)}

			{final && (
				<>
					<h2 className="sec">{arm === "A" ? "What the model returned" : "What it concluded"}</h2>
					<p className="note">
						{arm === "A"
							? "The eight features it used are in the payload, so the number can be checked by hand."
							: "The agent must return valid JSON, including anything it could not find."}
					</p>
					<pre className="out">{final}</pre>
					<Tldr raw={final} game={matchup} />
				</>
			)}
		</div>
	);
}

function ToolsTab() {
	const live = system.tools.filter((t) => t.status === "live").length;
	return (
		<div className="panel">
			<h2 className="sec first">Everything the model is allowed to do</h2>
			<p className="note">
				It cannot query a database, browse the web, or invent a number. It can
				call exactly these {system.tools.length} functions, {live} of which
				return real data today. Each one carries a written rule set loaded into
				the prompt at startup, so editing a rule changes behaviour with no code
				change.
			</p>
			<div className="tools">
				{system.tools.map((t) => (
					<div className="tool" key={t.name}>
						<div className="who">
							<span className="tool-name">{t.name}</span>
							<span className="when">{t.use_when}</span>
							<span className="reads">
								reads{" "}
								{((t as any).source_links ?? []).length === 0
									? "nothing yet"
									: ((t as any).source_links as { label: string; url: string; path: string }[]).map(
											(f, i) => (
												<span key={f.url}>
													{i > 0 && ", "}
													<a href={f.url} target="_blank" rel="noreferrer" title={f.path}>
														{f.label}
													</a>
												</span>
											),
										)}
							</span>
						</div>
						<span className={"state " + t.status}>
							<span className="dot" />
							{t.status === "live" ? "real data" : "awaiting input"}
						</span>
						{t.skill && (
							<div className="foot">
								<details>
									<summary>
										<Chevron />{" "}{t.skill_path}
									</summary>
									<pre>{t.skill_raw ?? t.skill}</pre>
								</details>
							</div>
						)}
					</div>
				))}
			</div>
		</div>
	);
}


function PromptTab() {
	const p = system.prompt;
	return (
		<div className="panel">
			<h2 className="sec first">Exactly what the model is told</h2>
			<p className="note">
				This is the complete string handed to Gemma on every run, not a summary of
				it. It is assembled at startup from two places: the base rules in{" "}
				<code>{p.base_in}</code>, and a block composed from the{" "}
				<code>skills/</code> files by <code>{p.composed_in}</code>. Nothing else
				reaches the model except tool results.
			</p>
			<dl className="readout">
				<Row term="Base rules">
					{p.base_chars.toLocaleString()} characters
					<span className="after">{p.base_in}</span>
				</Row>
				<Row term="Tool skills block">
					{p.skills_chars.toLocaleString()} characters
					<span className="after">composed from skills/*.md at startup</span>
				</Row>
				<Row term="Total sent to the model">
					<em>{p.total_chars.toLocaleString()} characters</em>
					<span className="after">every run</span>
				</Row>
				<Row term="Same prompt without the model tool">
					{p.arm_b_total.toLocaleString()} characters
					<span className="after">
						the {(p.total_chars - p.arm_b_total).toLocaleString()} character
						difference is the entire experiment
					</span>
				</Row>
			</dl>

			<h2 className="sec">The base rules</h2>
			<p className="note">
				These say when to call what, and what to do when something is missing. The
				rule about never quoting a betting line is here rather than in a skill,
				because it applies whether or not the tool exists.
			</p>
			<pre className="out">{p.base}</pre>

			<h2 className="sec">The tool skills block, appended after it</h2>
			<p className="note">
				One entry per tool the agent was actually given. This is where "when to read
				it and how to weight it" lives: each entry states when to call the tool, how
				to read what comes back, and the rules it must follow. Editing a{" "}
				<code>skills/*.md</code> file changes behaviour with no code change, which is
				why the evaluation is re-run after any edit.
			</p>
			<pre className="out">{p.skills_block}</pre>
		</div>
	);
}

function DataTab() {
	return (
		<div className="panel">
			<h2 className="sec first">Everything it reads</h2>
			<p className="note">
				Nine files on disk. No database, no API calls at run time, no network. Two of
				them are rebuilt copies with the answer columns removed, because the raw
				versions keep the result in the same row as the inputs.
			</p>
			<div className="tablewrap">
				<table>
					<thead>
						<tr>
							<th>File</th>
							<th>What it is</th>
							<th>Rows</th>
							<th>Size</th>
							<th>Collected by</th>
						</tr>
					</thead>
					<tbody>
						{system.datasets.map((f) => (
							<tr key={f.path}>
								<td className="mono">
									<a href={f.url} target="_blank" rel="noreferrer">
										{f.path.split("/").pop()}
									</a>
								</td>
								<td>{f.what}</td>
								<td className="mono">{f.rows?.toLocaleString() ?? "?"}</td>
								<td className="mono">{f.kb.toLocaleString()} KB</td>
								<td>{f.who}</td>
							</tr>
						))}
					</tbody>
				</table>
			</div>
			<p className="note" style={{ marginTop: 16 }}>
				<code>player_features_2026.csv</code> and <code>odds_only.csv</code> are the
				stripped copies. The engineered export keeps a player's points beside the
				rolling averages that predict them, and the raw odds file keeps the final
				score beside the line, so the agent reads versions with those columns
				removed. Tests assert they stay removed.
			</p>
		</div>
	);
}


type ArmStat = { n: number; accuracy: number; log_loss: number; brier: number };
const LABELS: Record<string, string> = {
	arm_A: "A. Model only",
	arm_B: "B. Agent only",
	arm_C: "C. Agent + model",
	vegas: "Vegas closing line",
	always_home: "Always pick home",
};

function pct(n: number) {
	return (n * 100).toFixed(1) + "%";
}

function ArmTable({
	title,
	note,
	data,
	order,
}: {
	title: string;
	note: string;
	data: Record<string, ArmStat>;
	order: string[];
}) {
	const rows = order.filter((k) => data[k]);
	const best = rows.reduce((a, b) => (data[b].accuracy > data[a].accuracy ? b : a), rows[0]);
	return (
		<>
			<h2 className="sec">{title}</h2>
			<p className="note">{note}</p>
			<div className="tablewrap">
				<table>
					<thead>
						<tr>
							<th>Approach</th>
							<th>Accuracy</th>
							<th>Log loss</th>
							<th>Brier</th>
							<th>Games</th>
						</tr>
					</thead>
					<tbody>
						{rows.map((k) => (
							<tr key={k}>
								<td>{LABELS[k] ?? k}</td>
								{/* Only the leader is coloured. Tinting every arm made a losing
								    arm read as a good result, which is the opposite of the finding. */}
								<td className={"mono" + (k === best ? " win" : "")}>
									{pct(data[k].accuracy)}
								</td>
								{/* Always-pick-home predicts a hard 1.0, so its log loss is a
								    divide-by-epsilon artefact rather than a measurement. */}
								<td className="mono">
									{k === "always_home" ? "n/a" : data[k].log_loss.toFixed(3)}
								</td>
								<td className="mono">
									{k === "always_home" ? "n/a" : data[k].brier.toFixed(3)}
								</td>
								<td className="mono">{data[k].n.toLocaleString()}</td>
							</tr>
						))}
					</tbody>
				</table>
			</div>
		</>
	);
}

function CompareTab() {
	const a = system.arms as any;
	return (
		<div className="panel">
			<h2 className="sec first">Three ways to predict the same games</h2>
			<p className="note">
				The project's actual question. Every number below was produced by the replay
				harness and is reproducible from the CSVs in the repo.
			</p>
			<dl className="readout">
				{Object.entries(a.definitions as Record<string, string>).map(([k, v]) => (
					<Row key={k} term={LABELS[k] ?? k}>
						<span style={{ fontFamily: "var(--sans)", fontSize: 14, color: "var(--ink-2)" }}>
							{v}
						</span>
					</Row>
				))}
			</dl>

			<ArmTable
				title="The full season"
				note="Every game of 2025-26. Only the model can be run at this scale: the agent takes about 40 seconds a game locally, which is 15 hours for the season."
				data={a.season}
				order={["arm_A", "vegas", "always_home"]}
			/>

			<ArmTable
				title="All three arms, 40 games"
				note="The same 40 games given to each approach. This is the comparison the report is built around, and the result was negative: handing the agent the model's number made it worse than the model alone."
				data={a.sample40}
				order={["arm_A", "arm_B", "arm_C", "vegas", "always_home"]}
			/>

			<ArmTable
				title="A second sample, different seed"
				note="A different 40 games, to check the first sample was not a fluke. It was not."
				data={a.sample40_seed1}
				order={["arm_A", "arm_B", "arm_C", "vegas", "always_home"]}
			/>

			<ArmTable
				title="After the skills layer was added"
				note="The same 40 games as the first sample, with nothing changed but the written rules the agent is given. Arm C recovered most of the gap without a line of code changing."
				data={a.skills_after}
				order={["arm_A", "arm_C", "vegas"]}
			/>

			<p className="note" style={{ marginTop: 22 }}>
				Forty games is a small sample and these are single runs, so treat the
				direction as the finding and not the decimal. The season row is the one with
				statistical weight behind it.
			</p>
		</div>
	);
}

function SystemTab() {
	const { llm, win_model, stat_model, baselines, gate_rules } = system;
	const pct = (n: number) => (n * 100).toFixed(1) + "%";
	return (
		<div className="panel">
			<h2 className="sec first">The language model</h2>
			<p className="note">{llm.why_cutoff_matters}</p>
			<dl className="readout">
				<Row term="Model">
					{llm.name}
					<span className="after">
						{llm.params} parameters, {llm.quant}
					</span>
				</Row>
				<Row term="Knowledge cutoff">
					<em>{llm.cutoff}</em>
					<span className="after">before the season it is tested on</span>
				</Row>
				<Row term="Where it runs">
					{llm.runtime}
					<span className="after">no API key, nothing leaves the laptop</span>
				</Row>
			</dl>

			<h2 className="sec">The win-probability model</h2>
			<p className="note">
				{win_model.kind.replace(/_/g, " ")} over {win_model.features} features,
				trained on {win_model.train_seasons.join(" and ")} and tested on a
				season it never saw.
			</p>
			<dl className="readout">
				<Row term="Accuracy on 2025-26">
					<em>{pct(win_model.accuracy)}</em>
					<span className="after">
						across {win_model.n.toLocaleString()} games
					</span>
				</Row>
				<Row term="Always pick the home team">
					{pct(baselines.always_home)}
					<span className="after">the floor worth beating</span>
				</Row>
				<Row term="Vegas closing line">
					{pct(baselines.vegas_closing)}
					<span className="after">the ceiling we measure against</span>
				</Row>
				<Row term="Log loss / Brier">
					{win_model.log_loss} / {win_model.brier}
					<span className="after">calibration, lower is better</span>
				</Row>
			</dl>

			<h2 className="sec">The stat-line model</h2>
			<p className="note">
				Trained on {stat_model.train_seasons.join(", ")} and validated on{" "}
				{stat_model.test_season}, never fitted on the season being replayed. It
				beats simply using a player's last five games, but only just, and the
				agent is told to say so rather than imply the model found something
				extra.
			</p>
			<div className="tablewrap">
				<table>
					<thead>
						<tr>
							<th>Projecting</th>
							<th>Model error</th>
							<th>Their last 5 games</th>
							<th>R squared</th>
						</tr>
					</thead>
					<tbody>
						{Object.entries(stat_model.targets).map(([k, v]) => (
							<tr key={k}>
								<td>{k.replace("total_", "")}</td>
								<td className="mono win">{v.mae.toFixed(3)}</td>
								<td className="mono">{v.baseline_mae.toFixed(3)}</td>
								<td className="mono">{v.r2.toFixed(3)}</td>
							</tr>
						))}
					</tbody>
				</table>
			</div>

			<h2 className="sec">How the future is kept out</h2>
			<p className="note">
				Every read passes through one file. There is no second path to the data,
				which is what makes the claim checkable rather than a promise.
			</p>
			<dl className="readout">
				{gate_rules.map((r) => (
					<div className="row" key={r.title}>
						<dt>{r.title}</dt>
						<dd
							style={{
								fontFamily: "var(--sans)",
								fontSize: 14,
								color: "var(--ink-2)",
								lineHeight: 1.6,
							}}
						>
							{r.body}
						</dd>
					</div>
				))}
			</dl>
		</div>
	);
}

export default function App() {
	const [tab, setTab] = useState<"agent" | "compare" | "prompt" | "tools" | "data" | "system">("agent");
	const [health, setHealth] = useState<Health | null>(null);

	useEffect(() => {
		// Timeout, not just catch. From the hosted copy this request crosses into a
		// private network and Chrome may leave it pending rather than reject it, which
		// would hold the UI unresolved forever and show a reviewer nothing at all.
		fetch(BRIDGE + "/api/health", { signal: AbortSignal.timeout(2500) })
			.then((r) => r.json())
			.then(setHealth)
			.catch(() =>
				setHealth({
					bridge: false,
					ollama: false,
					models: [],
					source: "unavailable",
				}),
			);
	}, []);

	return (
		<div className="shell">
			<header className="top">
				<div className="brand">
					<h1>NBA Game Intelligence Agent</h1>
					<span className="course">CECS 499</span>
				</div>
				<span className="spacer" />
				<div className="pills">
					<Pill
						ok={health && health.bridge}
						label={health?.bridge ? "agent reachable" : "agent offline"}
					/>
					<Pill
						ok={health && health.ollama}
						label={health?.ollama ? "gemma4 loaded" : "ollama off"}
					/>
					<ThemeToggle />
				</div>
			</header>

			<nav className="tabs" role="tablist">
				{(
					[
						["agent", "Agent"],
						["compare", "Compare"],
						["prompt", "Prompt"],
						["tools", "Tools"],
						["data", "Data"],
						["system", "System"],
					] as const
				).map(([k, label]) => (
					<button
						key={k}
						role="tab"
						aria-selected={tab === k}
						onClick={() => setTab(k)}
					>
						{label}
					</button>
				))}
			</nav>

			{tab === "agent" && <AgentTab health={health} />}
			{tab === "compare" && <CompareTab />}
			{tab === "prompt" && <PromptTab />}
			{tab === "tools" && <ToolsTab />}
			{tab === "data" && <DataTab />}
			{tab === "system" && <SystemTab />}
		</div>
	);
}
