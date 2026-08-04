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

const DAY = { weekday: "short", month: "short", day: "numeric" } as const;
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

function AgentTab({ health }: { health: Health | null }) {
	const [matchup, setMatchup] = useState<Game>(GAMES.find((g) => g.id === "CHI-ORL-2025-12-01") ?? GAMES[0]);
	const [steps, setSteps] = useState<Step[]>([]);
	const [final, setFinal] = useState("");
	const [running, setRunning] = useState(false);
	const [elapsed, setElapsed] = useState(0);
	const endRef = useRef<HTMLDivElement>(null);

	useEffect(() => {
		if (steps.length)
			endRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
	}, [steps.length]);

	async function run() {
		setSteps([]);
		setFinal("");
		setRunning(true);
		setElapsed(0);
		try {
			const res = await fetch(BRIDGE + "/api/run", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					matchup_id: matchup.id,
					as_of_date: matchup.asOf,
					include_model: true,
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
					<label className="field">
						<span>Game to predict</span>
						<select
							value={matchup.id}
							onChange={(e) =>
								setMatchup(
									GAMES.find((m) => m.id === e.target.value) ?? GAMES[0],
								)
							}
						>
							{GAMES.map((m) => (
								<option key={m.id} value={m.id}>
									{labelFor(m)}
								</option>
							))}
						</select>
					</label>
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
					Gemma 4 runs on this laptop and takes roughly 40 seconds a game. Every
					call below is filtered to the as-of date before it returns, so the
					agent cannot see the result it is being asked to predict.
				</p>
			</div>

			{steps.length > 0 && (
				<>
					<h2 className="sec">What it did</h2>
					<p className="note">
						Each line is a real function call, in the order the model chose to
						make it.
					</p>
					<div className="log">
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
						<div ref={endRef} />
					</div>
				</>
			)}

			{final && (
				<>
					<h2 className="sec">What it concluded</h2>
					<p className="note">
						The agent must return valid JSON, including anything it could not
						find.
					</p>
					<pre className="out">{final}</pre>
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
								reads {t.sources.join(", ") || "nothing yet"}
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
										<Chevron />
										the rules it is given
									</summary>
									<pre>{t.skill}</pre>
								</details>
							</div>
						)}
					</div>
				))}
			</div>
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
	const [tab, setTab] = useState<"agent" | "tools" | "system">("agent");
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
				</div>
			</header>

			<nav className="tabs" role="tablist">
				{(
					[
						["agent", "Agent"],
						["tools", "Tools"],
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
			{tab === "tools" && <ToolsTab />}
			{tab === "system" && <SystemTab />}
		</div>
	);
}
