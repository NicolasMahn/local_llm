import Clutter from 'gi://Clutter';
import GLib from 'gi://GLib';
import GObject from 'gi://GObject';
import Gio from 'gi://Gio';
import St from 'gi://St';

import * as Main from 'resource:///org/gnome/shell/ui/main.js';
import * as PanelMenu from 'resource:///org/gnome/shell/ui/panelMenu.js';
import * as PopupMenu from 'resource:///org/gnome/shell/ui/popupMenu.js';
import {Extension} from 'resource:///org/gnome/shell/extensions/extension.js';

const GATEWAY = 'http://127.0.0.1:8400';
// What other machines on the network use; the hostname resolves through mDNS.
const PUBLIC_GATEWAY = `http://${GLib.get_host_name()}.local:8400`;
// Everything measured comes from monitor.py, which samples every 2 s.
const STATS = GLib.build_filenamev([GLib.get_user_state_dir(), 'local-llm', 'stats.json']);
const STALE_SECONDS = 15; // older than this, the monitor is not running
const POLL_SECONDS = 3;
const HISTORY = 60; // monitor samples shown in each graph, so two minutes

// What a model's dot can show; stylesheet.css gives each one its color.
const State = {
    OFF: 'off',
    CHANGING: 'changing', // starting, waiting its turn, or stopping
    LIVE: 'live',
    FAILED: 'failed',
};

Gio._promisify(Gio.Subprocess.prototype, 'communicate_utf8_async');

async function run(argv) {
    const proc = Gio.Subprocess.new(argv,
        Gio.SubprocessFlags.STDOUT_PIPE | Gio.SubprocessFlags.STDERR_MERGE);
    const [stdout] = await proc.communicate_utf8_async(null, null);
    if (!proc.get_successful())
        throw new Error(stdout.trim() || `${argv.join(' ')} failed`);
    return stdout;
}

function readStats() {
    try {
        const [, bytes] = GLib.file_get_contents(STATS);
        const stats = JSON.parse(new TextDecoder().decode(bytes));
        const age = (Date.now() - Date.parse(stats.updated)) / 1000;
        return age < STALE_SECONDS ? stats : null;
    } catch {
        return null;
    }
}

const Graph = GObject.registerClass(
class Graph extends St.DrawingArea {
    _init(ceiling) {
        super._init({styleClass: 'local-llm-graph', xExpand: true});
        this._ceiling = ceiling; // fixed top of the scale, or null to fit the peak
        this.samples = [];
    }

    setSamples(samples) {
        this.samples = samples.slice(-HISTORY);
        this.queue_repaint();
    }

    vfunc_repaint() {
        const [width, height] = this.get_surface_size();
        const cr = this.get_context();
        const top = this._ceiling ?? Math.max(1, ...this.samples);
        const barWidth = width / HISTORY;
        const color = this.get_theme_node().get_foreground_color();

        cr.setSourceRGBA(color.red / 255, color.green / 255, color.blue / 255,
            color.alpha / 255);
        // Newest on the right, so the graph scrolls left like a system monitor.
        const offset = HISTORY - this.samples.length;
        this.samples.forEach((value, index) => {
            const barHeight = Math.min(1, value / top) * (height - 2);
            cr.rectangle((offset + index) * barWidth, height - barHeight,
                Math.max(1, barWidth - 2), barHeight);
        });
        cr.fill();
        cr.$dispose();
    }
});

const GraphItem = GObject.registerClass(
class GraphItem extends PopupMenu.PopupBaseMenuItem {
    _init(ceiling = null) {
        super._init({reactive: false});
        this.graph = new Graph(ceiling);
        this.value = new St.Label({
            styleClass: 'local-llm-value',
            yAlign: Clutter.ActorAlign.CENTER,
        });
        this.add_child(this.graph);
        this.add_child(this.value);
    }

    show(samples, text) {
        this.graph.setSamples(samples);
        this.value.text = text;
    }
});

// A row that stays open when clicked, for switches and copying.
const StayOpenItem = GObject.registerClass(
class StayOpenItem extends PopupMenu.PopupBaseMenuItem {
    _init(onActivate) {
        super._init();
        this._onActivate = onActivate;
        this.lines = new St.BoxLayout({
            orientation: Clutter.Orientation.VERTICAL,
            xExpand: true,
            yAlign: Clutter.ActorAlign.CENTER,
        });
        this.title = new St.Label();
        this.detail = new St.Label({styleClass: 'local-llm-detail'});
        // Failure reasons can be long; wrap them instead of widening the menu.
        this.detail.clutterText.lineWrap = true;
        this.lines.add_child(this.title);
        this.lines.add_child(this.detail);
        this.add_child(this.lines);
    }

    activate() {
        this._onActivate();
    }
});

const StatusButton = GObject.registerClass(
class StatusButton extends PanelMenu.Button {
    _init(extension) {
        super._init(0.5, 'Local LLM');
        this._llm = GLib.build_filenamev([extension.path, 'llm']);
        this._models = new Map(); // name -> {item, dot, toggle, error, refused}
        this._starting = new Set(); // names whose `llm up` is still running
        this._stopping = new Set(); // names whose container is still shutting down

        const box = new St.BoxLayout({styleClass: 'panel-status-menu-box'});
        this._icon = new St.Icon({
            iconName: 'network-transmit-receive-symbolic',
            styleClass: 'system-status-icon',
        });
        this._label = new St.Label({text: '–', yAlign: Clutter.ActorAlign.CENTER});
        box.add_child(this._icon);
        box.add_child(this._label);
        this.add_child(box);

        // What the monitor found wrong, above everything else so it is seen first.
        this._problems = new PopupMenu.PopupMenuSection();
        this._problemsSeparator = new PopupMenu.PopupSeparatorMenuItem();
        this._shownProblems = '';
        this.menu.addMenuItem(this._problems);
        this.menu.addMenuItem(this._problemsSeparator);

        this._power = new GraphItem();
        this._tokens = new GraphItem();
        this._requests = new GraphItem();
        this._busy = new GraphItem(100);
        this._energy = new PopupMenu.PopupMenuItem('', {reactive: false});
        this._hardware = new PopupMenu.PopupMenuItem('', {reactive: false});
        for (const item of [this._power, this._tokens, this._requests, this._busy,
            this._energy, this._hardware])
            this.menu.addMenuItem(item);

        this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem());
        this._modelSection = new PopupMenu.PopupMenuSection();
        this.menu.addMenuItem(this._modelSection);

        this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem());
        this._key = '';
        this._addCopyRow('OpenAI', `${PUBLIC_GATEWAY}/v1`);
        this._addCopyRow('Anthropic', PUBLIC_GATEWAY);
        this._keyRow = this._addCopyRow('Key', () => this._key);
        // Opening this on a phone saves the key there, so the test chat just works.
        this._chatRow = this._addCopyRow('Chat', () => `${PUBLIC_GATEWAY}/?api=${this._key}`);

        this._load().catch(logError);
    }

    async _load() {
        const list = await run([this._llm, 'list']);
        for (const line of list.trim().split('\n')) {
            this._addModelRow(line.split(' ')[0]);
        }
        this._key = (await run([this._llm, 'key'])).trim();
        this._keyRow.shown = `${this._key.slice(0, 12)}…`;
        this._keyRow.detail.text = this._keyRow.shown;
        this._chatRow.shown = `${PUBLIC_GATEWAY}/?api=…`;
        this._chatRow.detail.text = this._chatRow.shown;

        this._poll().catch(logError);
        this._timer = GLib.timeout_add_seconds(GLib.PRIORITY_DEFAULT, POLL_SECONDS, () => {
            this._poll().catch(logError);
            return GLib.SOURCE_CONTINUE;
        });
    }

    _addCopyRow(title, text) {
        const value = typeof text === 'function' ? text : () => text;
        const row = new StayOpenItem(() => {
            St.Clipboard.get_default().set_text(St.ClipboardType.CLIPBOARD, value());
            row.detail.text = 'copied';
            GLib.timeout_add_seconds(GLib.PRIORITY_DEFAULT, 2, () => {
                row.detail.text = row.shown;
                return GLib.SOURCE_REMOVE;
            });
        });
        row.title.text = title;
        row.shown = typeof text === 'function' ? '' : text;
        row.detail.text = row.shown;
        this.menu.addMenuItem(row);
        return row;
    }

    // The switch holds what you asked for and only moves when clicked; the dot
    // shows what is actually happening.
    _addModelRow(name) {
        const model = {error: null, refused: null};
        model.toggle = new PopupMenu.Switch(false);
        model.item = new StayOpenItem(() => this._flip(name).catch(logError));
        model.item.title.text = name;
        model.dot = new St.Widget({yAlign: Clutter.ActorAlign.CENTER});
        model.item.insert_child_below(model.dot, model.item.lines);
        model.item.add_child(model.toggle);
        this._modelSection.addMenuItem(model.item);
        this._models.set(name, model);
    }

    async _flip(name) {
        const model = this._models.get(name);
        const on = !model.toggle.state;
        model.toggle.state = on;
        model.error = null;
        model.refused = null;
        if (!on) {
            this._starting.delete(name);
            this._stopping.add(name);
            this._show(model, State.CHANGING, 'stopping');
            try {
                await run([this._llm, 'stop', name]);
            } finally {
                this._stopping.delete(name);
            }
            return;
        }
        // `llm up` returns once the model answers, or right away if it failed.
        this._starting.add(name);
        this._show(model, State.CHANGING, 'starting');
        try {
            await run([this._llm, 'up', name]);
        } catch (error) {
            // A model that crashed explains itself in its log; `llm up` itself
            // only fails without a container when it refused to start one.
            const reason = error.message.split('\n').at(-1);
            model.refused = reason.startsWith(`${name} `) ? reason.slice(name.length + 1) : reason;
        } finally {
            this._starting.delete(name);
        }
    }

    async _poll() {
        // A slow podman call must not let polls pile up.
        if (this._polling)
            return;
        this._polling = true;
        try {
            await this._pollOnce();
        } finally {
            this._polling = false;
        }
    }

    async _pollOnce() {
        const stats = readStats();
        const containers = await this._containers().catch(() => new Map());

        for (const [name, model] of this._models) {
            const container = containers.get(`llm-${name}`);
            const live = stats?.models[name];
            const up = Boolean(live?.up);
            const failed = !up && container?.state !== 'running' &&
                !this._starting.has(name) && container && container.exitCode !== 0;
            if (!failed)
                model.error = null;

            let state, detail;
            if (this._stopping.has(name)) {
                [state, detail] = [State.CHANGING, 'stopping'];
            } else if (up) {
                const parts = [];
                if (live.running)
                    parts.push(`${live.running} running`);
                if (live.waiting)
                    parts.push(`${live.waiting} waiting`);
                if (live.tokens_per_second >= 1)
                    parts.push(`${Math.round(live.tokens_per_second)} tok/s`);
                if (live.cache >= 0.01)
                    parts.push(`cache ${Math.round(live.cache * 100)} %`);
                [state, detail] = [State.LIVE, parts.length ? parts.join(' · ') : 'ready'];
            } else if (container?.state === 'running') {
                [state, detail] = [State.CHANGING, `starting since ${container.since}`];
            } else if (this._starting.has(name)) {
                [state, detail] = [State.CHANGING, 'waiting for another model to start'];
            } else if (model.refused && !container) {
                [state, detail] = [State.FAILED, model.refused];
            } else if (failed) {
                model.error ??= await this._lastError(name);
                [state, detail] = [State.FAILED, model.error];
            } else {
                [state, detail] = [State.OFF, 'off'];
            }
            this._show(model, state, detail);

            // While a click is carried out the switch keeps what was asked for;
            // otherwise it tells the truth: on while the model runs or starts
            // (also when started elsewhere, e.g. ./llm), off when stopped or failed.
            const idle = !this._starting.has(name) && !this._stopping.has(name);
            if (idle)
                model.toggle.state = state === State.LIVE || state === State.CHANGING;
        }

        this._showMeasurements(stats);
    }

    _showProblems(problems) {
        const errors = problems.some(problem => problem.severity === 'error');
        this._icon.iconName = errors ? 'dialog-warning-symbolic' : 'network-transmit-receive-symbolic';
        // Rebuilt only on change, so an open menu does not flicker every poll.
        const key = JSON.stringify(problems.map(problem => problem.message));
        if (key === this._shownProblems)
            return;
        this._shownProblems = key;
        this._problems.removeAll();
        for (const problem of problems) {
            const item = new PopupMenu.PopupBaseMenuItem({reactive: false});
            const dot = new St.Widget({
                styleClass: `local-llm-dot ${problem.severity === 'error' ? State.FAILED : State.CHANGING}`,
                yAlign: Clutter.ActorAlign.CENTER,
            });
            const text = new St.Label({text: problem.message, styleClass: 'local-llm-detail', xExpand: true});
            text.clutterText.lineWrap = true;
            item.add_child(dot);
            item.add_child(text);
            this._problems.addMenuItem(item);
        }
        this._problemsSeparator.visible = problems.length > 0;
    }

    _showMeasurements(stats) {
        this._showProblems(stats?.problems ?? []);
        this._icon.opacity = stats?.gateway_up ? 255 : 100;
        this._label.text = stats?.gateway_up ? `${stats.requests}` : 'off';
        if (!stats) {
            this._energy.label.text = 'Nothing measured: the monitor is not running (./llm monitor)';
            this._hardware.label.text = '';
            return;
        }
        const history = stats.history;
        this._power.show(history.power_w, `${Math.round(stats.power_w)} W`);
        this._tokens.show(history.tokens_per_second, `${Math.round(stats.tokens_per_second)} tok/s`);
        this._requests.show(history.requests, `${stats.requests} requests`);
        this._busy.show(history.gpu_busy, `${Math.round(stats.gpu_busy ?? 0)} % GPU`);

        // Small amounts read better in Wh and cents than as 0.00 kWh and 0.00 €.
        const kwh = stats.energy_kwh, eur = stats.cost_eur;
        const energy = kwh < 1 ? `${(kwh * 1000).toFixed(kwh < 0.01 ? 1 : 0)} Wh` : `${kwh.toFixed(2)} kWh`;
        const cost = eur < 1 ? `${(eur * 100).toFixed(eur < 0.01 ? 2 : 1)} ct` : `${eur.toFixed(2)} €`;
        this._energy.label.text = stats.price_ct_per_kwh === null ? `${energy} since boot`
            : `${energy} · ${cost} since boot · now ${Math.round(stats.price_ct_per_kwh)} ct/kWh`;

        const memory = stats.all_memory;
        const hardware = [`${Math.round(memory.gpu_gb + memory.other_gb)} of ${Math.round(memory.total_gb)} GB`];
        if (stats.temperature_c)
            hardware.push(`${Math.round(stats.temperature_c)} °C`);
        this._hardware.label.text = hardware.join(' · ');
    }

    _show(model, state, detail) {
        model.dot.styleClass = `local-llm-dot ${state}`;
        model.item.detail.text = detail;
    }

    async _containers() {
        const out = await run(['podman', 'ps', '-a', '--filter', 'name=^llm-',
            '--format', '{{.Names}}\t{{.State}}\t{{.ExitCode}}\t{{.RunningFor}}']);
        const containers = new Map();
        for (const line of out.trim().split('\n')) {
            const [name, state, exitCode, since] = line.split('\t');
            if (since === undefined)
                continue;
            containers.set(name, {
                state, exitCode: Number(exitCode), since: since.replace(/ ago$/, ''),
            });
        }
        return containers;
    }

    // The last "...Error: ..." line is the reason, unless it is vLLM's generic
    // "Engine core initialization failed", which only points further up.
    async _lastError(name) {
        const log = await run(['podman', 'logs', '--tail', '300', `llm-${name}`])
            .catch(() => '');
        const errors = log.split('\n').filter(line =>
            /\w*Error: /.test(line) && !line.includes('Engine core initialization failed'));
        const line = errors.at(-1);
        if (!line)
            return `see ./llm logs ${name}`;
        const reason = line.slice(line.search(/\w*Error: /)).split('. ')[0];
        return reason.length > 90 ? `${reason.slice(0, 90)}…` : reason;
    }

    destroy() {
        if (this._timer)
            GLib.source_remove(this._timer);
        this._timer = null;
        super.destroy();
    }
});

export default class LocalLlmExtension extends Extension {
    enable() {
        this._status = new StatusButton(this);
        Main.panel.addToStatusArea(`${this.uuid}-status`, this._status);
    }

    disable() {
        this._status?.destroy();
        this._status = null;
    }
}
