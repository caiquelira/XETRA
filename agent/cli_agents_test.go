package agent

import (
	"context"
	"database/sql"
	"database/sql/driver"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Autumn-27/artex/db"
	"github.com/Autumn-27/artex/guard"
	"github.com/Autumn-27/artex/intercept"
	"github.com/Autumn-27/artex/llmrec"
	"github.com/Autumn-27/artex/mcpbridge"
	"github.com/Autumn-27/norma/harness"
	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/permission"
	actool "github.com/Autumn-27/norma/tool"
)

// --- Minimal in-memory sql/driver backing db.DB & db.ExplorationStore for CLI tests ---

type memNode struct {
	id             int64
	expID          int64
	kind           string
	payload        string
	priority       int
	state          string
	origin         string
	owner          string
	blockedReason  string
	deleteReason   string
	coldSinceRound *int64
	contentVersion int
	createdAt      time.Time
}

type memEdge struct {
	expID int64
	src   int64
	rel   string
	dst   int64
}

type memConstraint struct {
	id        int64
	expID     int64
	kind      string
	text      string
	origin    string
	createdAt time.Time
}

type memUsageRow struct {
	taskID      string
	expID       int64
	worker      string
	model       string
	profileName string
	latencyMs   int
	input       int
	output      int
	cacheRead   int
	cacheWrite  int
	status      string
}

type memPGState struct {
	mu           sync.Mutex
	nextID       int64
	explorations map[int64]struct {
		desc  string
		goal  string
		round int64
	}
	nodes       map[int64]*memNode
	edges       []memEdge
	constraints []memConstraint
	usages      []memUsageRow
}

func newMemExplorationDB(t *testing.T, expID int64, desc, goal string) (*db.DB, *db.ExplorationStore, *memPGState) {
	t.Helper()
	st := &memPGState{
		nextID: 100,
		explorations: map[int64]struct {
			desc  string
			goal  string
			round int64
		}{
			expID: {desc: desc, goal: goal},
		},
		nodes: make(map[int64]*memNode),
	}
	sqlDB := sql.OpenDB(&memPGConnector{st: st})
	t.Cleanup(func() { _ = sqlDB.Close() })
	pg := &db.DB{DB: sqlDB}
	return pg, pg.Exploration(expID), st
}

func (s *memPGState) addOriginFact(expID int64) int64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.nextID++
	id := s.nextID
	s.nodes[id] = &memNode{
		id:        id,
		expID:     expID,
		kind:      db.KindFact,
		payload:   `{"summary":"task origin"}`,
		state:     db.StateOrigin,
		origin:    "system",
		createdAt: time.Now().UTC(),
	}
	return id
}

func (s *memPGState) snapshotNodes(expID int64, kind string) []*memNode {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []*memNode
	for _, n := range s.nodes {
		if n.expID == expID && (kind == "" || n.kind == kind) {
			cp := *n
			out = append(out, &cp)
		}
	}
	slices.SortFunc(out, func(a, b *memNode) int {
		if a.id > b.id {
			return -1
		}
		if a.id < b.id {
			return 1
		}
		return 0
	})
	return out
}

func (s *memPGState) snapshotUsages() []memUsageRow {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]memUsageRow, len(s.usages))
	copy(out, s.usages)
	return out
}

type memPGConnector struct{ st *memPGState }

func (c *memPGConnector) Connect(context.Context) (driver.Conn, error) {
	return &memPGConn{st: c.st}, nil
}
func (c *memPGConnector) Driver() driver.Driver { return nil }

type memPGConn struct{ st *memPGState }

func (c *memPGConn) Prepare(string) (driver.Stmt, error) { return nil, nil }
func (c *memPGConn) Close() error                        { return nil }
func (c *memPGConn) Begin() (driver.Tx, error)           { return &memPGTx{}, nil }
func (c *memPGConn) BeginTx(context.Context, driver.TxOptions) (driver.Tx, error) {
	return &memPGTx{}, nil
}

type memPGTx struct{}

func (memPGTx) Commit() error   { return nil }
func (memPGTx) Rollback() error { return nil }

func (c *memPGConn) ExecContext(_ context.Context, query string, args []driver.NamedValue) (driver.Result, error) {
	q := strings.Join(strings.Fields(query), " ")
	s := c.st
	s.mu.Lock()
	defer s.mu.Unlock()

	switch {
	case strings.HasPrefix(q, "INSERT INTO exploration_anchors"):
		return driver.RowsAffected(1), nil

	case strings.HasPrefix(q, "INSERT INTO exploration_edges"):
		expID := asInt64(args[0].Value)
		src := asInt64(args[1].Value)
		rel := asString(args[2].Value)
		dst := asInt64(args[3].Value)
		for _, e := range s.edges {
			if e.expID == expID && e.src == src && e.rel == rel && e.dst == dst {
				return driver.RowsAffected(0), nil
			}
		}
		s.edges = append(s.edges, memEdge{expID: expID, src: src, rel: rel, dst: dst})
		return driver.RowsAffected(1), nil

	case strings.HasPrefix(q, "UPDATE exploration_nodes SET state=$1"):
		state := asString(args[0].Value)
		id := asInt64(args[1].Value)
		expID := asInt64(args[2].Value)
		if n := s.nodes[id]; n != nil && n.expID == expID {
			n.state = state
			n.contentVersion++
			return driver.RowsAffected(1), nil
		}
		return driver.RowsAffected(0), nil

	case strings.HasPrefix(q, "INSERT INTO llm_usage"):
		row := memUsageRow{
			taskID:      asString(args[0].Value),
			expID:       asInt64(args[1].Value),
			worker:      asString(args[2].Value),
			model:       asString(args[3].Value),
			profileName: asString(args[4].Value),
			latencyMs:   int(asInt64(args[5].Value)),
			input:       int(asInt64(args[6].Value)),
			output:      int(asInt64(args[7].Value)),
			cacheRead:   int(asInt64(args[8].Value)),
			cacheWrite:  int(asInt64(args[9].Value)),
			status:      asString(args[10].Value),
		}
		s.usages = append(s.usages, row)
		return driver.RowsAffected(1), nil
	}
	return driver.RowsAffected(0), nil
}

func (c *memPGConn) QueryContext(_ context.Context, query string, args []driver.NamedValue) (driver.Rows, error) {
	q := strings.Join(strings.Fields(query), " ")
	s := c.st
	s.mu.Lock()
	defer s.mu.Unlock()

	switch {
	case strings.HasPrefix(q, "INSERT INTO exploration_nodes("):
		s.nextID++
		id := s.nextID
		s.nodes[id] = &memNode{
			id:        id,
			expID:     asInt64(args[0].Value),
			kind:      asString(args[1].Value),
			payload:   asString(args[2].Value),
			priority:  int(asInt64(args[3].Value)),
			state:     asString(args[4].Value),
			origin:    asString(args[5].Value),
			createdAt: time.Now().UTC(),
		}
		return newMemRows([]string{"id"}, [][]driver.Value{{id}}), nil

	case strings.HasPrefix(q, "INSERT INTO task_constraints("):
		s.nextID++
		id := s.nextID
		s.constraints = append(s.constraints, memConstraint{
			id:        id,
			expID:     asInt64(args[0].Value),
			kind:      asString(args[1].Value),
			text:      asString(args[2].Value),
			origin:    asString(args[3].Value),
			createdAt: time.Now().UTC(),
		})
		return newMemRows([]string{"id"}, [][]driver.Value{{id}}), nil

	case strings.HasPrefix(q, "SELECT id FROM exploration_nodes WHERE exploration_id=$1 AND kind='fact' AND state='origin'"):
		expID := asInt64(args[0].Value)
		for id, n := range s.nodes {
			if n.expID == expID && n.kind == db.KindFact && n.state == db.StateOrigin {
				return newMemRows([]string{"id"}, [][]driver.Value{{id}}), nil
			}
		}
		return newMemRows([]string{"id"}, nil), nil

	case strings.Contains(q, "FROM exploration_nodes WHERE id=$1 AND exploration_id=$2"):
		id := asInt64(args[0].Value)
		expID := asInt64(args[1].Value)
		if n := s.nodes[id]; n != nil && n.expID == expID {
			return nodeRows([]*memNode{n}), nil
		}
		return nodeRows(nil), nil

	case strings.Contains(q, "FROM exploration_nodes WHERE exploration_id=$1 AND kind=$2 ORDER BY id DESC LIMIT $3"):
		expID := asInt64(args[0].Value)
		kind := asString(args[1].Value)
		limit := int(asInt64(args[2].Value))
		return nodeRows(collectNodes(s.nodes, expID, kind, "", limit, true)), nil

	case strings.Contains(q, "FROM exploration_nodes WHERE exploration_id=$1 AND kind=$2 AND ($3 <= 0 OR id < $3)"):
		expID := asInt64(args[0].Value)
		kind := asString(args[1].Value)
		before := asInt64(args[2].Value)
		var qFilter string
		var limit int
		if len(args) >= 5 {
			qFilter = strings.ToLower(strings.Trim(asString(args[3].Value), "%"))
			limit = int(asInt64(args[4].Value))
		} else {
			limit = int(asInt64(args[3].Value))
		}
		var matched []*memNode
		for _, n := range collectNodes(s.nodes, expID, kind, "", 0, true) {
			if before > 0 && n.id >= before {
				continue
			}
			if qFilter != "" && !strings.Contains(strings.ToLower(n.payload), qFilter) {
				continue
			}
			matched = append(matched, n)
			if limit > 0 && len(matched) >= limit {
				break
			}
		}
		return nodeRows(matched), nil

	case strings.Contains(q, "SELECT COUNT(*) FROM exploration_nodes WHERE exploration_id=$1 AND kind=$2"):
		expID := asInt64(args[0].Value)
		kind := asString(args[1].Value)
		var qFilter string
		if len(args) >= 3 {
			qFilter = strings.ToLower(strings.Trim(asString(args[2].Value), "%"))
		}
		cnt := int64(0)
		for _, n := range s.nodes {
			if n.expID == expID && n.kind == kind {
				if qFilter == "" || strings.Contains(strings.ToLower(n.payload), qFilter) {
					cnt++
				}
			}
		}
		return newMemRows([]string{"count"}, [][]driver.Value{{cnt}}), nil

	case strings.Contains(q, "FROM exploration_nodes WHERE exploration_id=$1 AND kind='intent' AND state='open' ORDER BY priority DESC"):
		expID := asInt64(args[0].Value)
		limit := int(asInt64(args[1].Value))
		list := collectNodes(s.nodes, expID, db.KindIntent, "open", 0, false)
		slices.SortFunc(list, func(a, b *memNode) int {
			if a.priority != b.priority {
				return b.priority - a.priority
			}
			return int(a.id - b.id)
		})
		if limit > 0 && len(list) > limit {
			list = list[:limit]
		}
		return nodeRows(list), nil

	case strings.Contains(q, "FROM exploration_nodes WHERE exploration_id=$1 AND kind=$2 AND state=$3 ORDER BY id"):
		expID := asInt64(args[0].Value)
		kind := asString(args[1].Value)
		state := asString(args[2].Value)
		return nodeRows(collectNodes(s.nodes, expID, kind, state, 0, false)), nil

	case strings.Contains(q, "SELECT COUNT(*) FROM exploration_nodes WHERE exploration_id=$1 AND kind='intent' AND state IN ('done','blocked','exhausted')"):
		expID := asInt64(args[0].Value)
		cnt := int64(0)
		for _, n := range s.nodes {
			if n.expID == expID && n.kind == db.KindIntent && (n.state == "done" || n.state == "blocked" || n.state == "exhausted") {
				cnt++
			}
		}
		return newMemRows([]string{"count"}, [][]driver.Value{{cnt}}), nil

	case strings.Contains(q, "SELECT COUNT(*) FROM exploration_nodes WHERE exploration_id=$1 AND kind='intent' AND state='open'"):
		expID := asInt64(args[0].Value)
		cnt := int64(0)
		for _, n := range s.nodes {
			if n.expID == expID && n.kind == db.KindIntent && n.state == "open" {
				cnt++
			}
		}
		return newMemRows([]string{"count"}, [][]driver.Value{{cnt}}), nil

	case strings.Contains(q, "SELECT EXISTS(SELECT 1 FROM exploration_nodes WHERE exploration_id=$1 AND kind='intent' AND state IN ('open','running'))"):
		expID := asInt64(args[0].Value)
		exists := false
		for _, n := range s.nodes {
			if n.expID == expID && n.kind == db.KindIntent && (n.state == "open" || n.state == "running") {
				exists = true
				break
			}
		}
		return newMemRows([]string{"exists"}, [][]driver.Value{{exists}}), nil

	case strings.HasPrefix(q, "SELECT src_id, rel, dst_id FROM exploration_edges WHERE exploration_id=$1"):
		expID := asInt64(args[0].Value)
		limit := int(asInt64(args[1].Value))
		var data [][]driver.Value
		for _, e := range s.edges {
			if e.expID == expID {
				data = append(data, []driver.Value{e.src, e.rel, e.dst})
				if limit > 0 && len(data) >= limit {
					break
				}
			}
		}
		return newMemRows([]string{"src_id", "rel", "dst_id"}, data), nil

	case strings.HasPrefix(q, "SELECT description, goal FROM explorations WHERE id=$1"):
		expID := asInt64(args[0].Value)
		if exp, ok := s.explorations[expID]; ok {
			return newMemRows([]string{"description", "goal"}, [][]driver.Value{{exp.desc, exp.goal}}), nil
		}
		return newMemRows([]string{"description", "goal"}, nil), nil

	case strings.HasPrefix(q, "UPDATE explorations SET round_no = round_no + 1 WHERE id=$1 RETURNING round_no"):
		expID := asInt64(args[0].Value)
		exp := s.explorations[expID]
		exp.round++
		s.explorations[expID] = exp
		return newMemRows([]string{"round_no"}, [][]driver.Value{{exp.round}}), nil

	case strings.Contains(q, "FROM task_constraints WHERE exploration_id=$1"):
		expID := asInt64(args[0].Value)
		var data [][]driver.Value
		for _, ct := range s.constraints {
			if ct.expID == expID {
				data = append(data, []driver.Value{ct.id, ct.kind, ct.text, ct.origin, ct.createdAt})
			}
		}
		return newMemRows([]string{"id", "kind", "text", "origin", "created_at"}, data), nil

	case strings.Contains(q, "SELECT id, cold_since_round FROM exploration_nodes"):
		return newMemRows([]string{"id", "cold_since_round"}, nil), nil

	case strings.Contains(q, "SELECT e.dst_id, d.id FROM exploration_edges"):
		return newMemRows([]string{"dst_id", "id"}, nil), nil

	case strings.Contains(q, "FROM tasks owner"):
		return newMemRows([]string{"id", "exploration_id", "description", "goal", "status"}, nil), nil

	case strings.Contains(q, "FROM task_credentials"):
		return newMemRows([]string{"id", "task_id", "domain", "username", "password", "source", "note", "created_at", "updated_at"}, nil), nil

	case strings.Contains(q, "SELECT id FROM tasks WHERE exploration_id=$1"):
		return newMemRows([]string{"id"}, nil), nil

	case strings.Contains(q, "SELECT e.dst_id, e.src_id FROM exploration_edges"):
		return newMemRows([]string{"dst_id", "src_id"}, nil), nil

	case strings.Contains(q, "FROM findings f WHERE f.node_id IN"):
		return newMemRows([]string{"id", "node_id", "count"}, nil), nil
	}
	return nil, fmt.Errorf("unexpected test SQL query: %s", q)
}

func collectNodes(nodes map[int64]*memNode, expID int64, kind, state string, limit int, desc bool) []*memNode {
	var out []*memNode
	for _, n := range nodes {
		if n.expID != expID {
			continue
		}
		if kind != "" && n.kind != kind {
			continue
		}
		if state != "" && n.state != state {
			continue
		}
		out = append(out, n)
	}
	slices.SortFunc(out, func(a, b *memNode) int {
		if desc {
			return int(b.id - a.id)
		}
		return int(a.id - b.id)
	})
	if limit > 0 && len(out) > limit {
		out = out[:limit]
	}
	return out
}

func nodeRows(nodes []*memNode) driver.Rows {
	cols := []string{"id", "kind", "payload", "priority", "state", "origin", "owner", "blocked_reason", "delete_reason", "created_at"}
	data := make([][]driver.Value, 0, len(nodes))
	for _, n := range nodes {
		data = append(data, []driver.Value{
			n.id, n.kind, []byte(n.payload), int64(n.priority), n.state,
			n.origin, n.owner, n.blockedReason, n.deleteReason, n.createdAt,
		})
	}
	return newMemRows(cols, data)
}

type memRows struct {
	cols []string
	data [][]driver.Value
	idx  int
}

func newMemRows(cols []string, data [][]driver.Value) driver.Rows {
	return &memRows{cols: cols, data: data}
}

func (r *memRows) Columns() []string { return r.cols }
func (r *memRows) Close() error      { return nil }
func (r *memRows) Next(dest []driver.Value) error {
	if r.idx >= len(r.data) {
		return io.EOF
	}
	copy(dest, r.data[r.idx])
	r.idx++
	return nil
}

func asInt64(v any) int64 {
	switch x := v.(type) {
	case int64:
		return x
	case int:
		return int64(x)
	case nil:
		return 0
	default:
		n, _ := strconv.ParseInt(fmt.Sprint(x), 10, 64)
		return n
	}
}

func asString(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case []byte:
		return string(x)
	case nil:
		return ""
	default:
		return fmt.Sprint(x)
	}
}

type fakeFindingRecorder struct {
	mu      sync.Mutex
	records []db.RecordFindingInput
	nextID  int64
}

func (r *fakeFindingRecorder) Record(_ context.Context, in db.RecordFindingInput, _ []db.TrafficRef) (*db.RecordedFinding, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.nextID++
	r.records = append(r.records, in)
	return &db.RecordedFinding{
		FindingID: r.nextID,
		NodeID:    500 + r.nextID,
		Traffic:   &db.FindingTraffic{},
	}, nil
}

type wrappedCLIProv struct {
	inner llm.Provider
}

func (w wrappedCLIProv) Stream(ctx context.Context, req llm.CompletionRequest) iter.Seq2[llm.StreamEvent, error] {
	return w.inner.Stream(ctx, req)
}

func (w wrappedCLIProv) Complete(ctx context.Context, req llm.CompletionRequest) (llm.Message, string, llm.Usage, error) {
	return w.inner.Complete(ctx, req)
}

func (w wrappedCLIProv) Unwrap() llm.Provider { return w.inner }

type dynamicCLIResolver struct {
	cfg Config
}

func (d dynamicCLIResolver) Stream(context.Context, llm.CompletionRequest) iter.Seq2[llm.StreamEvent, error] {
	return nil
}

func (d dynamicCLIResolver) Complete(context.Context, llm.CompletionRequest) (llm.Message, string, llm.Usage, error) {
	return llm.Message{}, "", llm.Usage{}, nil
}

func (d dynamicCLIResolver) ResolveCLIConfig(context.Context) (Config, bool) {
	return d.cfg, d.cfg.IsCLI()
}

func TestResolveCLIAgentOptions(t *testing.T) {
	bin := buildFakeCLIBinary(t)
	cliCfg := Config{Format: FormatClaudeCLI, BaseURL: bin, Model: "sonnet"}
	prov, err := cliCfg.NewProvider()
	if err != nil {
		t.Fatal(err)
	}

	// Explicit CLI Config wins.
	agyCfg := Config{Format: FormatAgyCLI, BaseURL: bin, Model: "gemini-4-hydrogen-high"}
	got, ok := resolveCLIAgentOptions(&CLIAgentOptions{Config: agyCfg, HookBin: bin}, prov)
	if !ok || got.Config.Format != FormatAgyCLI || got.HookBin != bin {
		t.Fatalf("explicit CLI override = (%+v, %v)", got, ok)
	}

	// Explicit non-CLI Config forces Norma path even if provider wraps CLI.
	if got, ok := resolveCLIAgentOptions(&CLIAgentOptions{Config: Config{Format: llm.FormatAnthropic}}, prov); ok {
		t.Fatalf("expected non-CLI explicit config to disable CLI mode, got %+v", got)
	}

	// Direct *cliCompletionProvider resolves cleanly and merges Bridge/UsagePG/HookBin when explicit Config is zero.
	srv, err := mcpbridge.NewServer()
	if err != nil {
		t.Fatal(err)
	}
	defer srv.Close()
	got, ok = resolveCLIAgentOptions(&CLIAgentOptions{Bridge: srv, HookBin: bin}, prov)
	if !ok || got.Config.Format != FormatClaudeCLI || got.Bridge != srv || got.HookBin != bin {
		t.Fatalf("merged provider resolution = (%+v, %v)", got, ok)
	}

	// Unwrap() and ResolveCLIConfig() providers also resolve cleanly.
	if got, ok := resolveCLIAgentOptions(nil, wrappedCLIProv{inner: prov}); !ok || got.Config.Format != FormatClaudeCLI {
		t.Fatalf("wrapped provider resolution = (%+v, %v)", got, ok)
	}
	if got, ok := resolveCLIAgentOptions(nil, dynamicCLIResolver{cfg: agyCfg}); !ok || got.Config.Format != FormatAgyCLI {
		t.Fatalf("dynamic resolver resolution = (%+v, %v)", got, ok)
	}

	// Regular non-CLI provider returns false.
	if _, ok := resolveCLIAgentOptions(nil, &countingProvider{}); ok {
		t.Fatal("expected non-CLI provider to return ok=false")
	}
}

func TestWorkerCLI(t *testing.T) {
	bin := buildFakeCLIBinary(t)

	for _, format := range []llm.Format{FormatClaudeCLI, FormatAgyCLI} {
		t.Run(string(format), func(t *testing.T) {
			workDir := t.TempDir()
			const taskID int64 = 7
			const expID int64 = 12
			pg, ts, st := newMemExplorationDB(t, expID, "test worker CLI", "find XSS")
			st.addOriginFact(expID)
			intentPayload, _ := json.Marshal(map[string]any{"summary": "test /search XSS"})
			intentID, err := ts.AddNode(db.KindIntent, map[string]any{"summary": "test /search XSS"}, 8, "running", "planner", nil)
			if err != nil {
				t.Fatal(err)
			}
			intentNode := &db.Node{ID: intentID, Kind: db.KindIntent, Payload: intentPayload}

			cfg := Config{
				Format:      format,
				BaseURL:     bin,
				Model:       "model-agent-worker",
				ProfileName: "prof-worker-" + string(format),
			}
			prov, err := cfg.NewProvider()
			if err != nil {
				t.Fatalf("NewProvider: %v", err)
			}

			var extraCalls atomic.Int32
			extraTool := actool.Build(actool.Spec{
				Name:        "extra_probe",
				Description: "Extra worker tool",
				Schema:      map[string]any{"type": "object"},
				Permissions: func(context.Context, json.RawMessage, permission.Context) permission.Decision {
					return permission.Allowed()
				},
				Run: func(context.Context, json.RawMessage, *actool.ToolContext) (actool.Result, error) {
					extraCalls.Add(1)
					return actool.Text("extra-ok"), nil
				},
			})

			rec := &fakeFindingRecorder{}
			hooks := &testHookRunner{
				blockSubstr: "rm -rf /blocked",
				blockReason: "guard denied dangerous command",
			}

			w := NewWorker(prov, cfg.Model, workDir, nil, 100000, 10, extraTool)
			w.SetFindingRecorder(rec)
			w.SetProxy("http://127.0.0.1:8788", "")
			w.SetCLIOptions(CLIAgentOptions{
				UsagePG: pg,
				HookBin: bin,
			})

			var notified []string
			var acts []db.Activity
			ctx := llmrec.WithTaskID(context.Background(), strconv.FormatInt(taskID, 10))
			reason, counts, err := w.Execute(ctx, "work#1", taskID, nil, ts, intentNode, hooks, func(a db.Activity) {
				acts = append(acts, a)
			}, nil, func(iid int64, sum string) {
				notified = append(notified, fmt.Sprintf("%d:%s", iid, sum))
			})
			if err != nil {
				t.Fatalf("Worker.Execute failed: %v", err)
			}
			if reason != harness.ReasonCompleted {
				t.Fatalf("reason = %q, want completed", reason)
			}
			if counts.Facts != 1 || counts.Findings != 1 {
				t.Fatalf("WriteCounts = %+v, want Facts=1 Findings=1", counts)
			}
			if extraCalls.Load() != 1 {
				t.Fatalf("extraCalls = %d, want 1", extraCalls.Load())
			}
			if len(notified) != 1 || !strings.Contains(notified[0], "verified XSS on /search") {
				t.Fatalf("notifyFinding = %v", notified)
			}
			if len(acts) == 0 || acts[len(acts)-1].Worker != "work#1" || acts[len(acts)-1].NodeID == nil || *acts[len(acts)-1].NodeID != intentID {
				t.Fatalf("unexpected worker activities: %+v", acts)
			}

			// Verify guard hooks intercepted native Bash/run_command.
			var sawBlocked, sawAllowed bool
			for _, c := range hooks.snapshot() {
				if c.Kind == "pre" && c.Tool == "Bash" && strings.Contains(c.Input, "rm -rf /blocked") {
					sawBlocked = true
				}
				if c.Kind == "post" && c.Tool == "Bash" && strings.Contains(c.Input, "echo allowed") {
					sawAllowed = true
				}
			}
			if !sawBlocked || !sawAllowed {
				t.Fatalf("missing expected hook calls: blocked=%v allowed=%v calls=%+v", sawBlocked, sawAllowed, hooks.snapshot())
			}

			// Verify CLI session persisted under <workDir>/cli-sessions/exp12-worker-i<intentID>.json.
			sessKey := WorkerSessionID(expID, intentID)
			sessStore := NewCLISessionStore(filepath.Join(workDir, "cli-sessions"))
			srec, ok, err := sessStore.Load(sessKey)
			if err != nil || !ok || srec.SessionID == "" {
				t.Fatalf("worker CLI session not saved: rec=%+v ok=%v err=%v", srec, ok, err)
			}

			// Verify isolated workDir exists and parent task dir has no .agents dir.
			if _, err := os.Stat(filepath.Join(workDir, "tasks", strconv.FormatInt(taskID, 10), "i"+strconv.FormatInt(intentID, 10))); err != nil {
				t.Fatalf("expected isolated intent runDir: %v", err)
			}
			if _, err := os.Stat(filepath.Join(workDir, "tasks", strconv.FormatInt(taskID, 10), ".agents")); err == nil {
				t.Fatal("parent task dir unexpectedly contains .agents")
			}

			// Run ExecuteWithMessage to verify session resume and human steer prompt propagation.
			reason2, counts2, err := w.ExecuteWithMessage(ctx, "work#1", taskID, nil, ts, intentNode, hooks, nil, nil, nil, "req-42", "focus on steer-target param")
			if err != nil {
				t.Fatalf("Worker.ExecuteWithMessage failed: %v", err)
			}
			if reason2 != harness.ReasonCompleted || counts2.Facts != 1 {
				t.Fatalf("ExecuteWithMessage = (reason=%q counts=%+v), want completed + 1 fact", reason2, counts2)
			}

			facts := st.snapshotNodes(expID, db.KindFact)
			if len(facts) < 3 { // origin + initial + steered
				t.Fatalf("expected >=3 fact nodes in store, got %d", len(facts))
			}

			usages := st.snapshotUsages()
			if len(usages) != 2 || usages[0].worker != "worker" || usages[1].worker != "worker" {
				t.Fatalf("unexpected llm_usage rows: %+v", usages)
			}
		})
	}
}

func TestPlannerCLI(t *testing.T) {
	bin := buildFakeCLIBinary(t)

	for _, format := range []llm.Format{FormatClaudeCLI, FormatAgyCLI} {
		t.Run(string(format), func(t *testing.T) {
			workDir := t.TempDir()
			const taskID int64 = 9
			const expID int64 = 21
			pg, ts, st := newMemExplorationDB(t, expID, "test planner CLI", "prove auth bypass")
			st.addOriginFact(expID)
			goalID, err := ts.AddGoal(map[string]any{"text": "Prove auth bypass"}, "goals")
			if err != nil {
				t.Fatal(err)
			}

			cfg := Config{
				Format:      format,
				BaseURL:     bin,
				Model:       "model-agent-planner",
				ProfileName: "prof-planner-" + string(format),
			}
			prov, err := cfg.NewProvider()
			if err != nil {
				t.Fatal(err)
			}

			hooks := &testHookRunner{
				blockSubstr: "rm -rf /blocked",
				blockReason: "blocked",
			}

			p := NewPlanner(prov, cfg.Model, workDir, nil, 100000, 10)
			p.SetCLIOptions(CLIAgentOptions{
				UsagePG: pg,
				HookBin: bin,
			})

			ctx := llmrec.WithTaskID(context.Background(), strconv.FormatInt(taskID, 10))
			var acts1 []db.Activity
			met1, _, err := p.Plan(ctx, taskID, nil, ts, "prove auth bypass", nil, hooks, func(a db.Activity) {
				acts1 = append(acts1, a)
			})
			if err != nil {
				t.Fatalf("Plan round 1 failed: %v", err)
			}
			if met1 {
				t.Fatal("expected met=false after round 1")
			}
			if len(acts1) == 0 || acts1[len(acts1)-1].Worker != "planner" {
				t.Fatalf("unexpected planner round 1 activities: %+v", acts1)
			}

			// Verify TodoStore persisted round 1 items and add_intent wrote an open intent.
			todos1 := p.todoFor(expID).List()
			if len(todos1) != 2 || todos1[0].Content != "step 1: recon" || todos1[0].Status != actool.TodoInProgress {
				t.Fatalf("round 1 todos mismatch: %+v", todos1)
			}
			intents := st.snapshotNodes(expID, db.KindIntent)
			if len(intents) != 1 {
				t.Fatalf("expected 1 intent after round 1, got %d", len(intents))
			}

			// Record an observed tool_output fact so round 2 can call prove_goal.
			factID, err := ts.AddNode(db.KindFact, map[string]any{
				"summary":    "auth bypass confirmed",
				"confidence": "observed",
				"source":     "tool_output",
			}, 5, "confirmed", "worker", nil)
			if err != nil {
				t.Fatal(err)
			}

			met2, reason2, err := p.Plan(ctx, taskID, nil, ts, "prove auth bypass", []TriggerEvent{{Kind: "done", IntentID: intents[0].id}}, hooks, nil)
			if err != nil {
				t.Fatalf("Plan round 2 failed: %v", err)
			}
			if !met2 || !strings.Contains(reason2, "met") {
				t.Fatalf("Plan round 2 = (met=%v reason=%q), want met=true", met2, reason2)
			}

			todos2 := p.todoFor(expID).List()
			if len(todos2) != 2 || todos2[0].Status != actool.TodoCompleted {
				t.Fatalf("round 2 todos not updated: %+v", todos2)
			}
			goalNode, err := ts.GetNode(goalID)
			if err != nil || goalNode == nil || goalNode.State != "met" {
				t.Fatalf("goal %d state = %+v (err=%v), want met (evidence=%d)", goalID, goalNode, err, factID)
			}

			// Verify Planner stays round-stateless (no CLI session saved) and isolated in .../planner.
			if _, err := os.Stat(filepath.Join(workDir, "tasks", strconv.FormatInt(taskID, 10), "planner")); err != nil {
				t.Fatalf("expected isolated planner dir: %v", err)
			}
			if _, err := os.Stat(filepath.Join(workDir, "cli-sessions")); err == nil {
				entries, _ := os.ReadDir(filepath.Join(workDir, "cli-sessions"))
				if len(entries) > 0 {
					t.Fatalf("planner should be stateless, found session files: %v", entries)
				}
			}

			usages := st.snapshotUsages()
			if len(usages) != 2 || usages[0].worker != "planner" || usages[1].worker != "planner" {
				t.Fatalf("unexpected planner llm_usage rows: %+v", usages)
			}
		})
	}
}

func TestGoalsCLI(t *testing.T) {
	bin := buildFakeCLIBinary(t)

	for _, format := range []llm.Format{FormatClaudeCLI, FormatAgyCLI} {
		t.Run(string(format), func(t *testing.T) {
			dataDir := t.TempDir()
			const taskID int64 = 15
			const expID int64 = 33
			pg, ts, st := newMemExplorationDB(t, expID, "auth + api test", "test /admin and /api/users")
			st.addOriginFact(expID)

			cfg := Config{
				Format:      format,
				BaseURL:     bin,
				Model:       "model-agent-goals",
				ProfileName: "prof-goals-" + string(format),
			}
			prov, err := cfg.NewProvider()
			if err != nil {
				t.Fatal(err)
			}

			var acts []db.Activity
			ctx := llmrec.WithTaskID(context.Background(), strconv.FormatInt(taskID, 10))
			// First verify DecomposeGoalsWithProvider delegates seamlessly for *cliCompletionProvider.
			specs := DecomposeGoalsWithProvider(ctx, prov, dataDir, "test /admin and /api/users", "passive recon only", nil, ts, taskID, false, 0, func(a db.Activity) {
				acts = append(acts, a)
			})
			if len(specs) != 2 {
				t.Fatalf("DecomposeGoalsWithProvider specs = %+v, want 2 goals", specs)
			}
			texts := []string{specs[0].Text, specs[1].Text}
			if !slices.Contains(texts, "Prove authentication bypass on /admin") || !slices.Contains(texts, "Verify IDOR on /api/users") {
				t.Fatalf("unexpected decomposed goals: %+v", specs)
			}
			if len(acts) == 0 || acts[len(acts)-1].Worker != "planner" {
				t.Fatalf("expected goals activity to tag Worker='planner', got %+v", acts)
			}
			constraints, err := ts.ListConstraints()
			if err != nil || len(constraints) != 1 || constraints[0].Kind != "allow" {
				t.Fatalf("constraints = %+v (err=%v), want 1 allow constraint", constraints, err)
			}
			if _, err := os.Stat(filepath.Join(dataDir, "tasks", strconv.FormatInt(taskID, 10), "goals")); err != nil {
				t.Fatalf("expected isolated goals dir: %v", err)
			}

			// Now verify DecomposeGoalsWithCLI with UsagePG and empty dataDir (ephemeral temp dir cleaned up).
			_, ts2, st2 := newMemExplorationDB(t, expID+1, "desc", "goal")
			specs2 := DecomposeGoalsWithCLI(ctx, CLIAgentOptions{Config: cfg, UsagePG: pg}, "", "test /admin", "", nil, ts2, 0, nil)
			if len(specs2) != 2 {
				t.Fatalf("DecomposeGoalsWithCLI (ephemeral dir) specs = %+v, want 2", specs2)
			}
			_ = st2
			usages := st.snapshotUsages()
			if len(usages) != 1 || usages[0].worker != "goals" {
				t.Fatalf("expected 1 goals llm_usage row, got %+v", usages)
			}
		})
	}
}

func TestMainAgentCLI(t *testing.T) {
	bin := buildFakeCLIBinary(t)

	for _, format := range []llm.Format{FormatClaudeCLI, FormatAgyCLI} {
		t.Run(string(format), func(t *testing.T) {
			workDir := t.TempDir()
			const taskID int64 = 18
			const expID int64 = 44
			pg, ts, st := newMemExplorationDB(t, expID, "mainagent test", "goal")
			st.addOriginFact(expID)

			cfg := Config{
				Format:      format,
				BaseURL:     bin,
				Model:       "model-agent-mainagent",
				ProfileName: "prof-main-" + string(format),
			}
			prov, err := cfg.NewProvider()
			if err != nil {
				t.Fatal(err)
			}
			hooks := &testHookRunner{blockSubstr: "rm -rf /blocked", blockReason: "blocked"}

			m := NewMainAgent(prov, cfg.Model, workDir, nil, 100000, 10)
			m.SetCLIOptions(CLIAgentOptions{UsagePG: pg, HookBin: bin})

			var hintNotices [][]string
			var acts []db.Activity
			ctx := llmrec.WithTaskID(context.Background(), strconv.FormatInt(taskID, 10))
			reply1, err := m.Chat(ctx, taskID, 0, nil, ts, "goal", "focus on auth", hooks, func(a db.Activity) {
				acts = append(acts, a)
			}, nil, nil, nil, func(_ string, hints []string) {
				hintNotices = append(hintNotices, hints)
			})
			if err != nil {
				t.Fatalf("MainAgent.Chat turn 1 failed: %v", err)
			}
			if !strings.HasPrefix(reply1, "main fresh:") {
				t.Fatalf("turn 1 reply = %q, want main fresh:<id>", reply1)
			}
			seg0ID := strings.TrimPrefix(reply1, "main fresh:")
			if len(hintNotices) != 1 || len(acts) == 0 || acts[len(acts)-1].Worker != "mainagent" {
				t.Fatalf("unexpected hintNotices=%v acts=%+v", hintNotices, acts)
			}

			// Turn 2 on same segment 0 must resume seg0ID.
			reply2, err := m.Chat(ctx, taskID, 0, nil, ts, "goal", "dispatch test", hooks, nil, nil, nil, nil, nil)
			if err != nil {
				t.Fatalf("MainAgent.Chat turn 2 failed: %v", err)
			}
			if reply2 != "main resumed:"+seg0ID {
				t.Fatalf("turn 2 reply = %q, want main resumed:%s", reply2, seg0ID)
			}
			if intents := st.snapshotNodes(expID, db.KindIntent); len(intents) != 1 {
				t.Fatalf("expected 1 intent added on turn 2, got %d", len(intents))
			}

			// Turn 3 on segment 1 must rotate cleanly to a fresh session exp44-main-s1.
			reply3, err := m.Chat(ctx, taskID, 1, nil, ts, "goal", "seg1-fresh message", hooks, nil, nil, nil, nil, nil)
			if err != nil {
				t.Fatalf("MainAgent.Chat turn 3 (seg 1) failed: %v", err)
			}
			if !strings.HasPrefix(reply3, "main fresh:") {
				t.Fatalf("turn 3 (seg 1) reply = %q, want fresh session", reply3)
			}
			if strings.TrimPrefix(reply3, "main fresh:") == seg0ID {
				t.Fatalf("segment 1 reused segment 0 session id %q", seg0ID)
			}

			// Verify both segment session files exist.
			store := NewCLISessionStore(filepath.Join(workDir, "cli-sessions"))
			if _, ok, err := store.Load(fmt.Sprintf("exp%d-main", expID)); err != nil || !ok {
				t.Fatalf("missing segment 0 session file: ok=%v err=%v", ok, err)
			}
			if _, ok, err := store.Load(fmt.Sprintf("exp%d-main-s1", expID)); err != nil || !ok {
				t.Fatalf("missing segment 1 session file: ok=%v err=%v", ok, err)
			}
			if _, err := os.Stat(filepath.Join(workDir, "tasks", strconv.FormatInt(taskID, 10), "mainagent")); err != nil {
				t.Fatalf("expected isolated mainagent dir: %v", err)
			}

			usages := st.snapshotUsages()
			if len(usages) != 3 || usages[0].worker != "main" {
				t.Fatalf("unexpected mainagent llm_usage rows: %+v", usages)
			}
		})
	}
}

func TestChatCLI(t *testing.T) {
	bin := buildFakeCLIBinary(t)

	prevAugment := ToolAugment()
	t.Cleanup(func() { SetToolAugment(prevAugment) })

	var retesterCalls atomic.Int32
	SetToolAugment(func(_ context.Context, agentKey string) ([]actool.CoreTool, DeferredInfo, func()) {
		if agentKey != "finding_retester" {
			return nil, DeferredInfo{}, nil
		}
		probe := actool.Build(actool.Spec{
			Name:        "retester_probe",
			Description: "Finding retester MCP probe",
			Schema:      map[string]any{"type": "object"},
			Permissions: func(context.Context, json.RawMessage, permission.Context) permission.Decision {
				return permission.Allowed()
			},
			Run: func(context.Context, json.RawMessage, *actool.ToolContext) (actool.Result, error) {
				retesterCalls.Add(1)
				return actool.Text("retester-probe-ok"), nil
			},
		})
		return []actool.CoreTool{probe}, DeferredInfo{}, nil
	})

	for _, format := range []llm.Format{FormatClaudeCLI, FormatAgyCLI} {
		t.Run(string(format), func(t *testing.T) {
			retesterCalls.Store(0)
			workDir := t.TempDir()
			pg, _, st := newMemExplorationDB(t, 1, "chat", "chat")

			cfg := Config{
				Format:      format,
				BaseURL:     bin,
				Model:       "model-agent-chat",
				ProfileName: "prof-chat-" + string(format),
			}
			prov, err := cfg.NewProvider()
			if err != nil {
				t.Fatal(err)
			}

			ic := intercept.New(nil)
			ic.SetRulesForTest([]db.InterceptRule{
				{
					ID:          1,
					Name:        "deny /blocked in CLI test",
					Enabled:     true,
					Priority:    100,
					MatchTarget: "tool_input",
					MatchType:   "string",
					Action:      "deny",
					Pattern:     "/blocked",
					Message:     "guard denied dangerous command",
				},
				{
					ID:          2,
					Name:        "allow echo in CLI test",
					Enabled:     true,
					Priority:    50,
					MatchTarget: "tool_input",
					MatchType:   "string",
					Action:      "allow",
					Pattern:     "echo allowed",
					Message:     "allowed",
				},
			})
			g := guard.NewWithInterceptor(ic)
			c := NewChatAgent(prov, cfg.Model, workDir, nil, 100000)
			c.SetGuard(g)
			c.SetCLIOptions(CLIAgentOptions{UsagePG: pg, HookBin: bin})

			const sessionID = "conv-retest-99"
			var acts []db.Activity
			ctx := llmrec.WithWorker(context.Background(), "finding_retester")
			reply1, err := c.Chat(ctx, "finding_retester", sessionID, "retest finding #42", 8, 0, false, func(a db.Activity) {
				acts = append(acts, a)
			})
			if err != nil {
				t.Fatalf("Chat turn 1 failed: %v", err)
			}
			if !strings.HasPrefix(reply1, "chat fresh:") {
				t.Fatalf("turn 1 reply = %q, want chat fresh:<id>", reply1)
			}
			sid := strings.TrimPrefix(reply1, "chat fresh:")
			if retesterCalls.Load() != 1 {
				t.Fatalf("retesterCalls = %d, want 1", retesterCalls.Load())
			}
			if len(acts) == 0 || acts[len(acts)-1].Worker != "finding_retester" {
				t.Fatalf("unexpected chat activities: %+v", acts)
			}
			if _, err := os.Stat(filepath.Join(workDir, "sessions", sessionID)); err != nil {
				t.Fatalf("expected isolated sessionWorkDir: %v", err)
			}
			if len(g.Audit()) == 0 {
				t.Fatal("expected Guard audit entries from chat CLI hooks")
			}

			// Turn 2 resumes the same session and checks default "chat" worker label when WithWorker is absent.
			reply2, err := c.Chat(context.Background(), "finding_retester", sessionID, "follow up", 8, 0, false, nil)
			if err != nil {
				t.Fatalf("Chat turn 2 failed: %v", err)
			}
			if reply2 != "chat resumed:"+sid {
				t.Fatalf("turn 2 reply = %q, want chat resumed:%s", reply2, sid)
			}

			usages := st.snapshotUsages()
			if len(usages) != 2 || usages[0].worker != "finding_retester" || usages[1].worker != "chat" {
				t.Fatalf("unexpected chat llm_usage worker labels: %+v", usages)
			}
		})
	}
}

func TestCLIIsolatedAgentDirsConcurrent(t *testing.T) {
	bin := buildFakeCLIBinary(t)
	workDir := t.TempDir()
	const taskID int64 = 77
	const expID int64 = 88
	_, ts, st := newMemExplorationDB(t, expID, "concurrent isolation", "isolate .agents")
	st.addOriginFact(expID)
	intentPayload, _ := json.Marshal(map[string]any{"summary": "concurrent worker"})
	intentID, err := ts.AddNode(db.KindIntent, map[string]any{"summary": "concurrent worker"}, 5, "running", "planner", nil)
	if err != nil {
		t.Fatal(err)
	}
	intentNode := &db.Node{ID: intentID, Kind: db.KindIntent, Payload: intentPayload}

	srv, err := mcpbridge.NewServer()
	if err != nil {
		t.Fatal(err)
	}
	defer srv.Close()

	cfg := Config{
		Format:  FormatAgyCLI,
		BaseURL: bin,
		Model:   "model-agent-concurrent-agy",
	}
	prov, err := cfg.NewProvider()
	if err != nil {
		t.Fatal(err)
	}
	cliOpts := CLIAgentOptions{Config: cfg, Bridge: srv, HookBin: bin}

	w := NewWorker(prov, cfg.Model, workDir, nil, 100000, 4)
	w.SetCLIOptions(cliOpts)
	p := NewPlanner(prov, cfg.Model, workDir, nil, 100000, 4)
	p.SetCLIOptions(cliOpts)
	m := NewMainAgent(prov, cfg.Model, workDir, nil, 100000, 4)
	m.SetCLIOptions(cliOpts)

	var wg sync.WaitGroup
	errs := make(chan error, 3)

	wg.Add(3)
	go func() {
		defer wg.Done()
		_, _, err := w.Execute(context.Background(), "work#1", taskID, nil, ts, intentNode, nil, nil, nil, nil)
		errs <- err
	}()
	go func() {
		defer wg.Done()
		_, _, err := p.Plan(context.Background(), taskID, nil, ts, "isolate .agents", nil, nil, nil)
		errs <- err
	}()
	go func() {
		defer wg.Done()
		reply, err := m.Chat(context.Background(), taskID, 0, nil, ts, "isolate .agents", "status?", nil, nil, nil, nil, nil, nil)
		if err == nil && reply != "concurrent-ok:mainagent" {
			err = fmt.Errorf("unexpected mainagent reply %q", reply)
		}
		errs <- err
	}()
	wg.Wait()
	close(errs)

	for err := range errs {
		if err != nil {
			t.Fatalf("concurrent agy-cli isolation failed: %v", err)
		}
	}

	taskDir := filepath.Join(workDir, "tasks", strconv.FormatInt(taskID, 10))
	if _, err := os.Stat(filepath.Join(taskDir, ".agents")); err == nil {
		t.Fatalf("shared taskDir %q must never contain .agents", taskDir)
	}
}
