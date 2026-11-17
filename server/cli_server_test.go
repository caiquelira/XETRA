package server

import (
	"bytes"
	"context"
	"database/sql"
	"database/sql/driver"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Autumn-27/artex/agent"
	"github.com/Autumn-27/artex/db"
	"github.com/Autumn-27/artex/guard"
	"github.com/Autumn-27/artex/intercept"
	"github.com/Autumn-27/artex/llmpool"
	"github.com/Autumn-27/artex/llmrec"
	"github.com/Autumn-27/artex/mcpbridge"
	"github.com/Autumn-27/artex/sidequestion"
	"github.com/Autumn-27/norma/harness"
	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/permission"
	actool "github.com/Autumn-27/norma/tool"
	"github.com/Autumn-27/norma/transcript"
)

var (
	serverFakeCLIOnce sync.Once
	serverFakeCLIPath string
	serverFakeCLIErr  error

	serverAgyModelsOnce sync.Once
	serverAgyModelsPath string
	serverAgyModelsErr  error
)

func buildServerFakeCLIBinary(t *testing.T) string {
	t.Helper()
	serverFakeCLIOnce.Do(func() {
		_, thisFile, _, ok := runtime.Caller(0)
		if !ok {
			serverFakeCLIErr = errors.New("runtime.Caller failed")
			return
		}
		serverDir := filepath.Dir(thisFile)
		repoRoot := filepath.Dir(serverDir)
		tmpDir, err := os.MkdirTemp("", "artex-server-fakecli-*")
		if err != nil {
			serverFakeCLIErr = err
			return
		}
		binName := "fakecli"
		if runtime.GOOS == "windows" {
			binName += ".exe"
		}
		outPath := filepath.Join(tmpDir, binName)
		cmd := exec.Command("go", "build", "-o", outPath, "./agent/testdata/fakecli")
		cmd.Dir = repoRoot
		if out, err := cmd.CombinedOutput(); err != nil {
			serverFakeCLIErr = fmt.Errorf("build fakecli: %w\n%s", err, string(out))
			return
		}
		serverFakeCLIPath = outPath
	})
	if serverFakeCLIErr != nil {
		t.Fatalf("failed to build fakecli: %v", serverFakeCLIErr)
	}
	return serverFakeCLIPath
}

func buildFakeAgyModelsBinary(t *testing.T) string {
	t.Helper()
	serverAgyModelsOnce.Do(func() {
		tmpDir, err := os.MkdirTemp("", "artex-server-agymodels-*")
		if err != nil {
			serverAgyModelsErr = err
			return
		}
		src := `package main

import (
	"fmt"
	"os"
)

func main() {
	if len(os.Args) > 1 && os.Args[1] == "models" {
		fmt.Println("Fetching available models...")
		fmt.Println()
		fmt.Println("gemini-4-hydrogen-high\tGemini 4 Hydrogen (High)")
		fmt.Println("claude-opus-5-5-high\tClaude Opus 5.5 (High)")
		fmt.Println("gemini-4-hydrogen-high\tGemini 4 Hydrogen (High)")
		return
	}
	fmt.Println("{\"event\":\"init\",\"conversation_id\":\"agy-models-1\",\"init\":{}}")
	fmt.Println("{\"event\":\"step_update\",\"step_update\":{\"step_index\":1,\"state\":\"ACTIVE\",\"step_type\":\"agent_response\",\"text_delta\":\"OK\"}}")
	fmt.Println("{\"event\":\"result\",\"result\":{\"conversation_id\":\"agy-models-1\",\"status\":\"SUCCESS\",\"response\":\"OK\",\"num_turns\":1,\"finished_reason\":\"STOP\",\"usage\":{\"input_tokens\":5,\"output_tokens\":2}}}")
}
`
		srcPath := filepath.Join(tmpDir, "main.go")
		if err := os.WriteFile(srcPath, []byte(src), 0o600); err != nil {
			serverAgyModelsErr = err
			return
		}
		binName := "fakeagy"
		if runtime.GOOS == "windows" {
			binName += ".exe"
		}
		outPath := filepath.Join(tmpDir, binName)
		cmd := exec.Command("go", "build", "-o", outPath, srcPath)
		cmd.Dir = tmpDir
		if out, err := cmd.CombinedOutput(); err != nil {
			serverAgyModelsErr = fmt.Errorf("build fakeagy: %w\n%s", err, string(out))
			return
		}
		serverAgyModelsPath = outPath
	})
	if serverAgyModelsErr != nil {
		t.Fatalf("failed to build fake agy binary: %v", serverAgyModelsErr)
	}
	return serverAgyModelsPath
}

// --- In-memory database/sql driver backing db.DB for zero-skip server CLI tests ---

type memSrvProfile struct {
	id                    int64
	name                  string
	format                string
	baseURL               string
	proxy                 string
	model                 string
	apiKey                string
	apiKeyHint            string
	ratePerSecond         float64
	ratePerMinute         float64
	contextWindowK        int
	reasoningEffort       string
	isDefault             bool
	priority              int
	poolExclude           bool
	thinkingType          string
	streaming             bool
	maxTokens             int
	maxTokensField        string
	sessionHeaderKey      string
	retryConnectAttempts  int
	retryConnectInterval  int
	retryEmptyAttempts    int
	retryEmptyInterval    int
	retryStreamAttempts   int
	retryStreamInterval   int
	priceInputPerM        float64
	priceOutputPerM       float64
	priceCacheReadPerM    float64
	priceCacheWritePerM   float64
	userAgent             string
}

type memSrvAgent struct {
	id           int64
	key          string
	name         string
	role         string
	builtin      bool
	enabled      bool
	maxTurns     int
	runSeconds   int
	webSearch    bool
	llmProfileID *int64
}

type memSrvTask struct {
	id                   int64
	name                 string
	description          string
	goal                 string
	expID                int64
	status               string
	llmProfileID         *int64
	activeLLMProfileID   *int64
	llmChainRevision     int64
	chain                []db.TaskLLMProfile
	coverageEnabled      bool
	planHeartbeatSeconds int
	planQuietMode        bool
}

type memSrvNode struct {
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

type memSrvEdge struct {
	expID int64
	src   int64
	rel   string
	dst   int64
}

type memSrvConstraint struct {
	id        int64
	expID     int64
	kind      string
	text      string
	origin    string
	createdAt time.Time
}

type memSrvConv struct {
	id           int64
	agentKey     string
	title        string
	llmProfileID *int64
	targetGate   bool
	createdAt    time.Time
	updatedAt    time.Time
}

type memSrvConvAct struct {
	id     int64
	convID int64
	worker string
	kind   string
	tool   string
	detail string
}

type memSrvUsage struct {
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

type memSrvState struct {
	mu          sync.Mutex
	nextID      int64
	profiles    map[int64]*memSrvProfile
	agents      map[string]*memSrvAgent
	tasks       map[int64]*memSrvTask
	nodes       map[int64]*memSrvNode
	edges       []memSrvEdge
	constraints []memSrvConstraint
	convs       map[int64]*memSrvConv
	convActs    []memSrvConvAct
	activities  []db.Activity
	settings    map[string]string
	health      map[int64]db.LLMHealth
	usages      []memSrvUsage
}

func newMemServerDB(t *testing.T) (*db.DB, *memSrvState) {
	t.Helper()
	t.Setenv("ARTEX_SECRET_KEY", "000102030405060708090a0b0c0d0e0f000102030405060708090a0b0c0d0e0f")
	st := &memSrvState{
		nextID:   100,
		profiles: map[int64]*memSrvProfile{},
		agents: map[string]*memSrvAgent{
			"worker":    {id: 1, key: "worker", name: "Worker", role: "worker", builtin: true, enabled: true, runSeconds: 300},
			"planner":   {id: 2, key: "planner", name: "Planner", role: "planner", builtin: true, enabled: true, runSeconds: 300},
			"mainagent": {id: 3, key: "mainagent", name: "MainAgent", role: "mainagent", builtin: true, enabled: true, runSeconds: 300},
			"goals":     {id: 4, key: "goals", name: "Goals", role: "goals", builtin: true, enabled: true, runSeconds: 300},
			"chat_bot":  {id: 5, key: "chat_bot", name: "ChatBot", role: "assistant", builtin: false, enabled: true, runSeconds: 300},
		},
		tasks:    map[int64]*memSrvTask{},
		nodes:    map[int64]*memSrvNode{},
		convs:    map[int64]*memSrvConv{},
		settings: map[string]string{},
		health:   map[int64]db.LLMHealth{},
	}
	sqlDB := sql.OpenDB(&memSrvConnector{st: st})
	t.Cleanup(func() { _ = sqlDB.Close() })
	d := &db.DB{DB: sqlDB}
	if err := d.InitSecrets(t.TempDir()); err != nil {
		t.Fatalf("InitSecrets: %v", err)
	}
	return d, st
}

func (s *memSrvState) addOriginFact(expID int64) int64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.nextID++
	id := s.nextID
	s.nodes[id] = &memSrvNode{
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

func (s *memSrvState) snapshotUsages() []memSrvUsage {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]memSrvUsage, len(s.usages))
	copy(out, s.usages)
	return out
}

func (s *memSrvState) snapshotNodes(expID int64, kind string) []*memSrvNode {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []*memSrvNode
	for _, n := range s.nodes {
		if n.expID == expID && (kind == "" || n.kind == kind) {
			cp := *n
			out = append(out, &cp)
		}
	}
	slices.SortFunc(out, func(a, b *memSrvNode) int {
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

type memSrvConnector struct{ st *memSrvState }

func (c *memSrvConnector) Connect(context.Context) (driver.Conn, error) {
	return &memSrvConn{st: c.st}, nil
}
func (c *memSrvConnector) Driver() driver.Driver { return nil }

type memSrvConn struct{ st *memSrvState }

func (c *memSrvConn) Prepare(string) (driver.Stmt, error) { return nil, nil }
func (c *memSrvConn) Close() error                        { return nil }
func (c *memSrvConn) Begin() (driver.Tx, error)           { return &memSrvTx{}, nil }
func (c *memSrvConn) BeginTx(context.Context, driver.TxOptions) (driver.Tx, error) {
	return &memSrvTx{}, nil
}

type memSrvTx struct{}

func (memSrvTx) Commit() error   { return nil }
func (memSrvTx) Rollback() error { return nil }

func srvAsString(v any) string {
	if v == nil {
		return ""
	}
	switch x := v.(type) {
	case string:
		return x
	case []byte:
		return string(x)
	default:
		return fmt.Sprint(x)
	}
}

func srvAsInt64(v any) int64 {
	if v == nil {
		return 0
	}
	switch x := v.(type) {
	case int64:
		return x
	case int:
		return int64(x)
	case float64:
		return int64(x)
	case string:
		n, _ := strconv.ParseInt(x, 10, 64)
		return n
	default:
		return 0
	}
}

func srvAsFloat64(v any) float64 {
	if v == nil {
		return 0
	}
	switch x := v.(type) {
	case float64:
		return x
	case int64:
		return float64(x)
	case int:
		return float64(x)
	default:
		return 0
	}
}

func srvAsBool(v any) bool {
	if b, ok := v.(bool); ok {
		return b
	}
	return false
}

func srvAsNullInt64Ptr(v any) *int64 {
	if v == nil {
		return nil
	}
	n := srvAsInt64(v)
	if n == 0 {
		return nil
	}
	return &n
}

func srvProfileRow(p *memSrvProfile, withSecret bool) []driver.Value {
	keyCol := p.apiKeyHint
	if withSecret {
		keyCol = p.apiKey
	}
	return []driver.Value{
		p.id, p.name, p.format, p.baseURL, p.proxy, p.model, keyCol,
		p.ratePerSecond, p.ratePerMinute, int64(p.contextWindowK), p.reasoningEffort, p.isDefault,
		int64(p.priority), p.poolExclude, p.thinkingType, p.streaming,
		int64(p.maxTokens), p.maxTokensField, p.sessionHeaderKey,
		int64(p.retryConnectAttempts), int64(p.retryConnectInterval),
		int64(p.retryEmptyAttempts), int64(p.retryEmptyInterval),
		int64(p.retryStreamAttempts), int64(p.retryStreamInterval),
		p.priceInputPerM, p.priceOutputPerM, p.priceCacheReadPerM, p.priceCacheWritePerM,
		p.userAgent,
	}
}

var srvProfileCols = []string{
	"id", "name", "format", "base_url", "proxy", "model", "key_or_hint",
	"rate_per_second", "rate_per_minute", "context_window_k", "reasoning_effort", "is_default",
	"priority", "pool_exclude", "thinking_type", "streaming",
	"max_tokens", "max_tokens_field", "session_header_key",
	"rc_a", "rc_i", "re_a", "re_i", "rs_a", "rs_i",
	"pi", "po", "pcr", "pcw", "user_agent",
}

func srvNodeRows(nodes []*memSrvNode) driver.Rows {
	cols := []string{"id", "kind", "payload", "priority", "state", "origin", "owner", "blocked_reason", "delete_reason", "created_at"}
	rows := make([][]driver.Value, 0, len(nodes))
	for _, n := range nodes {
		rows = append(rows, []driver.Value{
			n.id, n.kind, []byte(n.payload), int64(n.priority), n.state, n.origin, n.owner, n.blockedReason, n.deleteReason, n.createdAt,
		})
	}
	return &memSrvRows{cols: cols, rows: rows}
}

func srvTaskRows(tasks []*memSrvTask) driver.Rows {
	cols := []string{"id", "name", "category_id", "category_name", "description", "goal", "exploration_id", "status", "paused", "queued", "queued_at", "queue_mode", "llm_profile_id", "active_llm_profile_id", "parent_ref", "pinned_at", "created_at", "completed_at", "timeout_seconds", "plan_heartbeat_seconds", "plan_quiet_mode", "coverage_enabled", "llm_rotate", "first_run_at", "deadline_at", "target_gate"}
	rows := make([][]driver.Value, 0, len(tasks))
	now := time.Now().UTC()
	for _, tk := range tasks {
		var lid, aid any
		if tk.llmProfileID != nil {
			lid = *tk.llmProfileID
		}
		if tk.activeLLMProfileID != nil {
			aid = *tk.activeLLMProfileID
		}
		rows = append(rows, []driver.Value{
			tk.id, tk.name, nil, "", tk.description, tk.goal, tk.expID, tk.status,
			false, false, nil, "bootstrap", lid, aid, "", nil, now, nil,
			int64(0), int64(tk.planHeartbeatSeconds), tk.planQuietMode, tk.coverageEnabled, false, nil, nil, false,
		})
	}
	return &memSrvRows{cols: cols, rows: rows}
}

func srvAgentRow(a *memSrvAgent) []driver.Value {
	var lid any
	if a.llmProfileID != nil {
		lid = *a.llmProfileID
	}
	return []driver.Value{
		a.id, a.key, a.name, "", a.role, a.builtin, a.enabled,
		int64(a.maxTurns), int64(a.runSeconds), a.webSearch, false,
		"", int64(0), "", int64(0), "serial", "all", int64(5), lid,
	}
}

var srvAgentCols = []string{
	"id", "key", "name", "description", "role", "builtin", "enabled",
	"max_turns", "run_seconds", "web_search", "interactive_shell",
	"wrapup_prompt", "wrapup_max_turns", "twp", "twm", "trm", "tmm", "tmp", "llm_profile_id",
}

func (c *memSrvConn) ExecContext(_ context.Context, query string, args []driver.NamedValue) (driver.Result, error) {
	q := strings.Join(strings.Fields(query), " ")
	s := c.st
	s.mu.Lock()
	defer s.mu.Unlock()

	switch {
	case strings.HasPrefix(q, "UPDATE llm_profiles SET price_input_per_m="):
		id := srvAsInt64(args[4].Value)
		if p := s.profiles[id]; p != nil {
			p.priceInputPerM = srvAsFloat64(args[0].Value)
			p.priceOutputPerM = srvAsFloat64(args[1].Value)
			p.priceCacheReadPerM = srvAsFloat64(args[2].Value)
			p.priceCacheWritePerM = srvAsFloat64(args[3].Value)
		}
		return driver.RowsAffected(1), nil

	case strings.HasPrefix(q, "UPDATE llm_profiles SET is_default=false WHERE is_default"):
		for _, p := range s.profiles {
			p.isDefault = false
		}
		return driver.RowsAffected(1), nil

	case strings.HasPrefix(q, "UPDATE llm_profiles SET is_default=true WHERE id=$1"):
		id := srvAsInt64(args[0].Value)
		p := s.profiles[id]
		if p == nil {
			return driver.RowsAffected(0), nil
		}
		p.isDefault = true
		return driver.RowsAffected(1), nil

	case strings.HasPrefix(q, "UPDATE llm_profiles SET name=$1,format=$2,base_url=NULLIF($3,''),proxy=NULLIF($4,''),model=$5,rate_per_second="):
		id := srvAsInt64(args[23].Value)
		p := s.profiles[id]
		if p == nil {
			return driver.RowsAffected(0), nil
		}
		p.name = srvAsString(args[0].Value)
		p.format = srvAsString(args[1].Value)
		p.baseURL = srvAsString(args[2].Value)
		p.proxy = srvAsString(args[3].Value)
		p.model = srvAsString(args[4].Value)
		p.ratePerSecond = srvAsFloat64(args[5].Value)
		p.ratePerMinute = srvAsFloat64(args[6].Value)
		p.contextWindowK = int(srvAsInt64(args[7].Value))
		p.reasoningEffort = srvAsString(args[8].Value)
		p.priority = int(srvAsInt64(args[9].Value))
		p.poolExclude = srvAsBool(args[10].Value)
		p.thinkingType = srvAsString(args[11].Value)
		p.streaming = srvAsBool(args[12].Value)
		p.maxTokens = int(srvAsInt64(args[13].Value))
		p.maxTokensField = srvAsString(args[14].Value)
		p.sessionHeaderKey = srvAsString(args[15].Value)
		p.retryConnectAttempts = int(srvAsInt64(args[16].Value))
		p.retryConnectInterval = int(srvAsInt64(args[17].Value))
		p.retryEmptyAttempts = int(srvAsInt64(args[18].Value))
		p.retryEmptyInterval = int(srvAsInt64(args[19].Value))
		p.retryStreamAttempts = int(srvAsInt64(args[20].Value))
		p.retryStreamInterval = int(srvAsInt64(args[21].Value))
		p.userAgent = srvAsString(args[22].Value)
		return driver.RowsAffected(1), nil

	case strings.HasPrefix(q, "UPDATE llm_profiles SET name=$1,format=$2,base_url=NULLIF($3,''),proxy=NULLIF($4,''),model=$5,api_key="):
		id := srvAsInt64(args[25].Value)
		p := s.profiles[id]
		if p == nil {
			return driver.RowsAffected(0), nil
		}
		p.name = srvAsString(args[0].Value)
		p.format = srvAsString(args[1].Value)
		p.baseURL = srvAsString(args[2].Value)
		p.proxy = srvAsString(args[3].Value)
		p.model = srvAsString(args[4].Value)
		p.apiKey = srvAsString(args[5].Value)
		p.apiKeyHint = srvAsString(args[6].Value)
		p.ratePerSecond = srvAsFloat64(args[7].Value)
		p.ratePerMinute = srvAsFloat64(args[8].Value)
		p.contextWindowK = int(srvAsInt64(args[9].Value))
		p.reasoningEffort = srvAsString(args[10].Value)
		p.priority = int(srvAsInt64(args[11].Value))
		p.poolExclude = srvAsBool(args[12].Value)
		p.thinkingType = srvAsString(args[13].Value)
		p.streaming = srvAsBool(args[14].Value)
		p.maxTokens = int(srvAsInt64(args[15].Value))
		p.maxTokensField = srvAsString(args[16].Value)
		p.sessionHeaderKey = srvAsString(args[17].Value)
		p.retryConnectAttempts = int(srvAsInt64(args[18].Value))
		p.retryConnectInterval = int(srvAsInt64(args[19].Value))
		p.retryEmptyAttempts = int(srvAsInt64(args[20].Value))
		p.retryEmptyInterval = int(srvAsInt64(args[21].Value))
		p.retryStreamAttempts = int(srvAsInt64(args[22].Value))
		p.retryStreamInterval = int(srvAsInt64(args[23].Value))
		p.userAgent = srvAsString(args[24].Value)
		return driver.RowsAffected(1), nil

	case strings.HasPrefix(q, "UPDATE agents SET llm_profile_id=$1 WHERE key=$2"):
		key := srvAsString(args[1].Value)
		if a := s.agents[key]; a != nil {
			a.llmProfileID = srvAsNullInt64Ptr(args[0].Value)
		}
		return driver.RowsAffected(1), nil

	case strings.HasPrefix(q, "INSERT INTO settings("):
		key := srvAsString(args[0].Value)
		val := srvAsString(args[1].Value)
		s.settings[key] = val
		return driver.RowsAffected(1), nil

	case strings.HasPrefix(q, "INSERT INTO llm_health"):
		id := srvAsInt64(args[0].Value)
		s.health[id] = db.LLMHealth{
			ProfileID: id,
			Fails:     int(srvAsInt64(args[1].Value)),
			Trips:     int(srvAsInt64(args[2].Value)),
			LastError: srvAsString(args[4].Value),
		}
		return driver.RowsAffected(1), nil

	case strings.HasPrefix(q, "DELETE FROM llm_health WHERE profile_id=$1"):
		delete(s.health, srvAsInt64(args[0].Value))
		return driver.RowsAffected(1), nil

	case strings.HasPrefix(q, "INSERT INTO llm_usage("):
		s.usages = append(s.usages, memSrvUsage{
			taskID:      srvAsString(args[0].Value),
			expID:       srvAsInt64(args[1].Value),
			worker:      srvAsString(args[2].Value),
			model:       srvAsString(args[3].Value),
			profileName: srvAsString(args[4].Value),
			latencyMs:   int(srvAsInt64(args[5].Value)),
			input:       int(srvAsInt64(args[6].Value)),
			output:      int(srvAsInt64(args[7].Value)),
			cacheRead:   int(srvAsInt64(args[8].Value)),
			cacheWrite:  int(srvAsInt64(args[9].Value)),
			status:      srvAsString(args[10].Value),
		})
		return driver.RowsAffected(1), nil

	case strings.HasPrefix(q, "INSERT INTO exploration_anchors"),
		strings.HasPrefix(q, "UPDATE tasks SET updated_at"),
		strings.HasPrefix(q, "UPDATE conversations SET updated_at"):
		return driver.RowsAffected(1), nil

	case strings.HasPrefix(q, "INSERT INTO exploration_edges"):
		expID := srvAsInt64(args[0].Value)
		src := srvAsInt64(args[1].Value)
		rel := srvAsString(args[2].Value)
		dst := srvAsInt64(args[3].Value)
		for _, e := range s.edges {
			if e.expID == expID && e.src == src && e.rel == rel && e.dst == dst {
				return driver.RowsAffected(0), nil
			}
		}
		s.edges = append(s.edges, memSrvEdge{expID: expID, src: src, rel: rel, dst: dst})
		return driver.RowsAffected(1), nil

	case strings.HasPrefix(q, "UPDATE task_llm_profiles SET status='quota_exhausted'"):
		taskID := srvAsInt64(args[0].Value)
		profID := srvAsInt64(args[1].Value)
		reason := srvAsString(args[2].Value)
		if tk := s.tasks[taskID]; tk != nil {
			now := time.Now().UTC()
			for i := range tk.chain {
				if tk.chain[i].ProfileID == profID {
					tk.chain[i].Status = "quota_exhausted"
					tk.chain[i].LastError = reason
					tk.chain[i].ExhaustedAt = &now
				}
			}
		}
		return driver.RowsAffected(1), nil

	case strings.HasPrefix(q, "UPDATE tasks SET active_llm_profile_id="):
		taskID := srvAsInt64(args[0].Value)
		if tk := s.tasks[taskID]; tk != nil {
			var next *int64
			if len(args) > 1 && args[1].Value != nil {
				v := srvAsInt64(args[1].Value)
				next = &v
			}
			tk.activeLLMProfileID = next
			tk.llmProfileID = next
			tk.llmChainRevision++
		}
		return driver.RowsAffected(1), nil
	}
	return driver.RowsAffected(0), nil
}

func (c *memSrvConn) QueryContext(_ context.Context, query string, args []driver.NamedValue) (driver.Rows, error) {
	q := strings.Join(strings.Fields(query), " ")
	s := c.st
	s.mu.Lock()
	defer s.mu.Unlock()

	switch {
	case strings.HasPrefix(q, "INSERT INTO llm_profiles("):
		s.nextID++
		id := s.nextID
		p := &memSrvProfile{
			id:                    id,
			name:                  srvAsString(args[0].Value),
			format:                srvAsString(args[1].Value),
			baseURL:               srvAsString(args[2].Value),
			proxy:                 srvAsString(args[3].Value),
			model:                 srvAsString(args[4].Value),
			apiKey:                srvAsString(args[5].Value),
			apiKeyHint:            srvAsString(args[6].Value),
			ratePerSecond:         srvAsFloat64(args[7].Value),
			ratePerMinute:         srvAsFloat64(args[8].Value),
			contextWindowK:        int(srvAsInt64(args[9].Value)),
			reasoningEffort:       srvAsString(args[10].Value),
			priority:              int(srvAsInt64(args[11].Value)),
			poolExclude:           srvAsBool(args[12].Value),
			thinkingType:          srvAsString(args[13].Value),
			streaming:             srvAsBool(args[14].Value),
			maxTokens:             int(srvAsInt64(args[15].Value)),
			maxTokensField:        srvAsString(args[16].Value),
			sessionHeaderKey:      srvAsString(args[17].Value),
			retryConnectAttempts:  int(srvAsInt64(args[18].Value)),
			retryConnectInterval:  int(srvAsInt64(args[19].Value)),
			retryEmptyAttempts:    int(srvAsInt64(args[20].Value)),
			retryEmptyInterval:    int(srvAsInt64(args[21].Value)),
			retryStreamAttempts:   int(srvAsInt64(args[22].Value)),
			retryStreamInterval:   int(srvAsInt64(args[23].Value)),
			userAgent:             srvAsString(args[24].Value),
		}
		s.profiles[id] = p
		return &memSrvRows{cols: []string{"id"}, rows: [][]driver.Value{{id}}}, nil

	case strings.Contains(q, "FROM llm_profiles ORDER BY id"):
		var list []*memSrvProfile
		for _, p := range s.profiles {
			list = append(list, p)
		}
		slices.SortFunc(list, func(a, b *memSrvProfile) int { return int(a.id - b.id) })
		rows := make([][]driver.Value, 0, len(list))
		for _, p := range list {
			rows = append(rows, srvProfileRow(p, false))
		}
		return &memSrvRows{cols: srvProfileCols, rows: rows}, nil

	case strings.Contains(q, "FROM llm_profiles WHERE is_default LIMIT 1"):
		for _, p := range s.profiles {
			if p.isDefault {
				return &memSrvRows{cols: srvProfileCols, rows: [][]driver.Value{srvProfileRow(p, true)}}, nil
			}
		}
		return &memSrvRows{cols: srvProfileCols}, nil

	case strings.Contains(q, "FROM llm_profiles WHERE id=$1") && strings.Contains(q, "api_key"):
		id := srvAsInt64(args[0].Value)
		if p := s.profiles[id]; p != nil {
			return &memSrvRows{cols: srvProfileCols, rows: [][]driver.Value{srvProfileRow(p, true)}}, nil
		}
		return &memSrvRows{cols: srvProfileCols}, nil

	case strings.Contains(q, "FROM llm_profiles WHERE (COALESCE(api_key,'')"):
		var list []*memSrvProfile
		for _, p := range s.profiles {
			if (p.apiKey != "" || db.IsKeylessProfileFormat(p.format)) && (p.isDefault || !p.poolExclude) {
				list = append(list, p)
			}
		}
		slices.SortFunc(list, func(a, b *memSrvProfile) int {
			if a.isDefault != b.isDefault {
				if a.isDefault {
					return -1
				}
				return 1
			}
			if a.priority != b.priority {
				return b.priority - a.priority
			}
			return int(a.id - b.id)
		})
		rows := make([][]driver.Value, 0, len(list))
		for _, p := range list {
			rows = append(rows, srvProfileRow(p, true))
		}
		return &memSrvRows{cols: srvProfileCols, rows: rows}, nil

	case strings.Contains(q, "FROM llm_health"):
		return &memSrvRows{cols: []string{"profile_id", "fails", "trips", "open_until", "last_error", "last_at"}}, nil

	case strings.Contains(q, "FROM agents WHERE key=$1"):
		key := srvAsString(args[0].Value)
		if a := s.agents[key]; a != nil {
			return &memSrvRows{cols: srvAgentCols, rows: [][]driver.Value{srvAgentRow(a)}}, nil
		}
		return &memSrvRows{cols: srvAgentCols}, nil

	case strings.HasPrefix(q, "SELECT value FROM settings WHERE key=$1"):
		key := srvAsString(args[0].Value)
		if v, ok := s.settings[key]; ok {
			return &memSrvRows{cols: []string{"value"}, rows: [][]driver.Value{{v}}}, nil
		}
		return &memSrvRows{cols: []string{"value"}}, nil

	case strings.Contains(q, "FROM tasks WHERE id=$1 AND deleted_at IS NULL"):
		id := srvAsInt64(args[0].Value)
		if tk := s.tasks[id]; tk != nil {
			return srvTaskRows([]*memSrvTask{tk}), nil
		}
		return srvTaskRows(nil), nil

	case strings.Contains(q, "FROM tasks t LEFT JOIN task_llm_profiles p ON p.task_id=t.id WHERE t.id=$1"):
		tid := srvAsInt64(args[0].Value)
		cols := []string{"legacy", "active", "rev", "pid", "pos", "status", "last_err", "exh"}
		var rows [][]driver.Value
		if tk := s.tasks[tid]; tk != nil {
			var lid, aid any
			if tk.llmProfileID != nil {
				lid = *tk.llmProfileID
			}
			if tk.activeLLMProfileID != nil {
				aid = *tk.activeLLMProfileID
			}
			if len(tk.chain) == 0 {
				rows = append(rows, []driver.Value{lid, aid, tk.llmChainRevision, nil, nil, nil, "", nil})
			} else {
				for _, e := range tk.chain {
					var exh any
					if e.ExhaustedAt != nil {
						exh = *e.ExhaustedAt
					}
					rows = append(rows, []driver.Value{lid, aid, tk.llmChainRevision, e.ProfileID, int64(e.Position), e.Status, e.LastError, exh})
				}
			}
		}
		return &memSrvRows{cols: cols, rows: rows}, nil

	case strings.Contains(q, "FROM tasks t LEFT JOIN task_llm_profiles p ON p.task_id=t.id"):
		cols := []string{"id", "legacy", "active", "rev", "pid", "pos", "status", "last_err", "exh"}
		var rows [][]driver.Value
		for _, tk := range s.tasks {
			var lid, aid any
			if tk.llmProfileID != nil {
				lid = *tk.llmProfileID
			}
			if tk.activeLLMProfileID != nil {
				aid = *tk.activeLLMProfileID
			}
			if len(tk.chain) == 0 {
				rows = append(rows, []driver.Value{tk.id, lid, aid, tk.llmChainRevision, nil, nil, nil, "", nil})
				continue
			}
			for _, e := range tk.chain {
				var exh any
				if e.ExhaustedAt != nil {
					exh = *e.ExhaustedAt
				}
				rows = append(rows, []driver.Value{tk.id, lid, aid, tk.llmChainRevision, e.ProfileID, int64(e.Position), e.Status, e.LastError, exh})
			}
		}
		return &memSrvRows{cols: cols, rows: rows}, nil

	case strings.Contains(q, "FROM task_relations WHERE task_id=ANY"):
		return &memSrvRows{cols: []string{"task_id", "source_task_id"}}, nil

	case strings.Contains(q, "FROM task_scope WHERE task_id=ANY"):
		return &memSrvRows{cols: []string{"task_id", "company_id"}}, nil

	case strings.HasPrefix(q, "SELECT profile_id, position, status, COALESCE(last_error,''), exhausted_at FROM task_llm_profiles WHERE task_id=$1"):
		tid := srvAsInt64(args[0].Value)
		cols := []string{"profile_id", "position", "status", "last_error", "exhausted_at"}
		var rows [][]driver.Value
		if tk := s.tasks[tid]; tk != nil {
			for _, e := range tk.chain {
				var exh any
				if e.ExhaustedAt != nil {
					exh = *e.ExhaustedAt
				}
				rows = append(rows, []driver.Value{e.ProfileID, int64(e.Position), e.Status, e.LastError, exh})
			}
		}
		return &memSrvRows{cols: cols, rows: rows}, nil

	case strings.HasPrefix(q, "SELECT active_llm_profile_id, llm_chain_revision FROM tasks WHERE id=$1 FOR UPDATE"):
		tid := srvAsInt64(args[0].Value)
		cols := []string{"active_llm_profile_id", "llm_chain_revision"}
		if tk := s.tasks[tid]; tk != nil {
			var aid any
			if tk.activeLLMProfileID != nil {
				aid = *tk.activeLLMProfileID
			}
			return &memSrvRows{cols: cols, rows: [][]driver.Value{{aid, tk.llmChainRevision}}}, nil
		}
		return &memSrvRows{cols: cols}, nil

	case strings.HasPrefix(q, "SELECT position, status FROM task_llm_profiles WHERE task_id=$1 AND profile_id=$2"):
		tid := srvAsInt64(args[0].Value)
		pid := srvAsInt64(args[1].Value)
		cols := []string{"position", "status"}
		if tk := s.tasks[tid]; tk != nil {
			for _, e := range tk.chain {
				if e.ProfileID == pid {
					return &memSrvRows{cols: cols, rows: [][]driver.Value{{int64(e.Position), e.Status}}}, nil
				}
			}
		}
		return &memSrvRows{cols: cols}, nil

	case strings.HasPrefix(q, "SELECT profile_id FROM task_llm_profiles WHERE task_id=$1 AND position>$2 AND status='ready'"):
		tid := srvAsInt64(args[0].Value)
		pos := int(srvAsInt64(args[1].Value))
		cols := []string{"profile_id"}
		if tk := s.tasks[tid]; tk != nil {
			for _, e := range tk.chain {
				if e.Position > pos && e.Status == "ready" {
					return &memSrvRows{cols: cols, rows: [][]driver.Value{{e.ProfileID}}}, nil
				}
			}
		}
		return &memSrvRows{cols: cols}, nil

	case strings.HasPrefix(q, "INSERT INTO exploration_nodes("):
		s.nextID++
		id := s.nextID
		s.nodes[id] = &memSrvNode{
			id:        id,
			expID:     srvAsInt64(args[0].Value),
			kind:      srvAsString(args[1].Value),
			payload:   srvAsString(args[2].Value),
			priority:  int(srvAsInt64(args[3].Value)),
			state:     srvAsString(args[4].Value),
			origin:    srvAsString(args[5].Value),
			createdAt: time.Now().UTC(),
		}
		return &memSrvRows{cols: []string{"id"}, rows: [][]driver.Value{{id}}}, nil

	case strings.HasPrefix(q, "INSERT INTO task_constraints("):
		s.nextID++
		id := s.nextID
		s.constraints = append(s.constraints, memSrvConstraint{
			id:        id,
			expID:     srvAsInt64(args[0].Value),
			kind:      srvAsString(args[1].Value),
			text:      srvAsString(args[2].Value),
			origin:    srvAsString(args[3].Value),
			createdAt: time.Now().UTC(),
		})
		return &memSrvRows{cols: []string{"id"}, rows: [][]driver.Value{{id}}}, nil

	case strings.Contains(q, "FROM exploration_nodes WHERE exploration_id=$1 AND kind='fact' AND state='origin'"):
		expID := srvAsInt64(args[0].Value)
		for id, n := range s.nodes {
			if n.expID == expID && n.kind == db.KindFact && n.state == db.StateOrigin {
				return &memSrvRows{cols: []string{"id"}, rows: [][]driver.Value{{id}}}, nil
			}
		}
		return &memSrvRows{cols: []string{"id"}}, nil

	case strings.Contains(q, "FROM exploration_nodes WHERE exploration_id=$1 AND kind=$2 ORDER BY id DESC LIMIT $3"):
		expID := srvAsInt64(args[0].Value)
		kind := srvAsString(args[1].Value)
		limit := int(srvAsInt64(args[2].Value))
		var matched []*memSrvNode
		for _, n := range s.nodes {
			if n.expID == expID && n.kind == kind {
				matched = append(matched, n)
			}
		}
		slices.SortFunc(matched, func(a, b *memSrvNode) int { return int(b.id - a.id) })
		if limit > 0 && len(matched) > limit {
			matched = matched[:limit]
		}
		return srvNodeRows(matched), nil

	case strings.HasPrefix(q, "INSERT INTO activity("):
		s.nextID++
		id := s.nextID
		s.activities = append(s.activities, db.Activity{
			ID:      id,
			Worker:  srvAsString(args[2].Value),
			Kind:    srvAsString(args[3].Value),
			Tool:    srvAsString(args[4].Value),
			Summary: srvAsString(args[7].Value),
			Detail:  srvAsString(args[8].Value),
		})
		return &memSrvRows{cols: []string{"id"}, rows: [][]driver.Value{{id}}}, nil

	case strings.Contains(q, "FROM conversations WHERE id=$1") && !strings.Contains(q, "FOR UPDATE"):
		id := srvAsInt64(args[0].Value)
		cols := []string{"id", "agent_key", "title", "llm_profile_id", "pinned_at", "created_at", "updated_at", "target_gate"}
		if cv := s.convs[id]; cv != nil {
			var lid any
			if cv.llmProfileID != nil {
				lid = *cv.llmProfileID
			}
			return &memSrvRows{
				cols: cols,
				rows: [][]driver.Value{{cv.id, cv.agentKey, cv.title, lid, nil, cv.createdAt, cv.updatedAt, cv.targetGate}},
			}, nil
		}
		return &memSrvRows{cols: cols}, nil

	case strings.HasPrefix(q, "INSERT INTO conversations("):
		s.nextID++
		id := s.nextID
		now := time.Now().UTC()
		cv := &memSrvConv{
			id:         id,
			agentKey:   srvAsString(args[0].Value),
			title:      srvAsString(args[1].Value),
			targetGate: srvAsBool(args[2].Value),
			createdAt:  now,
			updatedAt:  now,
		}
		s.convs[id] = cv
		return &memSrvRows{
			cols: []string{"id", "agent_key", "title", "llm_profile_id", "pinned_at", "created_at", "updated_at", "target_gate"},
			rows: [][]driver.Value{{cv.id, cv.agentKey, cv.title, nil, nil, cv.createdAt, cv.updatedAt, cv.targetGate}},
		}, nil

	case strings.HasPrefix(q, "INSERT INTO conversation_activities("):
		s.nextID++
		id := s.nextID
		s.convActs = append(s.convActs, memSrvConvAct{
			id:     id,
			convID: srvAsInt64(args[0].Value),
			worker: srvAsString(args[1].Value),
			kind:   srvAsString(args[2].Value),
			tool:   srvAsString(args[3].Value),
			detail: srvAsString(args[7].Value),
		})
		return &memSrvRows{cols: []string{"id"}, rows: [][]driver.Value{{id}}}, nil

	case strings.HasPrefix(q, "SELECT id FROM tasks WHERE id=$1") && strings.Contains(q, "FOR UPDATE"):
		id := srvAsInt64(args[0].Value)
		if tk := s.tasks[id]; tk != nil {
			return &memSrvRows{cols: []string{"id"}, rows: [][]driver.Value{{tk.id}}}, nil
		}
		return &memSrvRows{cols: []string{"id"}}, nil

	case strings.Contains(q, "FROM task_constraints WHERE exploration_id=$1"):
		expID := srvAsInt64(args[0].Value)
		cols := []string{"id", "exploration_id", "kind", "text", "origin", "created_at"}
		var rows [][]driver.Value
		for _, cst := range s.constraints {
			if cst.expID == expID {
				rows = append(rows, []driver.Value{cst.id, cst.expID, cst.kind, cst.text, cst.origin, cst.createdAt})
			}
		}
		return &memSrvRows{cols: cols, rows: rows}, nil
	}
	return &memSrvRows{}, nil
}

type memSrvRows struct {
	cols []string
	rows [][]driver.Value
	idx  int
}

func (r *memSrvRows) Columns() []string { return r.cols }
func (r *memSrvRows) Close() error      { return nil }
func (r *memSrvRows) Next(dest []driver.Value) error {
	if r.idx >= len(r.rows) {
		return io.EOF
	}
	row := r.rows[r.idx]
	r.idx++
	for i := range dest {
		if i < len(row) {
			dest[i] = row[i]
		}
	}
	return nil
}

func newMemTestServer(t *testing.T) (*Server, *memSrvState) {
	t.Helper()
	pg, st := newMemServerDB(t)
	dir := t.TempDir()
	ctx, cancel := context.WithCancel(context.Background())
	ic := intercept.New(pg)
	ic.SetRulesForTest([]db.InterceptRule{
		{
			ID:          1,
			Name:        "deny blocked rm",
			Enabled:     true,
			MatchTarget: "tool_input",
			MatchType:   "regex",
			Pattern:     `rm -rf /blocked`,
			Action:      "deny",
			Message:     "guard denied dangerous command",
		},
		{
			ID:          2,
			Name:        "allow echo",
			Enabled:     true,
			MatchTarget: "tool_input",
			MatchType:   "regex",
			Pattern:     `echo allowed`,
			Action:      "allow",
			Message:     "allowed safe command",
		},
	})
	m := &Manager{
		pg:          pg,
		dir:         dir,
		tasks:       map[string]*Task{},
		interceptor: ic,
	}
	s := &Server{
		m:              m,
		engine:         NewEngine(m),
		ctx:            ctx,
		chatBusy:       map[string]bool{},
		chatCancel:     map[string]context.CancelCauseFunc{},
		triggerQ:       map[string][]triggeredRun{},
		triggerActive:  map[string]int{},
		triggerCfg:     map[string]triggerBehavior{},
		profChatAgents: map[int64]*agent.ChatAgent{},
		provByProfile:  map[int64]*provEntry{},
		llmHealth:      newLLMHealthRegistry(nil),
		taskAgents:     map[string]*taskAgentBundle{},
		archiveWake:    make(chan struct{}, 1),
	}
	s.ensureMCPBridge()
	s.engine.SetAuthoritativeAgentResolver(func(tk *Task) (*agent.Planner, *agent.Worker) {
		if !s.taskRuntimeAvailable(tk, "planner", "worker") {
			return nil, nil
		}
		b := s.agentsForTask(tk)
		return b.pl, b.wk
	})
	s.engine.SetCLIHandlers(
		func(callCtx context.Context, tk *Task, agentKey string) any {
			return s.beginTaskCLIAttempt(callCtx, tk, agentKey)
		},
		func(snap any, committed bool, err error) (bool, error) {
			sn, _ := snap.(*cliAttemptSnapshot)
			return finishCLIAttempt(sn, committed, err)
		},
	)
	t.Cleanup(func() {
		cancel()
		_ = s.Close()
	})
	return s, st
}

// --- Tests ---

func TestCLIKeylessProfileManagementAndValidation(t *testing.T) {
	s, st := newMemTestServer(t)
	fakeBin := buildServerFakeCLIBinary(t)

	doSave := func(payload map[string]any) *httptest.ResponseRecorder {
		b, _ := json.Marshal(payload)
		req := httptest.NewRequest(http.MethodPost, "/api/llm/profiles", bytes.NewReader(b))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		s.pgSaveProfile(rec, req)
		return rec
	}

	// Create keyless claude-cli profile with local binary path override.
	rec := doSave(map[string]any{
		"name":             "claude-cli-local",
		"format":           "claude-cli",
		"model":            "sonnet",
		"base_url":         fakeBin,
		"proxy":            "http://proxy.internal:8080",
		"reasoning_effort": "high",
		"thinking_type":    "enabled",
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("create claude-cli profile: HTTP %d body=%s", rec.Code, rec.Body.String())
	}
	var saveResp struct {
		ID int64 `json:"id"`
	}
	_ = json.Unmarshal(rec.Body.Bytes(), &saveResp)
	if saveResp.ID == 0 {
		t.Fatalf("expected non-zero profile id, body=%s", rec.Body.String())
	}

	// Create keyless agy-cli profile omitting model (defaults to agy default model).
	recAgy := doSave(map[string]any{
		"name":     "agy-cli-local",
		"format":   "agy-cli",
		"base_url": fakeBin,
	})
	if recAgy.Code != http.StatusOK {
		t.Fatalf("create agy-cli profile: HTTP %d body=%s", recAgy.Code, recAgy.Body.String())
	}
	var saveAgy struct {
		ID int64 `json:"id"`
	}
	_ = json.Unmarshal(recAgy.Body.Bytes(), &saveAgy)

	// Update claude-cli profile keeping masked proxy and setting another local file path.
	localOverride := filepath.Join(t.TempDir(), "custom-claude-bin")
	recUpd := doSave(map[string]any{
		"id":       saveResp.ID,
		"name":     "claude-cli-local",
		"format":   "claude-cli",
		"model":    "opus",
		"base_url": localOverride,
		"proxy":    "__masked__",
	})
	if recUpd.Code != http.StatusOK {
		t.Fatalf("update claude-cli profile with local path failed: HTTP %d body=%s", recUpd.Code, recUpd.Body.String())
	}

	// Validation errors: empty name, bad format, control char in CLI path.
	if r := doSave(map[string]any{"name": "  ", "format": "claude-cli"}); r.Code != http.StatusBadRequest {
		t.Fatalf("empty name expected 400, got %d", r.Code)
	}
	if r := doSave(map[string]any{"name": "bad", "format": "invalid-fmt"}); r.Code != http.StatusBadRequest {
		t.Fatalf("bad format expected 400, got %d", r.Code)
	}
	if r := doSave(map[string]any{"name": "bad-bin", "format": "claude-cli", "base_url": "bad\npath"}); r.Code != http.StatusBadRequest {
		t.Fatalf("control char path expected 400, got %d", r.Code)
	}

	// Verify masked/redacted API list output preserves CLI base_url and masks proxy.
	listReq := httptest.NewRequest(http.MethodGet, "/api/llm/profiles", nil)
	listRec := httptest.NewRecorder()
	s.pgListProfiles(listRec, listReq)
	if listRec.Code != http.StatusOK {
		t.Fatalf("list profiles HTTP %d: %s", listRec.Code, listRec.Body.String())
	}
	var listResp struct {
		Profiles []LLMProfileDTO `json:"profiles"`
	}
	if err := json.Unmarshal(listRec.Body.Bytes(), &listResp); err != nil {
		t.Fatal(err)
	}
	var foundClaude, foundAgy *LLMProfileDTO
	for i := range listResp.Profiles {
		p := &listResp.Profiles[i]
		if p.Name == "claude-cli-local" {
			foundClaude = p
		}
		if p.Name == "agy-cli-local" {
			foundAgy = p
		}
	}
	if foundClaude == nil || foundClaude.Format != "claude-cli" || foundClaude.Model != "opus" || foundClaude.BaseURL != localOverride || !foundClaude.ProxySet || foundClaude.Proxy != "__masked__" || foundClaude.APIKeyHint != "" {
		t.Fatalf("unexpected claude-cli DTO: %+v", foundClaude)
	}
	if foundAgy == nil || foundAgy.Format != "agy-cli" || foundAgy.Model != "gemini-4-hydrogen-high" || foundAgy.BaseURL != fakeBin || foundAgy.ProxySet {
		t.Fatalf("unexpected agy-cli DTO: %+v", foundAgy)
	}
	if st.profiles[saveResp.ID].proxy == "" {
		t.Fatal("masked proxy update cleared stored proxy")
	}

	// Verify targetGateEndpoints excludes CLI binary paths while keeping HTTP API endpoints.
	if _, err := s.m.pg.SaveProfile(&db.LLMProfile{
		Name: "http-api", Format: "anthropic", BaseURL: "https://api.example.com/v1", APIKey: "sk-ant-test", Model: "claude-opus-4-8",
	}); err != nil {
		t.Fatal(err)
	}
	eps := s.m.targetGateEndpoints()
	if !slices.Contains(eps, "https://api.example.com/v1") || !slices.Contains(eps, "http://proxy.internal:8080") {
		t.Fatalf("expected HTTP endpoint and proxy in targetGateEndpoints, got %v", eps)
	}
	for _, ep := range eps {
		if ep == localOverride || ep == fakeBin {
			t.Fatalf("targetGateEndpoints unexpectedly included local CLI binary path %q in %v", ep, eps)
		}
	}
}

func TestCLITestLLMAndListModels(t *testing.T) {
	s, _ := newMemTestServer(t)
	fakeBin := buildServerFakeCLIBinary(t)
	agyBin := buildFakeAgyModelsBinary(t)

	pidClaude, err := s.m.pg.SaveProfile(&db.LLMProfile{
		Name: "test-claude", Format: "claude-cli", Model: "sonnet", BaseURL: fakeBin, Streaming: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	pidAgy, err := s.m.pg.SaveProfile(&db.LLMProfile{
		Name: "test-agy", Format: "agy-cli", Model: "gemini-4-hydrogen-high", BaseURL: fakeBin, Streaming: true,
	})
	if err != nil {
		t.Fatal(err)
	}

	callTest := func(body map[string]any) map[string]any {
		b, _ := json.Marshal(body)
		req := httptest.NewRequest(http.MethodPost, "/api/llm/test", bytes.NewReader(b))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		s.testLLM(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("testLLM HTTP %d: %s", rec.Code, rec.Body.String())
		}
		var out map[string]any
		_ = json.Unmarshal(rec.Body.Bytes(), &out)
		return out
	}

	// testLLM via form (keyless claude-cli) -> "OK".
	if resp := callTest(map[string]any{"provider": "claude-cli", "model": "sonnet", "base_url": fakeBin}); resp["ok"] != true || resp["reply"] != "OK" {
		t.Fatalf("testLLM claude-cli failed: %v", resp)
	}
	// testLLM via stored profile_id (keyless agy-cli) -> "OK".
	if resp := callTest(map[string]any{"profile_id": pidAgy}); resp["ok"] != true || resp["reply"] != "OK" {
		t.Fatalf("testLLM agy-cli stored profile_id failed: %v", resp)
	}
	_ = pidClaude
	// testLLM with missing binary -> clear error.
	missingPath := filepath.Join(t.TempDir(), "no-such-cli-bin")
	if resp := callTest(map[string]any{"provider": "claude-cli", "base_url": missingPath}); resp["ok"] != false || !strings.Contains(fmt.Sprint(resp["error"]), "not accessible") {
		t.Fatalf("testLLM missing binary unexpected response: %v", resp)
	}

	callModels := func(body map[string]any) map[string]any {
		b, _ := json.Marshal(body)
		req := httptest.NewRequest(http.MethodPost, "/api/llm/models", bytes.NewReader(b))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		s.pgListModels(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("pgListModels HTTP %d: %s", rec.Code, rec.Body.String())
		}
		var out map[string]any
		_ = json.Unmarshal(rec.Body.Bytes(), &out)
		return out
	}

	// listModels claude-cli -> static ["sonnet", "opus", "haiku"] keyless.
	claudeModels := callModels(map[string]any{"provider": "claude-cli"})
	if claudeModels["ok"] != true {
		t.Fatalf("listModels claude-cli failed: %v", claudeModels)
	}
	if raw, _ := json.Marshal(claudeModels["models"]); string(raw) != `["sonnet","opus","haiku"]` {
		t.Fatalf("claude models = %s, want [sonnet opus haiku]", raw)
	}

	// listModels agy-cli -> invokes `<bin> models` keyless.
	agyModels := callModels(map[string]any{"provider": "agy-cli", "base_url": agyBin})
	if agyModels["ok"] != true {
		t.Fatalf("listModels agy-cli failed: %v", agyModels)
	}
	if raw, _ := json.Marshal(agyModels["models"]); string(raw) != `["gemini-4-hydrogen-high","claude-opus-5-5-high"]` {
		t.Fatalf("agy models = %s, want [gemini-4-hydrogen-high claude-opus-5-5-high]", raw)
	}

	// listModels agy-cli with missing binary -> ok:false.
	if badModels := callModels(map[string]any{"provider": "agy-cli", "base_url": missingPath}); badModels["ok"] != false || badModels["error"] == "" {
		t.Fatalf("listModels missing agy binary expected error, got %v", badModels)
	}
}

func TestCLIKeylessReadinessAndTaskResolution(t *testing.T) {
	s, st := newMemTestServer(t)
	fakeBin := buildServerFakeCLIBinary(t)

	for _, f := range []string{"ollama", "claude-cli", "agy-cli"} {
		if !db.IsKeylessProfileFormat(f) {
			t.Fatalf("IsKeylessProfileFormat(%q) = false", f)
		}
	}
	for _, f := range []string{"openai", "anthropic", "openai-responses", ""} {
		if db.IsKeylessProfileFormat(f) {
			t.Fatalf("IsKeylessProfileFormat(%q) = true", f)
		}
	}

	// Ready CLI profile vs missing-binary CLI profile.
	pidReady, err := s.m.pg.SaveProfile(&db.LLMProfile{
		Name: "ready-cli", Format: "claude-cli", Model: "sonnet", BaseURL: fakeBin, Streaming: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	missingBin := filepath.Join(t.TempDir(), "missing-agy-bin")
	pidMissing, err := s.m.pg.SaveProfile(&db.LLMProfile{
		Name: "missing-cli", Format: "agy-cli", Model: "gemini-4-hydrogen-high", BaseURL: missingBin, Streaming: true,
	})
	if err != nil {
		t.Fatal(err)
	}

	pReady, _ := s.m.pg.ProfileByID(pidReady)
	pMissing, _ := s.m.pg.ProfileByID(pidMissing)

	resReady := s.resolutionFromProfile(pReady, "task_chain")
	if !resReady.Available || resReady.Reason != "" || resReady.Format != "claude-cli" {
		t.Fatalf("ready CLI resolution = %+v", resReady)
	}
	resMissing := s.resolutionFromProfile(pMissing, "task_chain")
	if resMissing.Available || !strings.Contains(resMissing.Reason, "not accessible") {
		t.Fatalf("missing CLI resolution = %+v, want unavailable with binary error", resMissing)
	}

	// Check ReadyFor / taskRuntimeAvailable with task chain.
	st.tasks[7] = &memSrvTask{
		id: 7, expID: 12, description: "cli readiness", goal: "test", status: "running",
		activeLLMProfileID: &pidReady,
		chain: []db.TaskLLMProfile{
			{ProfileID: pidReady, Position: 0, Status: "ready"},
		},
	}
	task := &Task{ID: "7", ExpID: 12, Store: s.m.pg.Exploration(12)}
	task.setLLMState(&pidReady, &pidReady, []int64{pidReady}, 0, "ready", "")
	if !s.engine.ReadyFor(task) {
		t.Fatal("ReadyFor expected true for ready CLI profile")
	}

	// Missing binary profile makes ReadyFor return false.
	taskMissing := &Task{ID: "8", ExpID: 13, Store: s.m.pg.Exploration(13)}
	taskMissing.setLLMState(&pidMissing, &pidMissing, []int64{pidMissing}, 0, "ready", "")
	if s.engine.ReadyFor(taskMissing) {
		t.Fatal("ReadyFor expected false for CLI profile with missing binary")
	}

	// Global env CLI fallback readiness check rejects missing binary.
	s.cfgMu.Lock()
	s.llmOn = true
	s.llmProf = "env-cli"
	s.llmCfg = agent.Config{Format: agent.FormatClaudeCLI, Model: "sonnet", BaseURL: missingBin}
	s.cfgMu.Unlock()
	st.tasks[9] = &memSrvTask{id: 9, expID: 14, description: "env fallback", goal: "test", status: "running"}
	taskEnv := &Task{ID: "9", ExpID: 14, Store: s.m.pg.Exploration(14)}
	roleRes, err := s.resolveTaskRoleLLM(taskEnv, "worker")
	if err != nil {
		t.Fatal(err)
	}
	if roleRes.Available || !strings.Contains(roleRes.Reason, "not accessible") {
		t.Fatalf("expected missing global CLI env resolution to be unavailable, got %+v", roleRes)
	}

	// targetGateEndpoints & llmHost never treat CLI local binary paths as HTTP URLs.
	s.cfgMu.Lock()
	s.llmCfg = agent.Config{Format: agent.FormatClaudeCLI, BaseURL: `C:\tools\claude.exe`}
	s.cfgMu.Unlock()
	if host := s.llmHost(); host != "" {
		t.Fatalf("llmHost for CLI profile = %q, want empty", host)
	}
	eps := s.m.targetGateEndpoints()
	if slices.Contains(eps, fakeBin) || slices.Contains(eps, missingBin) {
		t.Fatalf("targetGateEndpoints leaked CLI binary paths: %v", eps)
	}
}

func TestCLIConfigConversionUnwrapAndTaskFailover(t *testing.T) {
	s, st := newMemTestServer(t)
	fakeBin := buildServerFakeCLIBinary(t)

	pidClaude, err := s.m.pg.SaveProfile(&db.LLMProfile{
		Name: "cli-claude-1", Format: "claude-cli", Model: "sonnet", BaseURL: fakeBin,
		ReasoningEffort: "high", ThinkingType: "enabled", Proxy: "http://proxy.local:3128", Streaming: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	pidAgy, err := s.m.pg.SaveProfile(&db.LLMProfile{
		Name: "cli-agy-2", Format: "agy-cli", Model: "gemini-4-hydrogen-high", BaseURL: fakeBin,
		ReasoningEffort: "medium", Streaming: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	pidHTTP, err := s.m.pg.SaveProfile(&db.LLMProfile{
		Name: "http-fallback-3", Format: "anthropic", Model: "claude-opus-4-8",
		BaseURL: "https://api.example.invalid", APIKey: "sk-test-123", Streaming: true,
	})
	if err != nil {
		t.Fatal(err)
	}

	// 1. Config conversion preserves ProfileName, BaseURL binary override, Format, ReasoningEffort, ThinkingType, Proxy.
	cfgClaude, ok := s.loadProfileConfig(pidClaude)
	if !ok || cfgClaude.Format != agent.FormatClaudeCLI || cfgClaude.ProfileName != "cli-claude-1" ||
		cfgClaude.BaseURL != fakeBin || cfgClaude.ReasoningEffort != "high" ||
		cfgClaude.ThinkingType != "enabled" || cfgClaude.Proxy != "http://proxy.local:3128" {
		t.Fatalf("unexpected loadProfileConfig(claude): ok=%v cfg=%+v", ok, cfgClaude)
	}
	cfgAgy, ok := s.loadProfileConfig(pidAgy)
	if !ok || cfgAgy.Format != agent.FormatAgyCLI || cfgAgy.ProfileName != "cli-agy-2" || cfgAgy.BaseURL != fakeBin {
		t.Fatalf("unexpected loadProfileConfig(agy): ok=%v cfg=%+v", ok, cfgAgy)
	}

	// 2. Unwrap through llmrec.Recorder -> sidequestion.Bind -> llmpool.Pool.
	rawProv, err := cfgClaude.NewProvider()
	if err != nil {
		t.Fatal(err)
	}
	recProv := llmrec.Wrap(rawProv, s.m.pg, cfgClaude.Model, cfgClaude.ProfileName, cfgClaude.ThinkingType, cfgClaude.ReasoningEffort, func() bool { return false })
	if recProv.Unwrap() != rawProv {
		t.Fatalf("Recorder.Unwrap() = %v, want %v", recProv.Unwrap(), rawProv)
	}
	boundProv := bindSideProvider(recProv, cfgClaude, pidClaude, cfgClaude.ProfileName)
	unwrapper, isUnwrap := boundProv.(interface{ Unwrap() llm.Provider })
	if !isUnwrap || unwrapper.Unwrap() != recProv {
		t.Fatalf("sidequestion boundProvider did not unwrap cleanly")
	}

	rawAgy, err := cfgAgy.NewProvider()
	if err != nil {
		t.Fatal(err)
	}
	recAgy := llmrec.Wrap(rawAgy, s.m.pg, cfgAgy.Model, cfgAgy.ProfileName, "", "", func() bool { return false })
	pool := llmpool.New([]*llmpool.Member{
		{ID: pidClaude, Name: "cli-claude-1", Model: "sonnet", Format: "claude-cli", Rank: llmpool.RankActive, Prov: boundProv},
		{ID: pidAgy, Name: "cli-agy-2", Model: "gemini-4-hydrogen-high", Format: "agy-cli", Rank: 10, Prov: recAgy},
	}, s.llmHealth)
	if pool.Unwrap() != boundProv {
		t.Fatalf("Pool.Unwrap() = %v, want claude head", pool.Unwrap())
	}
	// Quota error in pool trips hard on 1st occurrence and switches Pool.Unwrap() to agy-cli-2.
	if !isQuotaExhaustedError(errors.New("Claude Code CLI quota exhausted: 429 rate limit exceeded")) {
		t.Fatal("expected CLI quota error to be classified as quota exhaustion")
	}
	if isQuotaExhaustedError(errors.New("HTTP status 429: rate limit exceeded")) {
		t.Fatal("ordinary HTTP 429 rate limit must not be classified as quota exhaustion")
	}
	pool.Health().Trip(pidClaude, "Claude Code CLI quota exhausted: You have hit your usage limit", true)
	if pool.Unwrap() != recAgy {
		t.Fatalf("after trip, Pool.Unwrap() = %v, want agy fallback", pool.Unwrap())
	}
	// Also test real Stream() failover on CLI quota error through llmpool.Pool!
	s.llmHealth.Reset(pidClaude)
	quotaCfg := cfgClaude
	quotaCfg.Model = "model-quota-result"
	quotaRaw, _ := quotaCfg.NewProvider()
	okAgyCfg := cfgAgy
	okAgyCfg.Model = "gemini-4-hydrogen-high"
	okAgyRaw, _ := okAgyCfg.NewProvider()
	pool2 := llmpool.New([]*llmpool.Member{
		{ID: pidClaude, Name: "cli-claude-1", Model: quotaCfg.Model, Format: "claude-cli", Rank: llmpool.RankActive, Prov: quotaRaw},
		{ID: pidAgy, Name: "cli-agy-2", Model: okAgyCfg.Model, Format: "agy-cli", Rank: 10, Prov: okAgyRaw},
	}, s.llmHealth)
	var gotText strings.Builder
	for ev, err := range pool2.Stream(context.Background(), llm.CompletionRequest{Messages: []llm.Message{llm.UserText("ping")}}) {
		if err != nil {
			t.Fatalf("pool2 Stream failover failed: %v", err)
		}
		if ev.Type == llm.SETextDelta {
			gotText.WriteString(ev.Text)
		}
	}
	if strings.TrimSpace(gotText.String()) != "OK" {
		t.Fatalf("pool2 Stream failover reply = %q, want OK", gotText.String())
	}
	if !s.llmHealth.IsOpen(pidClaude) {
		t.Fatal("expected CLI quota error inside llmpool to trip breaker immediately (hard failure)")
	}

	// 3. Task failover chain [claude-cli-1 -> agy-cli-2 -> anthropic-http-3] & per-role bindings.
	st.tasks[7] = &memSrvTask{
		id: 7, expID: 12, description: "failover", goal: "test", status: "running",
		llmProfileID: &pidClaude, activeLLMProfileID: &pidClaude, llmChainRevision: 1,
		chain: []db.TaskLLMProfile{
			{ProfileID: pidClaude, Position: 0, Status: "ready"},
			{ProfileID: pidAgy, Position: 1, Status: "ready"},
			{ProfileID: pidHTTP, Position: 2, Status: "ready"},
		},
	}
	tk := &Task{ID: "7", ExpID: 12, Store: s.m.pg.Exploration(12)}
	tk.setLLMState(&pidClaude, &pidClaude, []int64{pidClaude, pidAgy, pidHTTP}, 1, "ready", "")
	s.m.tasks["7"] = tk

	// Bind planner to agy-cli while worker follows task chain (starting on claude-cli).
	st.agents["planner"].llmProfileID = &pidAgy
	b := s.agentsForTask(tk)

	if cfg, ok := b.plannerRuntime.ResolveCLIConfig(context.Background()); !ok || cfg.Format != agent.FormatAgyCLI || cfg.ProfileName != "cli-agy-2" {
		t.Fatalf("planner role binding ResolveCLIConfig = (%+v, %v)", cfg, ok)
	}
	if cfg, ok := b.workerRuntime.ResolveCLIConfig(context.Background()); !ok || cfg.Format != agent.FormatClaudeCLI || cfg.ProfileName != "cli-claude-1" {
		t.Fatalf("worker initial ResolveCLIConfig = (%+v, %v)", cfg, ok)
	}
	// Pin override stays within task chain.
	pinCtx := withLLMProfilePin(context.Background(), pidAgy)
	if cfg, ok := b.workerRuntime.ResolveCLIConfig(pinCtx); !ok || cfg.Format != agent.FormatAgyCLI {
		t.Fatalf("worker pinned ResolveCLIConfig = (%+v, %v)", cfg, ok)
	}

	// Advance task chain from claude-cli-1 -> agy-cli-2 via Engine's wired CLI handlers on CLI quota error.
	snap1 := s.engine.beginCLIAttempt(context.Background(), tk, "worker")
	retry, err := s.engine.finishCLIAttempt(snap1, false, errors.New("Claude Code CLI quota exhausted: You have hit your usage limit"))
	if !retry || err != nil {
		t.Fatalf("engine.finishCLIAttempt(claude->agy) = (retry=%v, err=%v), want (true, nil)", retry, err)
	}
	if cfg, ok := b.workerRuntime.ResolveCLIConfig(context.Background()); !ok || cfg.Format != agent.FormatAgyCLI || cfg.ProfileName != "cli-agy-2" {
		t.Fatalf("after 1st failover ResolveCLIConfig = (%+v, %v), want agy-cli", cfg, ok)
	}

	// Advance task chain from agy-cli-2 -> anthropic-http-3.
	snap2 := s.beginTaskCLIAttempt(context.Background(), tk, "worker")
	retry, err = finishCLIAttempt(snap2, false, errors.New("Antigravity CLI quota exhausted: HTTP 429 rate limit"))
	if !retry || err != nil {
		t.Fatalf("finishCLIAttempt(agy->http) = (retry=%v, err=%v), want (true, nil)", retry, err)
	}
	// Active profile is now HTTP -> ResolveCLIConfig returns false!
	if cfg, ok := b.workerRuntime.ResolveCLIConfig(context.Background()); ok {
		t.Fatalf("expected ResolveCLIConfig=false when chain rotated to HTTP profile, got %+v", cfg)
	}

	// Exhausting final profile returns chainExhausted taskLLMError.
	st.tasks[8] = &memSrvTask{
		id: 8, expID: 13, description: "exhaust", goal: "test", status: "running",
		llmProfileID: &pidClaude, activeLLMProfileID: &pidClaude, llmChainRevision: 1,
		chain: []db.TaskLLMProfile{{ProfileID: pidClaude, Position: 0, Status: "ready"}},
	}
	tk8 := &Task{ID: "8", ExpID: 13, Store: s.m.pg.Exploration(13)}
	s.m.tasks["8"] = tk8
	snap8 := s.beginTaskCLIAttempt(context.Background(), tk8, "worker")
	retry, err = finishCLIAttempt(snap8, false, errors.New("Claude Code CLI quota exhausted: billing"))
	if retry || !isTaskLLMChainExhausted(err) {
		t.Fatalf("expected chainExhausted=true retry=false, got retry=%v err=%v", retry, err)
	}
}

func TestCLIServerBridgeLifecycleGoalsMainAndChatAgents(t *testing.T) {
	s, st := newMemTestServer(t)
	fakeBin := buildServerFakeCLIBinary(t)

	if s.mcpBridge == nil {
		t.Fatal("expected shared mcpBridge to be initialized")
	}
	oldAugment, oldResolve, oldPrompt := agent.ToolAugment(), agent.ToolResolve(), agent.PromptOverride()
	agent.SetToolResolve(nil)
	agent.SetPromptOverride(nil)
	agent.SetToolAugment(func(_ context.Context, agentKey string) ([]actool.CoreTool, agent.DeferredInfo, func()) {
		if agentKey == "chat_bot" {
			toolProbe := actool.Build(actool.Spec{
				Name:        "retester_probe",
				Description: "probe finding",
				Schema: map[string]any{
					"type":       "object",
					"properties": map[string]any{"finding_id": map[string]any{"type": "integer"}},
				},
				Permissions: func(context.Context, json.RawMessage, permission.Context) permission.Decision {
					return permission.Allowed()
				},
				Run: func(_ context.Context, in json.RawMessage, _ *actool.ToolContext) (actool.Result, error) {
					return actool.Text("probed:" + string(in)), nil
				},
			})
			return []actool.CoreTool{toolProbe}, agent.DeferredInfo{}, func() {}
		}
		return nil, agent.DeferredInfo{}, func() {}
	})
	t.Cleanup(func() {
		agent.SetToolAugment(oldAugment)
		agent.SetToolResolve(oldResolve)
		agent.SetPromptOverride(oldPrompt)
	})

	// Profiles for Goals, MainAgent, ChatAgent (exercising fakecli MCP tool calls).
	pidGoals, err := s.m.pg.SaveProfile(&db.LLMProfile{
		Name: "goals-claude", Format: "claude-cli", Model: "model-agent-goals", BaseURL: fakeBin, Streaming: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	pidMain, err := s.m.pg.SaveProfile(&db.LLMProfile{
		Name: "main-agy", Format: "agy-cli", Model: "model-agent-mainagent", BaseURL: fakeBin, Streaming: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	pidChat, err := s.m.pg.SaveProfile(&db.LLMProfile{
		Name: "chat-claude", Format: "claude-cli", Model: "model-agent-chat", BaseURL: fakeBin, Streaming: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	pidJudge, err := s.m.pg.SaveProfile(&db.LLMProfile{
		Name: "judge-claude", Format: "claude-cli", Model: "sonnet", BaseURL: fakeBin, Streaming: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.m.pg.SetActiveProfile(pidJudge); err != nil {
		t.Fatal(err)
	}
	s.reapplyActiveProfile()

	// Verify global chatAgent received SetCLIOptions and resolves CLI via Unwrap().
	s.cfgMu.Lock()
	globalCA := s.chatAgent
	s.cfgMu.Unlock()
	if globalCA == nil {
		t.Fatal("expected reapplyActiveProfile to install global chatAgent for active CLI profile")
	}
	if host := s.llmHost(); host != "" {
		t.Fatalf("expected s.llmHost() == \"\" for CLI profile, got %q", host)
	}

	// Setup task #20 (exp 200) for Goals + MainAgent.
	const taskNum int64 = 20
	const expID int64 = 200
	st.addOriginFact(expID)
	st.tasks[taskNum] = &memSrvTask{
		id: taskNum, expID: expID, description: "test app", goal: "test /admin and /api", status: "running",
		llmProfileID: &pidGoals, activeLLMProfileID: &pidGoals, llmChainRevision: 1,
		chain: []db.TaskLLMProfile{{ProfileID: pidGoals, Position: 0, Status: "ready"}},
	}
	tk := &Task{ID: "20", ExpID: expID, Description: "test app", Goal: "test /admin and /api", Store: s.m.pg.Exploration(expID)}
	s.m.tasks["20"] = tk

	// Bind mainagent to pidMain (agy-cli), chat_bot to pidChat (claude-cli).
	st.agents["mainagent"].llmProfileID = &pidMain
	st.agents["chat_bot"].llmProfileID = &pidChat

	// 1. Run createGoals -> must use DecomposeGoalsWithCLI + shared mcpBridge + UsagePG!
	var goalActivities []db.Activity
	specs := s.createGoals(context.Background(), tk, func(a db.Activity) {
		goalActivities = append(goalActivities, a)
	})
	if len(specs) != 2 {
		t.Fatalf("createGoals specs = %+v, want 2 decomposed goals via fakecli", specs)
	}
	goalsInStore := st.snapshotNodes(expID, db.KindGoal)
	if len(goalsInStore) != 2 {
		t.Fatalf("goals persisted in store = %d, want 2", len(goalsInStore))
	}

	// 2. Run MainAgent.Chat via agentsForTask -> must execute add_hint via s.mcpBridge!
	b := s.agentsForTask(tk)
	var mainActivities []db.Activity
	ctxMain := transcript.WithSessionID(context.Background(), "exp200-main")
	mainHooks := guard.NewWithInterceptor(s.m.interceptor).Hooks()
	reply, err := b.main.Chat(ctxMain, taskNum, 0, s.m.Assets(), tk.Store, tk.Goal, "focus on auth", mainHooks, func(a db.Activity) {
		mainActivities = append(mainActivities, a)
	}, nil, nil, nil, nil)
	if err != nil {
		t.Fatalf("MainAgent.Chat failed: %v", err)
	}
	if !strings.Contains(reply, "main fresh:") {
		t.Fatalf("unexpected MainAgent reply: %q", reply)
	}
	hints := st.snapshotNodes(expID, db.KindHint)
	if len(hints) == 0 {
		t.Fatal("expected MainAgent add_hint over server mcpBridge to persist a hint node")
	}

	// 3. Run ChatAgent via resolveChatAgent (and triggered run precheck) -> must execute retester_probe via s.mcpBridge!
	if s.resolveChatAgent(&db.Conversation{AgentKey: "chat_bot"}) == nil {
		t.Fatal("expected resolveChatAgent to resolve bound CLI agent")
	}
	conv, err := s.m.pg.CreateConversationGated("chat_bot", "CLI chat", nil, false)
	if err != nil {
		t.Fatal(err)
	}
	s.runConversationSync(conv, "probe finding 42", s.convBusyKey(conv.ID))
	st.mu.Lock()
	var sawProbeResult bool
	for _, act := range st.convActs {
		if act.convID == conv.ID && strings.Contains(act.detail, "chat fresh:") {
			sawProbeResult = true
		}
	}
	st.mu.Unlock()
	if !sawProbeResult {
		t.Fatalf("expected conversation activity from ChatAgent CLI run, got %+v", st.convActs)
	}

	// 4. Verify UsagePG recorded usage rows for goals, main, and chat with ProfileName!
	usages := st.snapshotUsages()
	var sawGoalsUsage, sawMainUsage, sawChatUsage bool
	for _, u := range usages {
		if u.worker == "goals" && u.profileName == "goals-claude" && u.status == "ok" {
			sawGoalsUsage = true
		}
		if u.worker == "main" && u.profileName == "main-agy" && u.status == "ok" {
			sawMainUsage = true
		}
		if u.worker == "chat" && u.profileName == "chat-claude" && u.status == "ok" {
			sawChatUsage = true
		}
	}
	if !sawGoalsUsage || !sawMainUsage || !sawChatUsage {
		t.Fatalf("missing CLI usage rows (goals=%v main=%v chat=%v): %+v", sawGoalsUsage, sawMainUsage, sawChatUsage, usages)
	}

	// 5. Verify intercept judgeProvider & sideProvider work with keyless CLI profile.
	judgeProv, jpid, err := s.judgeProvider(pidJudge)
	if err != nil || jpid != pidJudge || judgeProv == nil {
		t.Fatalf("judgeProvider(CLI) = (%v, %d, %v)", judgeProv, jpid, err)
	}
	sideP, err := s.sideProvider(sideModel(agent.Config{
		Format: agent.FormatClaudeCLI, BaseURL: fakeBin, Model: "sonnet", Stream: true,
	}, pidJudge, "judge-claude"))
	if err != nil || sideP == nil {
		t.Fatalf("sideProvider(CLI) failed: %v", err)
	}
	ans, err := (sidequestion.SideQuestionService{Provider: sideP}).Answer(
		context.Background(),
		llm.CompletionRequest{Messages: []llm.Message{llm.UserText("ping")}},
		true,
		nil,
	)
	if err != nil || strings.TrimSpace(ans.Text) != "OK" {
		t.Fatalf("sidequestion Answer via CLI failed: ans=%+v err=%v", ans, err)
	}

	// 6. Verify Server.Close() cleanly shuts down mcpBridge and prevents re-creation.
	bridge := s.mcpBridge
	tok := bridge.RegisterRun(mcpbridge.RunConfig{})
	url := bridge.MCPURL(tok)
	resp, err := http.Post(url, "application/json", strings.NewReader(`{"jsonrpc":"2.0","id":1,"method":"ping"}`))
	if err != nil {
		t.Fatalf("bridge should respond before Close: %v", err)
	}
	_ = resp.Body.Close()

	if err := s.Close(); err != nil {
		t.Fatalf("Server.Close() failed: %v", err)
	}
	if s.ensureMCPBridge() != nil {
		t.Fatal("expected ensureMCPBridge() to return nil after Close()")
	}
	if _, err := http.Post(url, "application/json", strings.NewReader(`{}`)); err == nil {
		t.Fatal("expected bridge HTTP listener to be closed after Server.Close()")
	}
	_ = goalActivities
	_ = mainActivities
	_ = harness.ReasonCompleted
	_ = atomic.Int64{}
}
