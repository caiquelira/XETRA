package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/Autumn-27/norma/harness"
	"github.com/Autumn-27/norma/llm"
)

const (
	// FormatClaudeCLI selects the headless Claude Code CLI backend (`claude`).
	FormatClaudeCLI llm.Format = "claude-cli"
	// FormatAgyCLI selects the headless Antigravity CLI backend (`agy`).
	FormatAgyCLI llm.Format = "agy-cli"
)

// CLIInvocationParams carries the inputs required by a CLIBackend to construct
// a headless CLI invocation and its ephemeral configuration files.
type CLIInvocationParams struct {
	Config       Config
	Prompt       string
	SystemPrompt string
	WorkDir      string
	ExtraDirs    []string
	MCPURL       string
	HookURL      string
	HookBin      string
	HookToken    string
	SessionKey   string
	SessionID    string
	IsResume     bool
	MaxTurns     int
	DisableTools bool
	Env          map[string]string
}

// CLIInvocation describes a concrete subprocess command and its ephemeral files.
type CLIInvocation struct {
	Binary  string
	Args    []string
	Dir     string
	Stdin   string
	Env     []string
	Cleanup func()
}

// CLIEvent represents a normalized event emitted from a CLI `stream-json` line.
type CLIEvent struct {
	Kind      string          `json:"kind"`
	SessionID string          `json:"session_id,omitempty"`
	Text      string          `json:"text,omitempty"`
	ToolID    string          `json:"tool_id,omitempty"`
	ToolName  string          `json:"tool_name,omitempty"`
	ToolInput json.RawMessage `json:"tool_input,omitempty"`
	IsError   bool            `json:"is_error,omitempty"`
	Usage     *llm.Usage      `json:"usage,omitempty"`
	Result    *CLIResult      `json:"result,omitempty"`
}

// CLIResult summarizes the terminal outcome of a headless CLI invocation.
type CLIResult struct {
	SessionID        string                 `json:"session_id,omitempty"`
	Text             string                 `json:"text,omitempty"`
	Reason           harness.TerminalReason `json:"reason"`
	ErrorText        string                 `json:"error_text,omitempty"`
	CostUSD          float64                `json:"cost_usd,omitempty"`
	NumTurns         int                    `json:"num_turns,omitempty"`
	Usage            llm.Usage              `json:"usage"`
	IsError          bool                   `json:"is_error,omitempty"`
	IsQuotaError     bool                   `json:"is_quota_error,omitempty"`
	IsSessionMissing bool                   `json:"is_session_missing,omitempty"`
}

// CLIStreamParser parses NDJSON `stream-json` lines sequentially for one run,
// allowing backends that emit incremental step updates (such as `agy`) to
// deduplicate per-step events without sharing mutable state on the backend.
type CLIStreamParser interface {
	ParseLine(line []byte) ([]CLIEvent, error)
}

// CLIBackend adapts a headless coding-agent CLI (such as `claude` or `agy`) to
// XETRA's agent runner and provider abstractions.
type CLIBackend interface {
	Format() llm.Format
	Name() string
	DefaultBinary() string
	DefaultModel() string
	CheckReady(cfg Config) error
	BuildInvocation(ctx context.Context, params CLIInvocationParams) (*CLIInvocation, error)
	NewParser() CLIStreamParser
	ParseLine(line []byte) ([]CLIEvent, error)
	ParseFinal(stdout []byte) (*CLIResult, error)
	ListModels(ctx context.Context, cfg Config) ([]string, error)
}

var (
	cliBackendsMu sync.RWMutex
	cliBackends   = make(map[llm.Format]CLIBackend)
)

// RegisterCLIBackend registers a CLIBackend under its Format().
func RegisterCLIBackend(b CLIBackend) {
	if b == nil {
		panic("agent: RegisterCLIBackend called with nil backend")
	}
	f := b.Format()
	if f == "" {
		panic("agent: RegisterCLIBackend called with empty format")
	}
	cliBackendsMu.Lock()
	defer cliBackendsMu.Unlock()
	cliBackends[f] = b
}

// LookupCLIBackend returns the registered CLIBackend for format, if any.
func LookupCLIBackend(format llm.Format) (CLIBackend, bool) {
	cliBackendsMu.RLock()
	defer cliBackendsMu.RUnlock()
	b, ok := cliBackends[format]
	return b, ok
}

// IsCLIFormat reports whether format corresponds to a registered CLIBackend.
func IsCLIFormat(format llm.Format) bool {
	_, ok := LookupCLIBackend(format)
	return ok
}

// CLIFormats returns all registered CLI backend formats in sorted order.
func CLIFormats() []llm.Format {
	cliBackendsMu.RLock()
	defer cliBackendsMu.RUnlock()
	out := make([]llm.Format, 0, len(cliBackends))
	for f := range cliBackends {
		out = append(out, f)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

func resolveCLIBinary(cfg Config, envVar, defaultBin string) string {
	if override := strings.TrimSpace(cfg.BaseURL); override != "" {
		return override
	}
	if envVar != "" {
		if envBin := strings.TrimSpace(os.Getenv(envVar)); envBin != "" {
			return envBin
		}
	}
	return defaultBin
}

func checkCLIBinaryReady(backendName, bin string) error {
	bin = strings.TrimSpace(bin)
	if bin == "" {
		return fmt.Errorf("%s CLI binary path is empty", backendName)
	}
	if filepath.IsAbs(bin) || strings.ContainsAny(bin, `/\`) {
		info, err := os.Stat(bin)
		if err != nil {
			return fmt.Errorf("%s CLI binary %q not accessible: %w", backendName, bin, err)
		}
		if info.IsDir() {
			return fmt.Errorf("%s CLI binary %q is a directory", backendName, bin)
		}
		return nil
	}
	if _, err := exec.LookPath(bin); err != nil {
		return fmt.Errorf("%s CLI binary %q not found in PATH: %w", backendName, bin, err)
	}
	return nil
}

func resolveHookBin(explicit string) string {
	if bin := strings.TrimSpace(explicit); bin != "" {
		return bin
	}
	if exe, err := os.Executable(); err == nil && exe != "" {
		return exe
	}
	return "artex"
}

func buildHookCommand(hookBin, cliName, event string) string {
	quoted := `"` + strings.ReplaceAll(hookBin, `"`, `\"`) + `"`
	return fmt.Sprintf("%s hook --cli %s --event %s", quoted, cliName, event)
}

func hookEnvList(hookURL, hookToken, cliName string, extra map[string]string) []string {
	var env []string
	if hookURL != "" {
		env = append(env,
			"XETRA_BRIDGE_URL="+hookURL,
			"XETRA_BRIDGE_TOKEN="+hookToken,
			"XETRA_HOOK_CLI="+cliName,
		)
	}
	if len(extra) > 0 {
		keys := make([]string, 0, len(extra))
		for k := range extra {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			env = append(env, k+"="+extra[k])
		}
	}
	return env
}

// isCLIQuotaError reports whether a CLI error message indicates account quota,
// billing/credit exhaustion, or HTTP 429 / rate-limit rejection.
func isCLIQuotaError(msg string) bool {
	if strings.TrimSpace(msg) == "" {
		return false
	}
	if IsQuotaExhaustedMessage(msg) {
		return true
	}
	lower := strings.ToLower(msg)
	if nonFailoverHTTPStatus.MatchString(lower) {
		return false
	}
	markers := []string{
		"http 429",
		"status 429",
		"status code 429",
		"429 too many requests",
		"rate_limit",
		"rate limit",
		"too many requests",
		"hit your limit",
		"usage limit",
		"limit reached",
		"exceeded your limit",
		"overloaded_error",
		"billing",
		"quota",
	}
	for _, m := range markers {
		if strings.Contains(lower, m) {
			return true
		}
	}
	return false
}

// isCLISessionMissingError reports whether a CLI error message indicates that
// the requested resume session or conversation ID does not exist.
func isCLISessionMissingError(msg string) bool {
	lower := strings.ToLower(strings.TrimSpace(msg))
	if lower == "" {
		return false
	}
	markers := []string{
		"no conversation found",
		"conversation not found",
		"could not find conversation",
		"invalid conversation",
		"session not found",
		"no session found",
		"could not find session",
		"invalid session",
		"no such session",
	}
	for _, m := range markers {
		if strings.Contains(lower, m) {
			return true
		}
	}
	return false
}

func makeCleanupFunc(fn func()) func() {
	var once sync.Once
	return func() {
		once.Do(fn)
	}
}
