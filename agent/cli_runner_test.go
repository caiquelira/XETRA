package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
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

	"github.com/Autumn-27/artex/db"
	"github.com/Autumn-27/artex/intercept"
	"github.com/Autumn-27/artex/llmrec"
	"github.com/Autumn-27/artex/mcpbridge"
	"github.com/Autumn-27/norma/harness"
	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/permission"
	actool "github.com/Autumn-27/norma/tool"
)

var (
	fakeCLIBinOnce sync.Once
	fakeCLIBinPath string
	fakeCLIBinErr  error
)

func buildFakeCLIBinary(t *testing.T) string {
	t.Helper()
	fakeCLIBinOnce.Do(func() {
		_, thisFile, _, ok := runtime.Caller(0)
		if !ok {
			fakeCLIBinErr = errors.New("runtime.Caller failed")
			return
		}
		agentDir := filepath.Dir(thisFile)
		tmpDir, err := os.MkdirTemp("", "artex-fakecli-*")
		if err != nil {
			fakeCLIBinErr = err
			return
		}
		binName := "fakecli"
		if runtime.GOOS == "windows" {
			binName += ".exe"
		}
		outPath := filepath.Join(tmpDir, binName)
		cmd := exec.Command("go", "build", "-o", outPath, "./testdata/fakecli")
		cmd.Dir = agentDir
		if out, err := cmd.CombinedOutput(); err != nil {
			fakeCLIBinErr = fmt.Errorf("build fakecli: %w\n%s", err, string(out))
			return
		}
		fakeCLIBinPath = outPath
	})
	if fakeCLIBinErr != nil {
		t.Fatalf("failed to build fakecli test helper: %v", fakeCLIBinErr)
	}
	return fakeCLIBinPath
}

type fakeHookCall struct {
	Kind   string
	Tool   string
	Input  string
	Result string
	IsErr  bool
}

type testHookRunner struct {
	mu          sync.Mutex
	calls       []fakeHookCall
	blockSubstr string
	blockTool   string
	blockReason string
}

func (h *testHookRunner) PreToolUse(ctx context.Context, name string, input []byte) (bool, string, []byte) {
	_ = intercept.WithCall(ctx, name, input)
	h.mu.Lock()
	defer h.mu.Unlock()
	inStr := string(input)
	h.calls = append(h.calls, fakeHookCall{Kind: "pre", Tool: name, Input: inStr})
	if h.blockTool != "" && name == h.blockTool {
		return true, h.blockReason, nil
	}
	if h.blockSubstr != "" && strings.Contains(inStr, h.blockSubstr) {
		return true, h.blockReason, nil
	}
	return false, "", nil
}

func (h *testHookRunner) PostToolUse(_ context.Context, name string, input, result []byte, isErr bool) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.calls = append(h.calls, fakeHookCall{
		Kind:   "post",
		Tool:   name,
		Input:  string(input),
		Result: string(result),
		IsErr:  isErr,
	})
}

func (h *testHookRunner) Stop(context.Context, []llm.Message) (bool, []string, string) {
	return false, nil, ""
}

func (h *testHookRunner) snapshot() []fakeHookCall {
	h.mu.Lock()
	defer h.mu.Unlock()
	out := make([]fakeHookCall, len(h.calls))
	copy(out, h.calls)
	return out
}

type traceEntry struct {
	IsAgy      bool     `json:"is_agy"`
	Args       []string `json:"args"`
	Model      string   `json:"model"`
	Effort     string   `json:"effort"`
	SessionID  string   `json:"session_id"`
	ResumeID   string   `json:"resume_id"`
	ConvID     string   `json:"conv_id"`
	Prompt     string   `json:"prompt"`
	System     string   `json:"system"`
	HTTPProxy  string   `json:"http_proxy"`
	HTTPSProxy string   `json:"https_proxy"`
	NoProxy    string   `json:"no_proxy"`
}

func readTraceEntries(t *testing.T, path string) []traceEntry {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read trace file %q: %v", path, err)
	}
	var out []traceEntry
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		var e traceEntry
		if err := json.Unmarshal([]byte(line), &e); err != nil {
			t.Fatalf("parse trace line %q: %v", line, err)
		}
		out = append(out, e)
	}
	return out
}

func TestProviderCLIFormatsAndEnv(t *testing.T) {
	bin := buildFakeCLIBinary(t)

	for _, tc := range []struct {
		format       llm.Format
		providerName string
		defaultModel string
	}{
		{FormatClaudeCLI, "claude-cli", "sonnet"},
		{FormatAgyCLI, "agy-cli", "gemini-4-hydrogen-high"},
	} {
		cfg := ConfigFrom(tc.providerName, "", bin, "", "")
		if !cfg.IsCLI() {
			t.Fatalf("%s: expected IsCLI()=true", tc.providerName)
		}
		if cfg.NeedsAPIKey() {
			t.Fatalf("%s: expected NeedsAPIKey()=false", tc.providerName)
		}
		if cfg.Provider() != tc.providerName {
			t.Fatalf("%s: Provider()=%q, want %q", tc.providerName, cfg.Provider(), tc.providerName)
		}
		if cfg.Model != tc.defaultModel {
			t.Fatalf("%s: default Model=%q, want %q", tc.providerName, cfg.Model, tc.defaultModel)
		}
		if !cfg.Stream {
			t.Fatalf("%s: expected Stream=true", tc.providerName)
		}
		if _, err := cfg.NewProvider(); err != nil {
			t.Fatalf("%s: NewProvider with valid bin failed: %v", tc.providerName, err)
		}
		missingCfg := ConfigFrom(tc.providerName, "", filepath.Join(t.TempDir(), "does-not-exist"), "", "")
		if _, err := missingCfg.NewProvider(); err == nil {
			t.Fatalf("%s: NewProvider with missing binary expected error", tc.providerName)
		}
	}

	if (Config{Format: llm.FormatAnthropic}).IsCLI() || (Config{Format: FormatOllama}).IsCLI() {
		t.Fatal("non-CLI format reported IsCLI()=true")
	}
	if !(Config{Format: llm.FormatAnthropic}).NeedsAPIKey() {
		t.Fatal("Anthropic expected NeedsAPIKey()=true")
	}
	if (Config{Format: FormatOllama}).NeedsAPIKey() {
		t.Fatal("Ollama expected NeedsAPIKey()=false")
	}

	// FromEnv works without ANTHROPIC_API_KEY or OPENAI_API_KEY for CLI formats.
	t.Setenv("ANTHROPIC_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("ARTEX_LLM_BASE_URL", "")
	t.Setenv("ARTEX_LLM_MODEL", "")

	t.Setenv("ARTEX_LLM_PROVIDER", "claude-cli")
	c1, ok := FromEnv()
	if !ok || c1.Format != FormatClaudeCLI || c1.Model != "sonnet" {
		t.Fatalf("FromEnv(claude-cli) = (%+v, %v)", c1, ok)
	}

	t.Setenv("ARTEX_LLM_PROVIDER", " AGY-CLI ")
	c2, ok := FromEnv()
	if !ok || c2.Format != FormatAgyCLI || c2.Model != "gemini-4-hydrogen-high" {
		t.Fatalf("FromEnv(agy-cli) = (%+v, %v)", c2, ok)
	}

	t.Setenv("ARTEX_LLM_PROVIDER", "claude-cli")
	t.Setenv("ARTEX_LLM_MODEL", "opus")
	t.Setenv("ARTEX_LLM_BASE_URL", bin)
	c3, ok := FromEnv()
	if !ok || c3.Model != "opus" || c3.BaseURL != bin {
		t.Fatalf("FromEnv(custom claude-cli) = (%+v, %v)", c3, ok)
	}
}

func TestCLICompletionProviderStreamCompleteAndUsage(t *testing.T) {
	bin := buildFakeCLIBinary(t)
	t.Setenv("CLAUDECODE", "1")
	t.Setenv("CLAUDE_CODE_ENTRYPOINT", "cli")
	t.Setenv("NO_PROXY", "10.0.0.0/8")

	for _, format := range []llm.Format{FormatClaudeCLI, FormatAgyCLI} {
		t.Run(string(format), func(t *testing.T) {
			cfg := Config{
				Format:          format,
				BaseURL:         bin,
				Model:           "model-stream-chunks",
				ProfileName:     "prof-" + string(format),
				ReasoningEffort: "high",
				Stream:          true,
			}
			prov, err := cfg.NewProvider()
			if err != nil {
				t.Fatalf("NewProvider: %v", err)
			}

			var sink []llm.Usage
			ctx := llmrec.WithUsageSink(context.Background(), func(u llm.Usage) {
				sink = append(sink, u)
			})

			// Stream mode: collects thinking & text deltas and usage once.
			var thinkBuf, textBuf strings.Builder
			var streamUsage llm.Usage
			var stopReason string
			for ev, err := range prov.Stream(ctx, llm.CompletionRequest{
				System:   []string{"sys line 1", "sys line 2"},
				Messages: []llm.Message{llm.UserText("ping")},
			}) {
				if err != nil {
					t.Fatalf("Stream error: %v", err)
				}
				switch ev.Type {
				case llm.SEThinkingDelta:
					thinkBuf.WriteString(ev.Text)
				case llm.SETextDelta:
					textBuf.WriteString(ev.Text)
				case llm.SEMessageDelta:
					stopReason = ev.StopReason
					streamUsage = ev.Usage
				}
			}
			if thinkBuf.String() != "think-1 think-2" {
				t.Fatalf("thinking = %q, want %q", thinkBuf.String(), "think-1 think-2")
			}
			if textBuf.String() != "part-1 part-2" {
				t.Fatalf("text = %q, want %q", textBuf.String(), "part-1 part-2")
			}
			if stopReason != "end_turn" {
				t.Fatalf("stopReason = %q, want end_turn", stopReason)
			}
			wantUsage := llm.Usage{InputTokens: 120, OutputTokens: 45, CacheReadTokens: 20, CacheWriteTokens: 10}
			if format == FormatAgyCLI {
				wantUsage.CacheWriteTokens = 0
			}
			if streamUsage != wantUsage {
				t.Fatalf("streamUsage = %+v, want %+v", streamUsage, wantUsage)
			}
			if len(sink) != 1 || sink[0] != wantUsage {
				t.Fatalf("sink = %+v, want [%+v]", sink, wantUsage)
			}

			// Multi-turn Complete + Thinking="disabled" clears ReasoningEffort.
			echoCfg := cfg
			echoCfg.Model = "model-echo-prompt"
			echoProv, err := echoCfg.NewProvider()
			if err != nil {
				t.Fatal(err)
			}
			multiReq := llm.CompletionRequest{
				System:   []string{"System A", "System B"},
				Thinking: "disabled",
				Messages: []llm.Message{
					llm.UserText("first question"),
					{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
						{Type: llm.BlockThinking, Thinking: "secret thought"},
						llm.TextBlock("calling tool"),
						{Type: llm.BlockToolUse, ID: "tu1", Name: "lookup", Input: json.RawMessage(`{"k":"v"}`)},
					}},
					{Role: llm.RoleUser, Content: []llm.ContentBlock{
						llm.ToolResultText("tu1", "lookup ok", false),
						llm.TextBlock("follow up"),
					}},
				},
			}
			msg, sr, compUsage, err := echoProv.Complete(ctx, multiReq)
			if err != nil {
				t.Fatalf("Complete error: %v", err)
			}
			if sr != "end_turn" || compUsage.InputTokens != 50 {
				t.Fatalf("Complete stop=%q usage=%+v", sr, compUsage)
			}
			gotReply := msg.Text()
			if !strings.Contains(gotReply, "System A\n\nSystem B") ||
				!strings.Contains(gotReply, "User: first question") ||
				!strings.Contains(gotReply, "Assistant: calling tool") ||
				!strings.Contains(gotReply, `[Tool call: lookup({"k":"v"})]`) ||
				!strings.Contains(gotReply, "[Tool result]: lookup ok") ||
				!strings.Contains(gotReply, "follow up") ||
				!strings.HasSuffix(gotReply, "|EFFORT=") {
				t.Fatalf("unexpected multi-turn serialization reply: %q", gotReply)
			}
			if strings.Contains(gotReply, "secret thought") {
				t.Fatalf("internal thinking leaked into prompt serialization: %q", gotReply)
			}

			// Wrapped by llmrec.Recorder: sink must fire once, not twice.
			sink = nil
			rec := llmrec.Wrap(echoProv, nil, echoCfg.Model, echoCfg.ProfileName, "", "", func() bool { return false })
			if _, _, _, err := rec.Complete(ctx, llm.CompletionRequest{Messages: []llm.Message{llm.UserText("hi")}}); err != nil {
				t.Fatalf("Recorder.Complete failed: %v", err)
			}
			if len(sink) != 1 {
				t.Fatalf("expected 1 sink call when wrapped by Recorder, got %d", len(sink))
			}

			// Quota result & quota stderr both classify via IsQuotaExhaustedMessage.
			for _, qModel := range []string{"model-quota-result", "model-quota-stderr"} {
				qCfg := cfg
				qCfg.Model = qModel
				qProv, _ := qCfg.NewProvider()
				_, _, _, qErr := qProv.Complete(context.Background(), llm.CompletionRequest{Messages: []llm.Message{llm.UserText("hi")}})
				if qErr == nil || !IsQuotaExhaustedMessage(qErr.Error()) {
					t.Fatalf("%s: expected IsQuotaExhaustedMessage error, got %v", qModel, qErr)
				}
			}

			// Non-zero exit without result preserves stderr.
			errCfg := cfg
			errCfg.Model = "model-exit-error-no-result"
			errProv, _ := errCfg.NewProvider()
			_, _, _, nErr := errProv.Complete(context.Background(), llm.CompletionRequest{Messages: []llm.Message{llm.UserText("hi")}})
			if nErr == nil || !strings.Contains(nErr.Error(), "fatal stderr failure line") {
				t.Fatalf("expected stderr in error, got %v", nErr)
			}
		})
	}
}

func TestTestConnectionCLI(t *testing.T) {
	bin := buildFakeCLIBinary(t)

	for _, tc := range []struct {
		name   string
		format llm.Format
		stream bool
	}{
		{"claude_stream", FormatClaudeCLI, true},
		{"claude_complete", FormatClaudeCLI, false},
		{"agy_stream", FormatAgyCLI, true},
		{"agy_complete", FormatAgyCLI, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := Config{
				Format:  tc.format,
				BaseURL: bin,
				Stream:  tc.stream,
			}
			lat, reply, err := TestConnection(context.Background(), cfg)
			if err != nil {
				t.Fatalf("TestConnection failed: %v", err)
			}
			if reply != "OK" {
				t.Fatalf("reply = %q, want OK", reply)
			}
			if lat < 0 {
				t.Fatalf("unexpected negative latency %v", lat)
			}
		})
	}

	// Empty reply must fail TestConnection.
	_, _, err := TestConnection(context.Background(), Config{
		Format:  FormatClaudeCLI,
		BaseURL: bin,
		Model:   "model-empty-text",
		Stream:  true,
	})
	if err == nil || !strings.Contains(err.Error(), "no content") {
		t.Fatalf("expected empty content error, got %v", err)
	}
}

func TestCLIRunnerMCPAndHooks(t *testing.T) {
	bin := buildFakeCLIBinary(t)
	t.Setenv("CLAUDECODE", "nested-secret")

	for _, format := range []llm.Format{FormatClaudeCLI, FormatAgyCLI} {
		t.Run(string(format), func(t *testing.T) {
			workDir := t.TempDir()
			var echoCalls atomic.Int32
			type ctxKey string
			const markerKey ctxKey = "run_marker"

			echoTool := actool.Build(actool.Spec{
				Name:        "echo_tool",
				Description: "Echoes msg with working dir",
				Schema: map[string]any{
					"type":       "object",
					"properties": map[string]any{"msg": map[string]any{"type": "string"}},
					"required":   []string{"msg"},
				},
				Permissions: func(context.Context, json.RawMessage, permission.Context) permission.Decision {
					return permission.Allowed()
				},
				Run: func(ctx context.Context, in json.RawMessage, tc *actool.ToolContext) (actool.Result, error) {
					echoCalls.Add(1)
					var args struct {
						Msg string `json:"msg"`
					}
					_ = json.Unmarshal(in, &args)
					marker, _ := ctx.Value(markerKey).(string)
					wd := ""
					if tc != nil {
						wd = tc.WorkingDir
					}
					return actool.Text(args.Msg + "|" + marker + "|" + wd), nil
				},
			})
			blockedTool := actool.Build(actool.Spec{
				Name:        "blocked_tool",
				Description: "Blocked by PreToolUse hook",
				Schema:      map[string]any{"type": "object"},
				Permissions: func(context.Context, json.RawMessage, permission.Context) permission.Decision {
					return permission.Allowed()
				},
				Run: func(context.Context, json.RawMessage, *actool.ToolContext) (actool.Result, error) {
					t.Error("blocked_tool must not execute when PreToolUse denies")
					return actool.Text("should not run"), nil
				},
			})

			hooks := &testHookRunner{
				blockSubstr: "rm -rf /blocked",
				blockTool:   "blocked_tool",
				blockReason: "policy denied dangerous call",
			}

			var activities []db.Activity
			ctx := context.WithValue(context.Background(), markerKey, "ctx-ok")
			text, reason, err := runCLI(ctx, CLIRunOptions{
				Config: Config{
					Format:      format,
					BaseURL:     bin,
					Model:       "model-mcp-and-hooks",
					ProfileName: "mcp-prof",
				},
				SystemPrompt: "Use MCP tools.",
				Prompt:       "Run MCP and hook checks.",
				WorkDir:      workDir,
				Tools:        []actool.CoreTool{echoTool, blockedTool},
				Hooks:        hooks,
				HookBin:      bin,
				WorkerLabel:  "worker",
				Emit: func(a db.Activity) {
					activities = append(activities, a)
				},
			})
			if err != nil {
				t.Fatalf("runCLI failed: %v", err)
			}
			if reason != harness.ReasonCompleted {
				t.Fatalf("reason = %q, want %q", reason, harness.ReasonCompleted)
			}
			if echoCalls.Load() != 1 {
				t.Fatalf("echoCalls = %d, want 1", echoCalls.Load())
			}
			wantSnippet := "hello-mcp|ctx-ok|" + workDir
			if !strings.Contains(text, wantSnippet) {
				t.Fatalf("final text %q missing %q", text, wantSnippet)
			}

			// Verify both CLI hooks (Bash/run_command) and MCP PreToolUse/PostToolUse hooks ran.
			hCalls := hooks.snapshot()
			var sawBlockedBash, sawAllowedBashPost, sawEchoMCPPost, sawBlockedMCPPre bool
			for _, c := range hCalls {
				if c.Kind == "pre" && c.Tool == "Bash" && strings.Contains(c.Input, "rm -rf /blocked") {
					sawBlockedBash = true
				}
				if c.Kind == "post" && c.Tool == "Bash" && strings.Contains(c.Input, "echo allowed") {
					sawAllowedBashPost = true
				}
				if c.Kind == "post" && c.Tool == "echo_tool" && strings.Contains(c.Result, wantSnippet) && !c.IsErr {
					sawEchoMCPPost = true
				}
				if c.Kind == "pre" && c.Tool == "blocked_tool" {
					sawBlockedMCPPre = true
				}
			}
			if !sawBlockedBash || !sawAllowedBashPost || !sawEchoMCPPost || !sawBlockedMCPPre {
				t.Fatalf("missing expected hook calls (%v,%v,%v,%v): %+v",
					sawBlockedBash, sawAllowedBashPost, sawEchoMCPPost, sawBlockedMCPPre, hCalls)
			}

			// Verify coalesced activities & trailing text deduplication.
			var kinds []string
			var resultCount, textCount int
			for _, a := range activities {
				kinds = append(kinds, a.Kind)
				if a.Kind == "result" {
					resultCount++
					if a.IsError {
						t.Fatalf("unexpected error result activity: %+v", a)
					}
					if a.InputTokens == nil || *a.InputTokens != 220 || a.OutputTokens == nil || *a.OutputTokens != 65 {
						t.Fatalf("unexpected result token counts: %+v", a)
					}
				}
				if a.Kind == "text" {
					textCount++
				}
			}
			if resultCount != 1 {
				t.Fatalf("expected 1 result activity, got %d (kinds=%v)", resultCount, kinds)
			}
			if textCount != 0 {
				t.Fatalf("expected trailing text matching result.Text to be deduplicated, got textCount=%d (kinds=%v)", textCount, kinds)
			}
			if !slices.Contains(kinds, "thinking") || !slices.Contains(kinds, "tool_use") || !slices.Contains(kinds, "tool_result") {
				t.Fatalf("missing expected activity kinds: %v", kinds)
			}
		})
	}

	// Verify caller-supplied shared Bridge (*mcpbridge.Server) also works cleanly.
	t.Run("shared_bridge", func(t *testing.T) {
		srv, err := mcpbridge.NewServer()
		if err != nil {
			t.Fatalf("NewServer: %v", err)
		}
		defer srv.Close()

		workDir := t.TempDir()
		var called atomic.Int32
		echoTool := actool.Build(actool.Spec{
			Name:        "echo_tool",
			Description: "Echoes msg",
			Schema:      map[string]any{"type": "object"},
			Permissions: func(context.Context, json.RawMessage, permission.Context) permission.Decision {
				return permission.Allowed()
			},
			Run: func(_ context.Context, in json.RawMessage, _ *actool.ToolContext) (actool.Result, error) {
				called.Add(1)
				return actool.Text("shared-bridge-ok:" + string(in)), nil
			},
		})
		hooks := &testHookRunner{
			blockSubstr: "rm -rf /blocked",
			blockTool:   "blocked_tool",
			blockReason: "denied",
		}
		text, reason, err := runCLI(context.Background(), CLIRunOptions{
			Config: Config{
				Format:  FormatClaudeCLI,
				BaseURL: bin,
				Model:   "model-mcp-and-hooks",
			},
			Prompt:  "shared bridge test",
			WorkDir: workDir,
			Tools:   []actool.CoreTool{echoTool},
			Hooks:   hooks,
			Bridge:  srv,
			HookBin: bin,
		})
		if err != nil || reason != harness.ReasonCompleted || called.Load() != 1 || !strings.Contains(text, "shared-bridge-ok") {
			t.Fatalf("shared bridge run = (%q, %q, %v), called=%d", text, reason, err, called.Load())
		}
	})
}

func TestCLIRunnerWindowsCmdWrapper(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("Windows-specific .cmd executable compatibility check")
	}
	bin := buildFakeCLIBinary(t)
	dir := t.TempDir()
	cmdWrapper := filepath.Join(dir, "fake-claude.cmd")
	script := fmt.Sprintf("@echo off\r\n\"%s\" %%*\r\n", bin)
	if err := os.WriteFile(cmdWrapper, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	text, reason, err := runCLI(context.Background(), CLIRunOptions{
		Config: Config{
			Format:  FormatClaudeCLI,
			BaseURL: cmdWrapper,
			Model:   "sonnet",
		},
		Prompt:  "ping",
		WorkDir: dir,
	})
	if err != nil {
		t.Fatalf("runCLI via .cmd wrapper failed: %v", err)
	}
	if reason != harness.ReasonCompleted || text != "OK" {
		t.Fatalf("got (%q, %q), want (OK, completed)", text, reason)
	}
}

func TestCLIRunnerSessionResumeAndStaleRetry(t *testing.T) {
	bin := buildFakeCLIBinary(t)

	for _, format := range []llm.Format{FormatClaudeCLI, FormatAgyCLI} {
		t.Run(string(format), func(t *testing.T) {
			workDir := t.TempDir()
			traceFile := filepath.Join(t.TempDir(), "trace.jsonl")
			sessionKey := "exp12-worker-i3"

			run := func(prompt string) (string, []db.Activity, error) {
				var acts []db.Activity
				text, _, err := runCLI(context.Background(), CLIRunOptions{
					Config: Config{
						Format:  format,
						BaseURL: bin,
						Model:   "model-session-resume",
					},
					Prompt:     prompt,
					WorkDir:    workDir,
					SessionKey: sessionKey,
					Env:        []string{"ARTEX_FAKE_TRACE_FILE=" + traceFile},
					Emit: func(a db.Activity) {
						acts = append(acts, a)
					},
				})
				return text, acts, err
			}

			// Run 1: fresh session persisted to cli-sessions/exp12-worker-i3.json.
			text1, _, err := run("step 1")
			if err != nil {
				t.Fatalf("run 1 failed: %v", err)
			}
			if !strings.HasPrefix(text1, "fresh:") {
				t.Fatalf("run 1 text = %q, want fresh:<id>", text1)
			}
			firstID := strings.TrimPrefix(text1, "fresh:")

			store := NewCLISessionStore(filepath.Join(workDir, "cli-sessions"))
			rec, ok, err := store.Load(sessionKey)
			if err != nil || !ok || rec.SessionID != firstID {
				t.Fatalf("stored session mismatch: rec=%+v ok=%v err=%v", rec, ok, err)
			}

			// Run 2: resumes existing session.
			text2, _, err := run("step 2")
			if err != nil {
				t.Fatalf("run 2 failed: %v", err)
			}
			if text2 != "resumed:"+firstID {
				t.Fatalf("run 2 text = %q, want resumed:%s", text2, firstID)
			}

			// Run 3: corrupt/stale session triggers Delete + single fresh retry, emitting only 1 result activity.
			if err := store.Save(sessionKey, CLISessionRecord{
				Format:    string(format),
				SessionID: "stale-nonexistent-id",
			}); err != nil {
				t.Fatal(err)
			}
			text3, acts3, err := run("step 3")
			if err != nil {
				t.Fatalf("run 3 (stale retry) failed: %v", err)
			}
			if !strings.HasPrefix(text3, "fresh:") || strings.TrimPrefix(text3, "fresh:") == "stale-nonexistent-id" {
				t.Fatalf("run 3 expected fresh session after stale retry, got %q", text3)
			}
			newID := strings.TrimPrefix(text3, "fresh:")
			rec3, ok, err := store.Load(sessionKey)
			if err != nil || !ok || rec3.SessionID != newID {
				t.Fatalf("stored session after retry mismatch: rec=%+v ok=%v err=%v", rec3, ok, err)
			}
			results := 0
			for _, a := range acts3 {
				if a.Kind == "result" {
					results++
				}
			}
			if results != 1 {
				t.Fatalf("expected 1 result activity across stale retry, got %d", results)
			}

			// Verify stderr-only missing session error also retries once fresh.
			if err := store.Save(sessionKey, CLISessionRecord{
				Format:    string(format),
				SessionID: "stale-stderr-id",
			}); err != nil {
				t.Fatal(err)
			}
			text4, _, err := run("step 4")
			if err != nil || !strings.HasPrefix(text4, "fresh:") {
				t.Fatalf("run 4 (stderr stale retry) = (%q, %v), want fresh session", text4, err)
			}

			traces := readTraceEntries(t, traceFile)
			if len(traces) != 6 { // run1 + run2 + run3(fail+retry) + run4(fail+retry)
				t.Fatalf("expected 6 CLI invocations across 4 runs (2 retries), got %d", len(traces))
			}
			if format == FormatClaudeCLI {
				if traces[1].ResumeID != firstID || traces[2].ResumeID != "stale-nonexistent-id" || traces[3].ResumeID != "" || traces[3].SessionID == "" {
					t.Fatalf("unexpected claude trace sequence: %+v", traces)
				}
			} else {
				if traces[1].ConvID != firstID || traces[2].ConvID != "stale-nonexistent-id" || traces[3].ConvID != "" {
					t.Fatalf("unexpected agy trace sequence: %+v", traces)
				}
			}
		})
	}
}

func TestCLIRunnerCancelKillsProcessTree(t *testing.T) {
	bin := buildFakeCLIBinary(t)
	workDir := t.TempDir()
	pidFile := filepath.Join(t.TempDir(), "grandchild.pid")

	ctx, cancelCause := context.WithCancelCause(context.Background())
	defer cancelCause(nil)

	var (
		mu         sync.Mutex
		activities []db.Activity
	)
	toolSeen := make(chan struct{})
	var toolSeenOnce sync.Once

	type runOut struct {
		text   string
		reason harness.TerminalReason
		err    error
	}
	done := make(chan runOut, 1)
	go func() {
		text, reason, err := runCLI(ctx, CLIRunOptions{
			Config: Config{
				Format:  FormatClaudeCLI,
				BaseURL: bin,
				Model:   "model-hang-grandchild",
			},
			Prompt:  pidFile,
			WorkDir: workDir,
			Env:     []string{"ARTEX_FAKE_PID_FILE=" + pidFile},
			Emit: func(a db.Activity) {
				mu.Lock()
				activities = append(activities, a)
				mu.Unlock()
				if a.Kind == "tool_use" {
					toolSeenOnce.Do(func() { close(toolSeen) })
				}
			},
		})
		done <- runOut{text: text, reason: reason, err: err}
	}()

	grandchildPID := waitForPIDFile(t, pidFile, 5*time.Second)
	t.Cleanup(func() { cleanupPID(grandchildPID) })
	select {
	case <-toolSeen:
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for tool_use event before cancel")
	}

	if !processAlive(grandchildPID) {
		t.Fatalf("expected grandchild pid %d to be alive before cancel", grandchildPID)
	}

	cancelStart := time.Now()
	cancelCause(AbortPausedByUser)

	var out runOut
	select {
	case out = <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("runCLI did not exit promptly after context cancellation")
	}
	if !errors.Is(out.err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", out.err)
	}
	if elapsed := time.Since(cancelStart); elapsed > 4*time.Second {
		t.Fatalf("cancellation took too long: %s", elapsed)
	}
	waitUntilDead(t, "grandchild", grandchildPID, 2*time.Second)

	mu.Lock()
	defer mu.Unlock()
	var resultAct *db.Activity
	for i := range activities {
		if activities[i].Kind == "result" {
			resultAct = &activities[i]
		}
	}
	if resultAct == nil {
		t.Fatalf("expected final result activity on cancellation, got %+v", activities)
	}
	if !strings.Contains(resultAct.Detail, "paused_by_user") || !strings.Contains(resultAct.Detail, "Bash") {
		t.Fatalf("expected abort cause and pending tool in result detail, got %q", resultAct.Detail)
	}
}

func TestCLIRunnerUsageSinkAndBudget(t *testing.T) {
	bin := buildFakeCLIBinary(t)
	t.Setenv("CLAUDECODE", "1")

	ledger := NewBudgetLedger(NewBudget(100_000, 10), NewBudget(2_000, 5))
	meter := ledger.StartIntent()
	ctx := llmrec.WithUsageSink(WithIntentMeter(context.Background(), meter), meter.Record)
	ctx = llmrec.WithTaskID(ctx, "42")

	traceFile := filepath.Join(t.TempDir(), "trace.jsonl")
	extraEnv := append(proxyEnv("http://127.0.0.1:8788", ""), "ARTEX_FAKE_TRACE_FILE="+traceFile)

	text, reason, err := runCLI(ctx, CLIRunOptions{
		Config: Config{
			Format:      FormatAgyCLI,
			BaseURL:     bin,
			Model:       "model-usage-only",
			ProfileName: "agy-budget-prof",
		},
		Prompt:      "check budget",
		WorkDir:     t.TempDir(),
		Env:         extraEnv,
		WorkerLabel: "worker",
	})
	if err != nil {
		t.Fatalf("runCLI failed: %v", err)
	}
	if reason != harness.ReasonCompleted || text != "budget ok" {
		t.Fatalf("unexpected outcome: text=%q reason=%q", text, reason)
	}

	// model-usage-only emits result usage: input=500, output=120, cache_read=60.
	// Budget accounts InputTokens+OutputTokens = 620 tokens and 1 turn.
	rem := meter.Remaining()
	if rem.Tokens != 2_000-620 || rem.Turns != 4 {
		t.Fatalf("meter remaining = %+v, want tokens=%d turns=4", rem, 2_000-620)
	}
	st := ledger.State()
	if st.Spent.Tokens != 620 || st.Spent.Turns != 1 {
		t.Fatalf("ledger spent = %+v, want {620 1}", st.Spent)
	}

	traces := readTraceEntries(t, traceFile)
	if len(traces) != 1 {
		t.Fatalf("expected 1 trace entry, got %d", len(traces))
	}
	if traces[0].HTTPProxy != "http://127.0.0.1:8788" {
		t.Fatalf("expected proxyEnv HTTP_PROXY in subprocess, got %q", traces[0].HTTPProxy)
	}
	if !strings.Contains(traces[0].NoProxy, "127.0.0.1") || !strings.Contains(traces[0].NoProxy, "localhost") {
		t.Fatalf("expected loopback NO_PROXY in subprocess, got %q", traces[0].NoProxy)
	}
}

func TestCLIRunnerQuotaAndErrors(t *testing.T) {
	bin := buildFakeCLIBinary(t)

	for _, format := range []llm.Format{FormatClaudeCLI, FormatAgyCLI} {
		t.Run(string(format), func(t *testing.T) {
			for _, model := range []string{"model-quota-result", "model-quota-stderr"} {
				var acts []db.Activity
				_, reason, err := runCLI(context.Background(), CLIRunOptions{
					Config: Config{
						Format:  format,
						BaseURL: bin,
						Model:   model,
					},
					Prompt:  "go",
					WorkDir: t.TempDir(),
					Emit: func(a db.Activity) {
						acts = append(acts, a)
					},
				})
				if err == nil || !IsQuotaExhaustedMessage(err.Error()) {
					t.Fatalf("%s/%s: expected IsQuotaExhaustedMessage error, got err=%v reason=%q", format, model, err, reason)
				}
				if reason != harness.ReasonModelError {
					t.Fatalf("%s/%s: reason=%q, want model_error", format, model, reason)
				}
				if len(acts) == 0 || !acts[len(acts)-1].IsError || acts[len(acts)-1].Kind != "result" {
					t.Fatalf("%s/%s: expected final IsError result activity, got %+v", format, model, acts)
				}
			}

			// Non-zero exit without result preserves bounded stderr.
			var acts []db.Activity
			_, reason, err := runCLI(context.Background(), CLIRunOptions{
				Config: Config{
					Format:  format,
					BaseURL: bin,
					Model:   "model-exit-error-no-result",
				},
				Prompt:  "go",
				WorkDir: t.TempDir(),
				Emit: func(a db.Activity) {
					acts = append(acts, a)
				},
			})
			if err == nil || !strings.Contains(err.Error(), "fatal stderr failure line") {
				t.Fatalf("%s: expected stderr in error, got %v", format, err)
			}
			if reason != harness.ReasonModelError {
				t.Fatalf("%s: reason=%q, want model_error", format, reason)
			}
			if len(acts) == 0 || acts[len(acts)-1].Kind != "result" || !acts[len(acts)-1].IsError {
				t.Fatalf("%s: expected error result activity, got %+v", format, acts)
			}
		})
	}

	// Ensure unused helper import check doesn't complain if strconv isn't referenced directly.
	_ = strconv.Itoa(0)
}
