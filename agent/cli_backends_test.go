package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/Autumn-27/norma/harness"
	"github.com/Autumn-27/norma/llm"
)

func TestCLIBackendRegistryAndReadiness(t *testing.T) {
	if !IsCLIFormat(FormatClaudeCLI) {
		t.Fatalf("expected FormatClaudeCLI to be registered")
	}
	if !IsCLIFormat(FormatAgyCLI) {
		t.Fatalf("expected FormatAgyCLI to be registered")
	}
	if IsCLIFormat(llm.FormatAnthropic) || IsCLIFormat(FormatOllama) {
		t.Fatalf("unexpected CLI format classification for non-CLI provider")
	}

	formats := CLIFormats()
	if !slices.Contains(formats, FormatClaudeCLI) || !slices.Contains(formats, FormatAgyCLI) {
		t.Fatalf("CLIFormats() = %v, missing built-in backends", formats)
	}

	cb, ok := LookupCLIBackend(FormatClaudeCLI)
	if !ok {
		t.Fatalf("LookupCLIBackend(FormatClaudeCLI) failed")
	}
	if cb.DefaultBinary() != "claude" || cb.DefaultModel() != "sonnet" {
		t.Fatalf("unexpected claude defaults: bin=%q model=%q", cb.DefaultBinary(), cb.DefaultModel())
	}

	ab, ok := LookupCLIBackend(FormatAgyCLI)
	if !ok {
		t.Fatalf("LookupCLIBackend(FormatAgyCLI) failed")
	}
	if ab.DefaultBinary() != "agy" || ab.DefaultModel() != "gemini-4-hydrogen-high" {
		t.Fatalf("unexpected agy defaults: bin=%q model=%q", ab.DefaultBinary(), ab.DefaultModel())
	}

	dir := t.TempDir()
	fakeBin := filepath.Join(dir, "fake-cli")
	if err := os.WriteFile(fakeBin, []byte("ok"), 0o755); err != nil {
		t.Fatal(err)
	}

	if err := cb.CheckReady(Config{BaseURL: fakeBin}); err != nil {
		t.Fatalf("CheckReady with valid path failed: %v", err)
	}
	if err := cb.CheckReady(Config{BaseURL: filepath.Join(dir, "missing-cli")}); err == nil {
		t.Fatalf("CheckReady with missing path expected error")
	}

	t.Setenv("ARTEX_AGY_BIN", fakeBin)
	if err := ab.CheckReady(Config{}); err != nil {
		t.Fatalf("CheckReady with ARTEX_AGY_BIN failed: %v", err)
	}
}

func TestCLISessionStore(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "cli-sessions")
	store := NewCLISessionStore(dir)

	// Missing key returns ok=false, err=nil.
	rec, ok, err := store.Load("exp1-worker-i1")
	if err != nil || ok || rec.SessionID != "" {
		t.Fatalf("Load(missing) = (%+v, %v, %v), want (zero, false, nil)", rec, ok, err)
	}

	// Save and reload normal key.
	want := CLISessionRecord{
		Format:    string(FormatClaudeCLI),
		SessionID: "11111111-2222-3333-4444-555555555555",
	}
	if err := store.Save("exp1-worker-i1", want); err != nil {
		t.Fatalf("Save failed: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "exp1-worker-i1.json")); err != nil {
		t.Fatalf("expected readable filename exp1-worker-i1.json: %v", err)
	}

	got, ok, err := store.Load("exp1-worker-i1")
	if err != nil || !ok {
		t.Fatalf("Load failed: ok=%v err=%v", ok, err)
	}
	if got.SessionID != want.SessionID || got.SessionKey != "exp1-worker-i1" || got.UpdatedAt.IsZero() {
		t.Fatalf("loaded record mismatch: %+v", got)
	}

	// Overwrite atomically.
	want.SessionID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
	if err := store.Save("exp1-worker-i1", want); err != nil {
		t.Fatalf("Save overwrite failed: %v", err)
	}
	got, ok, err = store.Load("exp1-worker-i1")
	if err != nil || !ok || got.SessionID != want.SessionID {
		t.Fatalf("Load after overwrite = (%+v, %v, %v)", got, ok, err)
	}

	// Sanitization prevents path traversal and avoids collisions between different keys.
	if err := store.Save("../evil/a:b", CLISessionRecord{SessionID: "id-colon"}); err != nil {
		t.Fatalf("Save sanitized key 1 failed: %v", err)
	}
	if err := store.Save("../evil/a/b", CLISessionRecord{SessionID: "id-slash"}); err != nil {
		t.Fatalf("Save sanitized key 2 failed: %v", err)
	}
	r1, ok1, err1 := store.Load("../evil/a:b")
	r2, ok2, err2 := store.Load("../evil/a/b")
	if err1 != nil || !ok1 || err2 != nil || !ok2 {
		t.Fatalf("Load sanitized keys failed: (%v,%v) (%v,%v)", ok1, err1, ok2, err2)
	}
	if r1.SessionID != "id-colon" || r2.SessionID != "id-slash" {
		t.Fatalf("sanitized keys collided: r1=%+v r2=%+v", r1, r2)
	}

	// Delete removes entry and is idempotent.
	if err := store.Delete("exp1-worker-i1"); err != nil {
		t.Fatalf("Delete failed: %v", err)
	}
	if _, ok, err := store.Load("exp1-worker-i1"); err != nil || ok {
		t.Fatalf("Load after Delete = (%v, %v), want (false, nil)", ok, err)
	}
	if err := store.Delete("exp1-worker-i1"); err != nil {
		t.Fatalf("idempotent Delete failed: %v", err)
	}

	// Corrupt JSON returns an error.
	if err := os.WriteFile(filepath.Join(dir, "corrupt.json"), []byte("{not-json"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, _, err := store.Load("corrupt"); err == nil {
		t.Fatalf("expected error loading corrupt JSON")
	}

	// Concurrent writes/reads do not leave temp files or fail.
	var wg sync.WaitGroup
	for i := range 12 {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			key := fmt.Sprintf("concurrent-%d", i%3)
			if err := store.Save(key, CLISessionRecord{SessionID: fmt.Sprintf("sid-%d", i)}); err != nil {
				t.Errorf("concurrent Save failed: %v", err)
			}
			if _, _, err := store.Load(key); err != nil {
				t.Errorf("concurrent Load failed: %v", err)
			}
		}(i)
	}
	wg.Wait()

	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".tmp") {
			t.Fatalf("leftover temp file in session dir: %s", e.Name())
		}
	}
}

func TestCLIClaudeBuildInvocation(t *testing.T) {
	b, _ := LookupCLIBackend(FormatClaudeCLI)
	ctx := context.Background()

	t.Run("full_session_mcp_hooks_and_large_system_prompt", func(t *testing.T) {
		workDir := t.TempDir()
		largeSys := strings.Repeat("Rule line for pentest agent.\n", 700) // >16KB

		inv, err := b.BuildInvocation(ctx, CLIInvocationParams{
			Config: Config{
				Format:          FormatClaudeCLI,
				BaseURL:         "/opt/bin/claude",
				Model:           "opus",
				ReasoningEffort: "high",
			},
			Prompt:       "Inspect https://example.com\nand check headers.",
			SystemPrompt: largeSys,
			WorkDir:      workDir,
			ExtraDirs:    []string{filepath.Join(workDir, "shared")},
			MCPURL:       "http://127.0.0.1:43210/mcp/tok123",
			HookURL:      "http://127.0.0.1:43210/hook/tok123",
			HookBin:      "/usr/local/bin/artex",
			HookToken:    "tok123",
			SessionKey:   "exp9-worker-i4",
			SessionID:    "uuid-1234",
			IsResume:     false,
			MaxTurns:     25,
		})
		if err != nil {
			t.Fatalf("BuildInvocation failed: %v", err)
		}
		defer inv.Cleanup()

		if inv.Binary != "/opt/bin/claude" {
			t.Fatalf("Binary = %q, want /opt/bin/claude", inv.Binary)
		}
		if inv.Stdin != "Inspect https://example.com\nand check headers." {
			t.Fatalf("Stdin mismatch: %q", inv.Stdin)
		}
		if inv.Dir != workDir {
			t.Fatalf("Dir = %q, want %q", inv.Dir, workDir)
		}

		assertArgPair(t, inv.Args, "--output-format", "stream-json")
		assertArgPair(t, inv.Args, "--permission-mode", "bypassPermissions")
		assertArgPair(t, inv.Args, "--model", "opus")
		assertArgPair(t, inv.Args, "--effort", "high")
		assertArgPair(t, inv.Args, "--max-turns", "25")
		assertArgPair(t, inv.Args, "--session-id", "uuid-1234")
		assertArgPair(t, inv.Args, "--add-dir", filepath.Join(workDir, "shared"))
		if !slices.Contains(inv.Args, "--verbose") || !slices.Contains(inv.Args, "--strict-mcp-config") {
			t.Fatalf("missing --verbose or --strict-mcp-config in %v", inv.Args)
		}

		mcpFile := argValue(inv.Args, "--mcp-config")
		if mcpFile == "" {
			t.Fatalf("missing --mcp-config in %v", inv.Args)
		}
		mcpBytes, err := os.ReadFile(mcpFile)
		if err != nil {
			t.Fatalf("read mcp config: %v", err)
		}
		var mcpJSON struct {
			MCPServers map[string]struct {
				Type string `json:"type"`
				URL  string `json:"url"`
			} `json:"mcpServers"`
		}
		if err := json.Unmarshal(mcpBytes, &mcpJSON); err != nil {
			t.Fatal(err)
		}
		if mcpJSON.MCPServers["xetra"].Type != "http" || mcpJSON.MCPServers["xetra"].URL != "http://127.0.0.1:43210/mcp/tok123" {
			t.Fatalf("unexpected mcp config JSON: %+v", mcpJSON)
		}

		settingsFile := argValue(inv.Args, "--settings")
		if settingsFile == "" {
			t.Fatalf("missing --settings in %v", inv.Args)
		}
		settingsBytes, err := os.ReadFile(settingsFile)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(settingsBytes), "PreToolUse") || !strings.Contains(string(settingsBytes), "PostToolUse") || !strings.Contains(string(settingsBytes), "artex") {
			t.Fatalf("unexpected settings JSON: %s", settingsBytes)
		}
		assertEnv(t, inv.Env, "XETRA_BRIDGE_URL", "http://127.0.0.1:43210/hook/tok123")
		assertEnv(t, inv.Env, "XETRA_BRIDGE_TOKEN", "tok123")
		assertEnv(t, inv.Env, "XETRA_HOOK_CLI", "claude")

		sysArg := argValue(inv.Args, "--system-prompt")
		sysFile := filepath.Join(workDir, ".xetra-system.md")
		if !strings.Contains(sysArg, sysFile) {
			t.Fatalf("--system-prompt = %q, expected reference to %q", sysArg, sysFile)
		}
		if data, err := os.ReadFile(sysFile); err != nil || string(data) != largeSys {
			t.Fatalf("overflow system prompt file mismatch: err=%v len=%d", err, len(data))
		}

		inv.Cleanup()
		inv.Cleanup() // idempotent
		for _, p := range []string{mcpFile, settingsFile, sysFile} {
			if _, err := os.Stat(p); !os.IsNotExist(err) {
				t.Fatalf("expected cleanup to remove %q", p)
			}
		}
	})

	t.Run("resume_and_stateless_disable_tools", func(t *testing.T) {
		invResume, err := b.BuildInvocation(ctx, CLIInvocationParams{
			Config:     Config{Format: FormatClaudeCLI},
			Prompt:     "continue",
			SessionKey: "exp9-worker-i4",
			SessionID:  "uuid-1234",
			IsResume:   true,
		})
		if err != nil {
			t.Fatal(err)
		}
		defer invResume.Cleanup()
		assertArgPair(t, invResume.Args, "--resume", "uuid-1234")
		assertArgPair(t, invResume.Args, "--model", "sonnet")

		invStateless, err := b.BuildInvocation(ctx, CLIInvocationParams{
			Config:       Config{Format: FormatClaudeCLI},
			Prompt:       "ping",
			SystemPrompt: "Reply OK",
			HookURL:      "http://127.0.0.1:1/hook/t", // suppressed when DisableTools=true
			DisableTools: true,
		})
		if err != nil {
			t.Fatal(err)
		}
		defer invStateless.Cleanup()

		assertArgPair(t, invStateless.Args, "--tools", "")
		assertArgPair(t, invStateless.Args, "--system-prompt", "Reply OK")
		if !slices.Contains(invStateless.Args, "--no-session-persistence") {
			t.Fatalf("expected --no-session-persistence in %v", invStateless.Args)
		}
		if slices.Contains(invStateless.Args, "--settings") {
			t.Fatalf("did not expect --settings when DisableTools=true: %v", invStateless.Args)
		}
	})
}

func TestCLIAgyBuildInvocation(t *testing.T) {
	b, _ := LookupCLIBackend(FormatAgyCLI)
	ctx := context.Background()

	t.Run("inline_prompt_and_ephemeral_plugin", func(t *testing.T) {
		workDir := t.TempDir()
		// Create an unrelated file inside .agents to prove Cleanup preserves non-xetra entries.
		unrelated := filepath.Join(workDir, ".agents", "keep.txt")
		if err := os.MkdirAll(filepath.Dir(unrelated), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(unrelated, []byte("keep"), 0o600); err != nil {
			t.Fatal(err)
		}

		inv, err := b.BuildInvocation(ctx, CLIInvocationParams{
			Config: Config{
				Format:          FormatAgyCLI,
				ReasoningEffort: "xhigh",
			},
			Prompt:       "Run quick check",
			SystemPrompt: "Be concise.",
			WorkDir:      workDir,
			ExtraDirs:    []string{filepath.Join(workDir, "task")},
			MCPURL:       "http://127.0.0.1:55555/mcp/abc",
			HookURL:      "http://127.0.0.1:55555/hook/abc",
			HookBin:      "/bin/artex",
			HookToken:    "abc",
			SessionID:    "conv-77",
			IsResume:     true,
		})
		if err != nil {
			t.Fatalf("BuildInvocation failed: %v", err)
		}
		defer inv.Cleanup()

		assertArgPair(t, inv.Args, "--output-format", "stream-json")
		assertArgPair(t, inv.Args, "--model", "gemini-4-hydrogen-high")
		assertArgPair(t, inv.Args, "--effort", "xhigh")
		assertArgPair(t, inv.Args, "--add-dir", filepath.Join(workDir, "task"))
		assertArgPair(t, inv.Args, "--conversation", "conv-77")
		if !slices.Contains(inv.Args, "--dangerously-skip-permissions") {
			t.Fatalf("missing --dangerously-skip-permissions in %v", inv.Args)
		}
		if !slices.Contains(inv.Args, "-p=Run quick check") {
			t.Fatalf("expected inline -p=Run quick check in %v", inv.Args)
		}

		pluginDir := filepath.Join(workDir, ".agents", "plugins", "xetra")
		pluginJSON, err := os.ReadFile(filepath.Join(pluginDir, "plugin.json"))
		if err != nil || !strings.Contains(string(pluginJSON), `"xetra"`) {
			t.Fatalf("unexpected plugin.json: err=%v body=%s", err, pluginJSON)
		}
		mcpJSON, err := os.ReadFile(filepath.Join(pluginDir, "mcp_config.json"))
		if err != nil || !strings.Contains(string(mcpJSON), `"bridge"`) || !strings.Contains(string(mcpJSON), "http://127.0.0.1:55555/mcp/abc") {
			t.Fatalf("unexpected mcp_config.json: err=%v body=%s", err, mcpJSON)
		}
		hooksJSON, err := os.ReadFile(filepath.Join(pluginDir, "hooks.json"))
		if err != nil || !strings.Contains(string(hooksJSON), "PreToolUse") || !strings.Contains(string(hooksJSON), "PostToolUse") {
			t.Fatalf("unexpected hooks.json: err=%v body=%s", err, hooksJSON)
		}
		agentsMD, err := os.ReadFile(filepath.Join(pluginDir, "rules", "AGENTS.md"))
		if err != nil || !strings.Contains(string(agentsMD), "Be concise.") || !strings.Contains(string(agentsMD), "xetra_bridge") || !strings.Contains(string(agentsMD), "call_mcp_tool") {
			t.Fatalf("unexpected rules/AGENTS.md: err=%v body=%s", err, agentsMD)
		}
		assertEnv(t, inv.Env, "XETRA_BRIDGE_URL", "http://127.0.0.1:55555/hook/abc")
		assertEnv(t, inv.Env, "XETRA_BRIDGE_TOKEN", "abc")
		assertEnv(t, inv.Env, "XETRA_HOOK_CLI", "agy")

		inv.Cleanup()
		if _, err := os.Stat(pluginDir); !os.IsNotExist(err) {
			t.Fatalf("expected xetra plugin dir to be removed")
		}
		if _, err := os.Stat(unrelated); err != nil {
			t.Fatalf("expected unrelated .agents/keep.txt to be preserved: %v", err)
		}
	})

	t.Run("large_multiline_prompt_and_chunked_rules", func(t *testing.T) {
		workDir := t.TempDir()
		longPrompt := "Line 1\n" + strings.Repeat("A", 4100)
		longSystem := strings.Repeat("System rule paragraph line.\n", 1200) // ~33KB > 20KB

		inv, err := b.BuildInvocation(ctx, CLIInvocationParams{
			Config:       Config{Format: FormatAgyCLI},
			Prompt:       longPrompt,
			SystemPrompt: longSystem,
			WorkDir:      workDir,
		})
		if err != nil {
			t.Fatal(err)
		}
		defer inv.Cleanup()

		wantFlag := "-p=Read and execute the complete instructions in .xetra-prompt.md in your working directory."
		if !slices.Contains(inv.Args, wantFlag) {
			t.Fatalf("expected overflow prompt flag %q in %v", wantFlag, inv.Args)
		}
		promptData, err := os.ReadFile(filepath.Join(workDir, ".xetra-prompt.md"))
		if err != nil || string(promptData) != longPrompt {
			t.Fatalf("prompt overflow file mismatch: err=%v", err)
		}

		rulesDir := filepath.Join(workDir, ".agents", "plugins", "xetra", "rules")
		rule01, err := os.ReadFile(filepath.Join(rulesDir, "01.md"))
		if err != nil || len(rule01) == 0 || len(rule01) > agyRuleChunkLimit {
			t.Fatalf("unexpected rules/01.md: err=%v len=%d", err, len(rule01))
		}
		rule02, err := os.ReadFile(filepath.Join(rulesDir, "02.md"))
		if err != nil || len(rule02) == 0 || len(rule02) > agyRuleChunkLimit {
			t.Fatalf("unexpected rules/02.md: err=%v len=%d", err, len(rule02))
		}
		agentsMD, err := os.ReadFile(filepath.Join(rulesDir, "AGENTS.md"))
		if err != nil || !strings.Contains(string(agentsMD), "rules/01.md") || !strings.Contains(string(agentsMD), "rules/02.md") {
			t.Fatalf("unexpected rules/AGENTS.md index: err=%v body=%s", err, agentsMD)
		}

		inv.Cleanup()
		if _, err := os.Stat(filepath.Join(workDir, ".xetra-prompt.md")); !os.IsNotExist(err) {
			t.Fatalf("expected .xetra-prompt.md to be cleaned up")
		}
		if _, err := os.Stat(filepath.Join(workDir, ".agents")); !os.IsNotExist(err) {
			t.Fatalf("expected empty .agents dir to be removed")
		}
	})
}

func TestCLIClaudeStreamParser(t *testing.T) {
	b, _ := LookupCLIBackend(FormatClaudeCLI)

	ndjson := strings.Join([]string{
		`{"type":"system","subtype":"init","session_id":"sess-claude-1"}`,
		`{"type":"assistant","message":{"content":[{"type":"thinking","thinking":"analyzing target"},{"type":"text","text":"Running probe"},{"type":"tool_use","id":"tu_1","name":"Bash","input":{"command":"curl -I https://example.com"}}]}}`,
		`{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"tu_1","content":"HTTP/2 200","is_error":false}]}}`,
		`{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"tu_2","content":[{"type":"text","text":"line1"},{"type":"text","text":"line2"}],"is_error":true}]}}`,
		`{"type":"result","subtype":"success","is_error":false,"result":"Done probing","session_id":"sess-claude-1","total_cost_usd":0.042,"num_turns":3,"usage":{"input_tokens":1200,"output_tokens":340,"cache_read_input_tokens":800,"cache_creation_input_tokens":150}}`,
	}, "\n")

	lines := strings.Split(ndjson, "\n")
	ev0, err := b.ParseLine([]byte(lines[0]))
	if err != nil || len(ev0) != 1 || ev0[0].Kind != "init" || ev0[0].SessionID != "sess-claude-1" {
		t.Fatalf("init parse mismatch: %+v, err=%v", ev0, err)
	}

	ev1, err := b.ParseLine([]byte(lines[1]))
	if err != nil || len(ev1) != 3 {
		t.Fatalf("assistant parse mismatch: %+v, err=%v", ev1, err)
	}
	if ev1[0].Kind != "thinking" || ev1[0].Text != "analyzing target" {
		t.Fatalf("unexpected thinking event: %+v", ev1[0])
	}
	if ev1[1].Kind != "text" || ev1[1].Text != "Running probe" {
		t.Fatalf("unexpected text event: %+v", ev1[1])
	}
	if ev1[2].Kind != "tool_use" || ev1[2].ToolID != "tu_1" || ev1[2].ToolName != "Bash" {
		t.Fatalf("unexpected tool_use event: %+v", ev1[2])
	}

	ev2, err := b.ParseLine([]byte(lines[2]))
	if err != nil || len(ev2) != 1 || ev2[0].Kind != "tool_result" || ev2[0].Text != "HTTP/2 200" || ev2[0].IsError {
		t.Fatalf("tool_result string parse mismatch: %+v, err=%v", ev2, err)
	}

	ev3, err := b.ParseLine([]byte(lines[3]))
	if err != nil || len(ev3) != 1 || ev3[0].Kind != "tool_result" || ev3[0].Text != "line1\nline2" || !ev3[0].IsError {
		t.Fatalf("tool_result array parse mismatch: %+v, err=%v", ev3, err)
	}

	res, err := b.ParseFinal([]byte(ndjson))
	if err != nil {
		t.Fatalf("ParseFinal failed: %v", err)
	}
	if res.Reason != harness.ReasonCompleted || res.Text != "Done probing" || res.SessionID != "sess-claude-1" {
		t.Fatalf("unexpected result: %+v", res)
	}
	if res.CostUSD != 0.042 || res.NumTurns != 3 {
		t.Fatalf("unexpected cost/turns: %+v", res)
	}
	if res.Usage.InputTokens != 1200 || res.Usage.OutputTokens != 340 || res.Usage.CacheReadTokens != 800 || res.Usage.CacheWriteTokens != 150 {
		t.Fatalf("unexpected usage: %+v", res.Usage)
	}
	if res.IsError || res.IsQuotaError || res.IsSessionMissing {
		t.Fatalf("unexpected error flags on success: %+v", res)
	}

	// Success text mentioning 429 must NOT trigger IsQuotaError.
	mention429 := `{"type":"result","subtype":"success","is_error":false,"result":"Endpoint returned HTTP 429 Too Many Requests and session not found"}`
	if r, err := b.ParseFinal([]byte(mention429)); err != nil || r.IsQuotaError || r.IsSessionMissing {
		t.Fatalf("successful response falsely flagged error: %+v err=%v", r, err)
	}

	// Max turns.
	maxTurns := `{"type":"result","subtype":"error_max_turns","is_error":false,"result":"","num_turns":10}`
	if r, err := b.ParseFinal([]byte(maxTurns)); err != nil || r.Reason != harness.ReasonMaxTurns {
		t.Fatalf("expected ReasonMaxTurns, got %+v err=%v", r, err)
	}

	// Quota error.
	quotaLine := `{"type":"result","subtype":"error_during_execution","is_error":true,"result":"Credit balance is too low","errors":["API Error: 429 rate_limit_error"]}`
	if r, err := b.ParseFinal([]byte(quotaLine)); err != nil || r.Reason != harness.ReasonModelError || !r.IsQuotaError || !r.IsError {
		t.Fatalf("expected quota error result, got %+v err=%v", r, err)
	}

	// Missing session error.
	missingLine := `{"type":"result","subtype":"error_during_execution","is_error":true,"errors":["No conversation found with session ID: deadbeef"]}`
	if r, err := b.ParseFinal([]byte(missingLine)); err != nil || r.Reason != harness.ReasonModelError || !r.IsSessionMissing {
		t.Fatalf("expected session-missing result, got %+v err=%v", r, err)
	}
}

func TestCLIAgyStreamParser(t *testing.T) {
	b, _ := LookupCLIBackend(FormatAgyCLI)

	ndjson := strings.Join([]string{
		`{"event":"init","conversation_id":"agy-conv-99","init":{}}`,
		`{"event":"step_update","step_update":{"conversation_id":"agy-conv-99","step_index":0,"state":"DONE","step_type":"user_input","text_delta":"ignored echo"}}`,
		`{"event":"step_update","step_update":{"conversation_id":"agy-conv-99","step_index":1,"state":"ACTIVE","step_type":"thought","text_delta":"checking endpoint"}}`,
		`{"event":"step_update","step_update":{"conversation_id":"agy-conv-99","step_index":2,"state":"ACTIVE","step_type":"tool","tool_name":"call_mcp_tool","tool_info":{"name":"call_mcp_tool","parameters":{"ServerName":"xetra_bridge","ToolName":"record_fact","Arguments":{"key":"open_port","value":"443"}}}}}`,
		`{"event":"step_update","step_update":{"conversation_id":"agy-conv-99","step_index":2,"state":"ACTIVE","step_type":"tool","tool_name":"call_mcp_tool","tool_info":{"name":"call_mcp_tool","parameters":{"ServerName":"xetra_bridge","ToolName":"record_fact","Arguments":{"key":"open_port","value":"443"}}}}}`,
		`{"event":"step_update","step_update":{"conversation_id":"agy-conv-99","step_index":2,"state":"DONE","step_type":"tool","tool_name":"call_mcp_tool","tool_info":{"name":"call_mcp_tool","parameters":{"ServerName":"xetra_bridge","ToolName":"record_fact","Arguments":{"key":"open_port","value":"443"}},"output":"fact recorded"},"usage":{"input_tokens":100,"output_tokens":20,"thinking_tokens":10,"cache_read_tokens":50}}}`,
		`{"event":"step_update","step_update":{"conversation_id":"agy-conv-99","step_index":3,"state":"DONE","step_type":"tool","tool_name":"run_command","tool_info":{"name":"run_command","parameters":{"CommandLine":"false"},"error":"exit status 1"}}}`,
		`{"event":"step_update","step_update":{"conversation_id":"agy-conv-99","step_index":4,"state":"ACTIVE","step_type":"agent_response","text_delta":"Recorded port 443."}}`,
		`{"event":"result","result":{"conversation_id":"agy-conv-99","status":"SUCCESS","response":"Recorded port 443.","num_turns":2,"usage":{"input_tokens":500,"output_tokens":120,"thinking_tokens":80,"cache_read_tokens":200,"total_tokens":900},"finished_reason":"STOP"}}`,
	}, "\n")

	parser := b.NewParser()
	var collected []CLIEvent
	for _, line := range strings.Split(ndjson, "\n") {
		evs, err := parser.ParseLine([]byte(line))
		if err != nil {
			t.Fatalf("ParseLine error on %s: %v", line, err)
		}
		collected = append(collected, evs...)
	}

	var toolUseCount, toolResultCount int
	for _, ev := range collected {
		if ev.Kind == "tool_use" && ev.ToolID == "step-2" {
			toolUseCount++
			if ev.ToolName != "record_fact" {
				t.Fatalf("expected unwrapped MCP tool name record_fact, got %q", ev.ToolName)
			}
			if !strings.Contains(string(ev.ToolInput), `"open_port"`) {
				t.Fatalf("expected unwrapped Arguments JSON, got %s", ev.ToolInput)
			}
		}
		if ev.Kind == "tool_result" && ev.ToolID == "step-2" {
			toolResultCount++
			if ev.Text != "fact recorded" || ev.IsError {
				t.Fatalf("unexpected tool_result for step-2: %+v", ev)
			}
		}
		if ev.Kind == "tool_result" && ev.ToolID == "step-3" {
			if !ev.IsError || ev.Text != "exit status 1" {
				t.Fatalf("unexpected error tool_result for step-3: %+v", ev)
			}
		}
		if ev.Kind == "text" && ev.Text == "ignored echo" {
			t.Fatalf("user_input echo should not emit text event")
		}
	}
	if toolUseCount != 1 {
		t.Fatalf("expected deduplicated tool_use count=1 for step-2, got %d", toolUseCount)
	}
	if toolResultCount != 1 {
		t.Fatalf("expected tool_result count=1 for step-2, got %d", toolResultCount)
	}

	res, err := b.ParseFinal([]byte(ndjson))
	if err != nil {
		t.Fatalf("ParseFinal failed: %v", err)
	}
	if res.Reason != harness.ReasonCompleted || res.SessionID != "agy-conv-99" || res.Text != "Recorded port 443." || res.NumTurns != 2 {
		t.Fatalf("unexpected agy result: %+v", res)
	}
	if res.Usage.InputTokens != 500 || res.Usage.OutputTokens != 200 || res.Usage.CacheReadTokens != 200 {
		t.Fatalf("unexpected agy usage: %+v", res.Usage)
	}

	// Timeout / error / missing conversation mapping.
	timeoutLine := `{"event":"result","result":{"conversation_id":"c1","status":"TIMEOUT","error":"deadline exceeded","finished_reason":"timeout"}}`
	if r, err := b.ParseFinal([]byte(timeoutLine)); err != nil || r.Reason != harness.ReasonTimeout || !r.IsError {
		t.Fatalf("expected ReasonTimeout, got %+v err=%v", r, err)
	}

	missingLine := `{"event":"result","result":{"conversation_id":"c1","status":"ERROR","error":"conversation not found: c1"}}`
	if r, err := b.ParseFinal([]byte(missingLine)); err != nil || r.Reason != harness.ReasonModelError || !r.IsSessionMissing {
		t.Fatalf("expected IsSessionMissing=true, got %+v err=%v", r, err)
	}

	quotaLine := `{"event":"result","result":{"conversation_id":"c1","status":"ERROR","error":"HTTP 429 quota_exceeded"}}`
	if r, err := b.ParseFinal([]byte(quotaLine)); err != nil || r.Reason != harness.ReasonModelError || !r.IsQuotaError {
		t.Fatalf("expected IsQuotaError=true, got %+v err=%v", r, err)
	}
}

func TestCLIAgyListModels(t *testing.T) {
	b, _ := LookupCLIBackend(FormatAgyCLI)
	dir := t.TempDir()

	var scriptPath string
	if runtime.GOOS == "windows" {
		scriptPath = filepath.Join(dir, "fake-agy.cmd")
		content := "@echo off\r\n" +
			"if \"%1\"==\"models\" (\r\n" +
			"  echo Fetching available models...\r\n" +
			"  echo.\r\n" +
			"  echo gemini-4-hydrogen-high\tGemini 4 Hydrogen ^(High^)\r\n" +
			"  echo claude-opus-5-5-high\tClaude Opus 5.5 ^(High^)\r\n" +
			"  echo gemini-4-hydrogen-high\tGemini 4 Hydrogen ^(High^)\r\n" +
			"  exit /b 0\r\n" +
			")\r\n" +
			"exit /b 1\r\n"
		if err := os.WriteFile(scriptPath, []byte(content), 0o755); err != nil {
			t.Fatal(err)
		}
	} else {
		scriptPath = filepath.Join(dir, "fake-agy")
		content := "#!/bin/sh\n" +
			"if [ \"$1\" = \"models\" ]; then\n" +
			"  printf 'Fetching available models...\\n\\n'\n" +
			"  printf 'gemini-4-hydrogen-high\\tGemini 4 Hydrogen (High)\\n'\n" +
			"  printf 'claude-opus-5-5-high\\tClaude Opus 5.5 (High)\\n'\n" +
			"  printf 'gemini-4-hydrogen-high\\tGemini 4 Hydrogen (High)\\n'\n" +
			"  exit 0\n" +
			"fi\n" +
			"exit 1\n"
		if err := os.WriteFile(scriptPath, []byte(content), 0o755); err != nil {
			t.Fatal(err)
		}
	}

	models, err := b.ListModels(context.Background(), Config{BaseURL: scriptPath})
	if err != nil {
		t.Fatalf("ListModels failed: %v", err)
	}
	want := []string{"gemini-4-hydrogen-high", "claude-opus-5-5-high"}
	if !slices.Equal(models, want) {
		t.Fatalf("ListModels() = %v, want %v", models, want)
	}
}

func assertArgPair(t *testing.T, args []string, flag, wantVal string) {
	t.Helper()
	for i := 0; i < len(args)-1; i++ {
		if args[i] == flag && args[i+1] == wantVal {
			return
		}
	}
	t.Fatalf("expected flag pair %q %q in args %v", flag, wantVal, args)
}

func argValue(args []string, flag string) string {
	for i := 0; i < len(args)-1; i++ {
		if args[i] == flag {
			return args[i+1]
		}
	}
	return ""
}

func assertEnv(t *testing.T, env []string, key, wantVal string) {
	t.Helper()
	want := key + "=" + wantVal
	if !slices.Contains(env, want) {
		t.Fatalf("expected env %q in %v", want, env)
	}
}
