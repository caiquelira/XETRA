package mcpbridge

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/Autumn-27/artex/mcphttp"
	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/permission"
	actool "github.com/Autumn-27/norma/tool"
)

type ctxKey string

const testCtxKey ctxKey = "run_marker"

type hookCall struct {
	kind    string
	tool    string
	input   string
	result  string
	isErr   bool
	marker  string
}

type mockHookRunner struct {
	mu           sync.Mutex
	calls        []hookCall
	preBlock     bool
	preReason    string
	preUpdated   []byte
	stopPrevent  bool
	stopErrors   []string
	stopMessage  string
}

func (m *mockHookRunner) PreToolUse(ctx context.Context, name string, input []byte) (bool, string, []byte) {
	m.mu.Lock()
	defer m.mu.Unlock()
	marker, _ := ctx.Value(testCtxKey).(string)
	m.calls = append(m.calls, hookCall{
		kind:   "pre",
		tool:   name,
		input:  string(input),
		marker: marker,
	})
	return m.preBlock, m.preReason, m.preUpdated
}

func (m *mockHookRunner) PostToolUse(ctx context.Context, name string, input, result []byte, isErr bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	marker, _ := ctx.Value(testCtxKey).(string)
	m.calls = append(m.calls, hookCall{
		kind:   "post",
		tool:   name,
		input:  string(input),
		result: string(result),
		isErr:  isErr,
		marker: marker,
	})
}

func (m *mockHookRunner) Stop(ctx context.Context, _ []llm.Message) (bool, []string, string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	marker, _ := ctx.Value(testCtxKey).(string)
	m.calls = append(m.calls, hookCall{
		kind:   "stop",
		marker: marker,
	})
	return m.stopPrevent, m.stopErrors, m.stopMessage
}

func (m *mockHookRunner) snapshot() []hookCall {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]hookCall, len(m.calls))
	copy(out, m.calls)
	return out
}

func makeEchoTool(called *atomic.Int32) actool.CoreTool {
	return actool.Build(actool.Spec{
		Name:        "echo_tool",
		Description: "Echoes input message",
		Schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"msg": map[string]any{"type": "string"},
			},
			"required": []string{"msg"},
		},
		Permissions: func(context.Context, json.RawMessage, permission.Context) permission.Decision {
			return permission.Allowed()
		},
		Run: func(ctx context.Context, in json.RawMessage, tc *actool.ToolContext) (actool.Result, error) {
			if called != nil {
				called.Add(1)
			}
			var args struct {
				Msg string `json:"msg"`
			}
			_ = json.Unmarshal(in, &args)
			marker, _ := ctx.Value(testCtxKey).(string)
			wd := ""
			if tc != nil {
				wd = tc.WorkingDir
			}
			return actool.Text(args.Msg + "|marker=" + marker + "|wd=" + wd), nil
		},
	})
}

func postJSON(t *testing.T, url string, headers map[string]string, body string) (*http.Response, []byte) {
	t.Helper()
	req, err := http.NewRequest(http.MethodPost, url, strings.NewReader(body))
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := loopbackHTTPClient(0).Do(req)
	if err != nil {
		t.Fatalf("Do POST %s: %v", url, err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	return resp, data
}

func TestMCPEndpointAuthAndMethods(t *testing.T) {
	srv, err := NewServer()
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	defer srv.Close()

	token := srv.RegisterRun(RunConfig{
		Tools: []actool.CoreTool{makeEchoTool(nil)},
	})

	// Unauthorized token.
	resp, _ := postJSON(t, srv.MCPURL("wrong-token"), nil, `{"jsonrpc":"2.0","id":1,"method":"initialize"}`)
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("wrong token status = %d, want 401", resp.StatusCode)
	}

	// GET returns 405 Method Not Allowed.
	getResp, err := loopbackHTTPClient(0).Get(srv.MCPURL(token))
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	getResp.Body.Close()
	if getResp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("GET status = %d, want 405", getResp.StatusCode)
	}

	// DELETE returns 200 OK.
	delReq, _ := http.NewRequest(http.MethodDelete, srv.MCPURL(token), nil)
	delResp, err := loopbackHTTPClient(0).Do(delReq)
	if err != nil {
		t.Fatalf("DELETE: %v", err)
	}
	delResp.Body.Close()
	if delResp.StatusCode != http.StatusOK {
		t.Fatalf("DELETE status = %d, want 200", delResp.StatusCode)
	}

	// Unregister rejects subsequent calls.
	srv.UnregisterRun(token)
	resp, _ = postJSON(t, srv.MCPURL(token), nil, `{"jsonrpc":"2.0","id":1,"method":"initialize"}`)
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("after UnregisterRun status = %d, want 401", resp.StatusCode)
	}
}

func TestMCPAgyHandshakeAndToolExecution(t *testing.T) {
	srv, err := NewServer()
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	defer srv.Close()

	var called atomic.Int32
	hooks := &mockHookRunner{}
	nilSchemaTool := actool.Build(actool.Spec{
		Name:        "no_schema_tool",
		Description: "Tool without schema",
		Run: func(context.Context, json.RawMessage, *actool.ToolContext) (actool.Result, error) {
			return actool.Text("ok"), nil
		},
	})
	panicTool := actool.Build(actool.Spec{
		Name:        "panic_tool",
		Description: "Panics",
		Run: func(context.Context, json.RawMessage, *actool.ToolContext) (actool.Result, error) {
			panic("boom")
		},
	})
	errTool := actool.Build(actool.Spec{
		Name:        "err_tool",
		Description: "Returns error",
		Run: func(context.Context, json.RawMessage, *actool.ToolContext) (actool.Result, error) {
			return actool.Result{}, errors.New("tool failed")
		},
	})

	runCtx := context.WithValue(context.Background(), testCtxKey, "ctx-42")
	token := srv.RegisterRun(RunConfig{
		Context: runCtx,
		Tools:   []actool.CoreTool{makeEchoTool(&called), nilSchemaTool, panicTool, errTool},
		Hooks:   hooks,
	})
	mcpURL := srv.MCPURL(token)

	// 1. agy's initial server/discover probe -> -32601 with HTTP 200 and Mcp-Session-Id.
	resp, data := postJSON(t, mcpURL, nil, `{"jsonrpc":"2.0","id":0,"method":"server/discover"}`)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("server/discover status = %d, want 200", resp.StatusCode)
	}
	sid := resp.Header.Get("Mcp-Session-Id")
	if sid == "" {
		t.Fatalf("expected Mcp-Session-Id header on server/discover")
	}
	var discoverRPC struct {
		ID    json.RawMessage `json:"id"`
		Error *rpcError       `json:"error"`
	}
	if err := json.Unmarshal(data, &discoverRPC); err != nil {
		t.Fatalf("unmarshal discover: %v (%s)", err, string(data))
	}
	if string(discoverRPC.ID) != "0" || discoverRPC.Error == nil || discoverRPC.Error.Code != -32601 {
		t.Fatalf("unexpected discover response: %s", string(data))
	}

	// 2. initialize.
	resp, data = postJSON(t, mcpURL, map[string]string{"Mcp-Session-Id": sid}, `{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25"}}`)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("initialize status = %d", resp.StatusCode)
	}
	if !strings.Contains(string(data), `"protocolVersion":"2025-11-25"`) {
		t.Fatalf("initialize response missing negotiated protocolVersion: %s", string(data))
	}

	// 3. notifications/initialized -> 202 Accepted with empty body.
	resp, data = postJSON(t, mcpURL, map[string]string{"Mcp-Session-Id": sid}, `{"jsonrpc":"2.0","method":"notifications/initialized"}`)
	if resp.StatusCode != http.StatusAccepted || len(bytes.TrimSpace(data)) != 0 {
		t.Fatalf("notification status=%d body=%q, want 202 empty", resp.StatusCode, string(data))
	}

	// 4. tools/list.
	_, data = postJSON(t, mcpURL, nil, `{"jsonrpc":"2.0","id":2,"method":"tools/list"}`)
	var listResp struct {
		Result struct {
			Tools []struct {
				Name        string         `json:"name"`
				InputSchema map[string]any `json:"inputSchema"`
			} `json:"tools"`
		} `json:"result"`
	}
	if err := json.Unmarshal(data, &listResp); err != nil {
		t.Fatalf("unmarshal tools/list: %v", err)
	}
	if len(listResp.Result.Tools) != 4 {
		t.Fatalf("got %d tools, want 4", len(listResp.Result.Tools))
	}
	if listResp.Result.Tools[1].InputSchema["type"] != "object" {
		t.Fatalf("expected normalized schema type=object, got %#v", listResp.Result.Tools[1].InputSchema)
	}

	// 5. tools/call success + hook verification + context propagation.
	_, data = postJSON(t, mcpURL, nil, `{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"echo_tool","arguments":{"msg":"hello"}}}`)
	var callResp struct {
		Result struct {
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
			IsError bool `json:"isError"`
		} `json:"result"`
		Error *rpcError `json:"error"`
	}
	if err := json.Unmarshal(data, &callResp); err != nil || callResp.Error != nil {
		t.Fatalf("tools/call failed: err=%v body=%s", err, string(data))
	}
	if callResp.Result.IsError || len(callResp.Result.Content) != 1 || callResp.Result.Content[0].Text != "hello|marker=ctx-42|wd=" {
		t.Fatalf("unexpected call result: %s", string(data))
	}
	if called.Load() != 1 {
		t.Fatalf("called = %d, want 1", called.Load())
	}
	recorded := hooks.snapshot()
	if len(recorded) != 2 || recorded[0].kind != "pre" || recorded[1].kind != "post" || recorded[0].marker != "ctx-42" || recorded[1].isErr {
		t.Fatalf("unexpected hooks: %#v", recorded)
	}

	// 6. tools/call panic recovery -> isError: true.
	_, data = postJSON(t, mcpURL, nil, `{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"panic_tool","arguments":{}}}`)
	if err := json.Unmarshal(data, &callResp); err != nil {
		t.Fatalf("unmarshal panic call: %v", err)
	}
	if !callResp.Result.IsError || !strings.Contains(callResp.Result.Content[0].Text, "boom") {
		t.Fatalf("expected panic isError=true containing boom, got %s", string(data))
	}

	// 7. tools/call error return -> isError: true.
	_, data = postJSON(t, mcpURL, nil, `{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"err_tool","arguments":{}}}`)
	if err := json.Unmarshal(data, &callResp); err != nil {
		t.Fatalf("unmarshal err call: %v", err)
	}
	if !callResp.Result.IsError || !strings.Contains(callResp.Result.Content[0].Text, "tool failed") {
		t.Fatalf("expected err_tool isError=true, got %s", string(data))
	}

	// 8. tools/call PreToolUse block -> isError: true, tool not called.
	hooks.mu.Lock()
	hooks.preBlock = true
	hooks.preReason = "blocked by policy"
	hooks.mu.Unlock()
	_, data = postJSON(t, mcpURL, nil, `{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"echo_tool","arguments":{"msg":"nope"}}}`)
	if err := json.Unmarshal(data, &callResp); err != nil {
		t.Fatalf("unmarshal blocked call: %v", err)
	}
	if !callResp.Result.IsError || callResp.Result.Content[0].Text != "blocked by policy" {
		t.Fatalf("expected blocked isError=true, got %s", string(data))
	}
	if called.Load() != 1 {
		t.Fatalf("blocked tool should not have been called; count=%d", called.Load())
	}

	// 9. tools/call PreToolUse updated input.
	hooks.mu.Lock()
	hooks.preBlock = false
	hooks.preReason = ""
	hooks.preUpdated = []byte(`{"msg":"rewritten"}`)
	hooks.mu.Unlock()
	_, data = postJSON(t, mcpURL, nil, `{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"echo_tool","arguments":{"msg":"orig"}}}`)
	if err := json.Unmarshal(data, &callResp); err != nil {
		t.Fatalf("unmarshal rewritten call: %v", err)
	}
	if callResp.Result.IsError || !strings.HasPrefix(callResp.Result.Content[0].Text, "rewritten|") {
		t.Fatalf("expected rewritten input output, got %s", string(data))
	}
}

func TestMCPWithArtExMCPHTTPClient(t *testing.T) {
	srv, err := NewServer()
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	defer srv.Close()

	token := srv.RegisterRun(RunConfig{
		Tools:      []actool.CoreTool{makeEchoTool(nil)},
		WorkingDir: "/work",
	})

	client, err := mcphttp.New(context.Background(), "xetra", srv.MCPURL(token), nil, false, 10)
	if err != nil {
		t.Fatalf("mcphttp.New: %v", err)
	}
	defer client.Close()

	tools, err := client.Tools(context.Background())
	if err != nil {
		t.Fatalf("client.Tools: %v", err)
	}
	if len(tools) != 1 || tools[0].Name() != "mcp__xetra__echo_tool" {
		t.Fatalf("unexpected tools: %v", tools)
	}
	out, err := client.Call(context.Background(), "echo_tool", map[string]any{"msg": "via-client"})
	if err != nil {
		t.Fatalf("client.Call: %v", err)
	}
	if !strings.Contains(out, "via-client|marker=|wd=/work") {
		t.Fatalf("unexpected output: %q", out)
	}
}

func TestNormalizeHookTool(t *testing.T) {
	cases := []struct {
		name      string
		inTool    string
		inJSON    string
		wantTool  string
		wantKey   string
		wantValue string
	}{
		{"run_command", "run_command", `{"CommandLine":"curl http://1.2.3.4","Cwd":"/tmp"}`, "Bash", "command", "curl http://1.2.3.4"},
		{"read_url_content", "read_url_content", `{"Url":"https://example.com"}`, "WebFetch", "url", "https://example.com"},
		{"write_to_file", "write_to_file", `{"TargetFile":"/tmp/x.txt","Overwrite":true}`, "Write", "file_path", "/tmp/x.txt"},
		{"replace_file_content", "replace_file_content", `{"TargetFile":"/tmp/y.txt"}`, "Edit", "file_path", "/tmp/y.txt"},
		{"multi_replace_file_content", "multi_replace_file_content", `{"TargetFile":"/tmp/z.txt"}`, "Edit", "file_path", "/tmp/z.txt"},
		{"native Bash", "Bash", `{"command":"id"}`, "Bash", "command", "id"},
		{"mcp tool", "mcp__x__scan", `{"target":"example.com"}`, "mcp__x__scan", "target", "example.com"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gotTool, gotBytes := NormalizeHookTool(tc.inTool, []byte(tc.inJSON))
			if gotTool != tc.wantTool {
				t.Fatalf("tool = %q, want %q", gotTool, tc.wantTool)
			}
			var m map[string]any
			if err := json.Unmarshal(gotBytes, &m); err != nil {
				t.Fatalf("unmarshal normalized: %v", err)
			}
			if m[tc.wantKey] != tc.wantValue {
				t.Fatalf("m[%q] = %v, want %q (full: %s)", tc.wantKey, m[tc.wantKey], tc.wantValue, string(gotBytes))
			}
		})
	}
}

func TestHookEndpointsClaudeAndAgy(t *testing.T) {
	srv, err := NewServer()
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	defer srv.Close()

	hooks := &mockHookRunner{}
	runCtx := context.WithValue(context.Background(), testCtxKey, "hook-ctx")
	token := srv.RegisterRun(RunConfig{
		Context: runCtx,
		Hooks:   hooks,
	})

	// Claude PreToolUse allow.
	resp, data := postJSON(t, srv.HookURL(token, "PreToolUse")+"?cli=claude", nil, `{"tool_name":"Bash","tool_input":{"command":"ls"}}`)
	if resp.StatusCode != http.StatusOK || strings.TrimSpace(string(data)) != "{}" {
		t.Fatalf("claude allow = (%d, %s), want (200, {})", resp.StatusCode, string(data))
	}

	// Agy PreToolUse allow (normalizing run_command -> Bash) + header token route.
	resp, data = postJSON(t, srv.BaseURL()+"/hook/pre-tool-use?cli=agy", map[string]string{"X-Xetra-Bridge-Token": token}, `{"toolCall":{"id":"tc1","name":"run_command","args":{"CommandLine":"nmap 10.0.0.1"}}}`)
	if resp.StatusCode != http.StatusOK || strings.TrimSpace(string(data)) != `{"decision":"allow"}` {
		t.Fatalf("agy allow = (%d, %s)", resp.StatusCode, string(data))
	}

	// Block both dialects.
	hooks.mu.Lock()
	hooks.preBlock = true
	hooks.preReason = "target out of scope"
	hooks.mu.Unlock()

	_, data = postJSON(t, srv.HookURL(token, "PreToolUse")+"?cli=claude", nil, `{"tool_name":"WebFetch","tool_input":{"url":"https://evil.test"}}`)
	var claudeDeny struct {
		HookSpecificOutput struct {
			HookEventName            string `json:"hookEventName"`
			PermissionDecision       string `json:"permissionDecision"`
			PermissionDecisionReason string `json:"permissionDecisionReason"`
		} `json:"hookSpecificOutput"`
	}
	if err := json.Unmarshal(data, &claudeDeny); err != nil {
		t.Fatalf("unmarshal claude deny: %v", err)
	}
	if claudeDeny.HookSpecificOutput.PermissionDecision != "deny" || claudeDeny.HookSpecificOutput.PermissionDecisionReason != "target out of scope" || claudeDeny.HookSpecificOutput.HookEventName != "PreToolUse" {
		t.Fatalf("unexpected claude deny: %s", string(data))
	}

	_, data = postJSON(t, srv.HookURL(token, "PreToolUse")+"?cli=agy", nil, `{"toolCall":{"id":"tc2","name":"read_url_content","args":{"Url":"https://evil.test"}}}`)
	if strings.TrimSpace(string(data)) != `{"decision":"deny","reason":"target out of scope"}` {
		t.Fatalf("unexpected agy deny: %s", string(data))
	}

	// PostToolUse claude + agy.
	_, data = postJSON(t, srv.HookURL(token, "PostToolUse")+"?cli=claude", nil, `{"tool_name":"Bash","tool_input":{"command":"id"},"tool_response":{"stdout":"uid=0"}}`)
	if strings.TrimSpace(string(data)) != "{}" {
		t.Fatalf("claude post = %s", string(data))
	}
	_, data = postJSON(t, srv.HookURL(token, "post-tool-use")+"?cli=agy", nil, `{"toolCall":{"id":"tc3","name":"write_to_file","args":{"TargetFile":"/tmp/f"}},"error":"disk full"}`)
	if strings.TrimSpace(string(data)) != "{}" {
		t.Fatalf("agy post = %s", string(data))
	}

	// Stop hook (prevent = true).
	hooks.mu.Lock()
	hooks.stopPrevent = true
	hooks.stopMessage = "continue scanning"
	hooks.mu.Unlock()
	_, data = postJSON(t, srv.HookURL(token, "Stop")+"?cli=claude", nil, `{}`)
	if !strings.Contains(string(data), `"decision":"block"`) || !strings.Contains(string(data), `"continue scanning"`) {
		t.Fatalf("unexpected stop response: %s", string(data))
	}

	// Inspect recorded normalized HookRunner calls.
	recorded := hooks.snapshot()
	if len(recorded) != 7 {
		t.Fatalf("recorded %d hook calls, want 7: %#v", len(recorded), recorded)
	}
	if recorded[1].tool != "Bash" || !strings.Contains(recorded[1].input, `"command":"nmap 10.0.0.1"`) || recorded[1].marker != "hook-ctx" {
		t.Fatalf("agy run_command not normalized properly: %#v", recorded[1])
	}
	if recorded[3].tool != "WebFetch" || !strings.Contains(recorded[3].input, `"url":"https://evil.test"`) {
		t.Fatalf("agy read_url_content not normalized properly: %#v", recorded[3])
	}
	if recorded[5].kind != "post" || recorded[5].tool != "Write" || !recorded[5].isErr || recorded[5].result != "disk full" {
		t.Fatalf("agy write_to_file PostToolUse mismatch: %#v", recorded[5])
	}
}

func TestLoopbackClientsBypassHTTPProxy(t *testing.T) {
	var proxyHits atomic.Int32
	trapProxy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		proxyHits.Add(1)
		http.Error(w, "proxy should not be contacted", http.StatusBadGateway)
	}))
	defer trapProxy.Close()

	t.Setenv("HTTP_PROXY", trapProxy.URL)
	t.Setenv("HTTPS_PROXY", trapProxy.URL)
	t.Setenv("http_proxy", trapProxy.URL)
	t.Setenv("https_proxy", trapProxy.URL)

	srv, err := NewServer()
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	defer srv.Close()

	hooks := &mockHookRunner{preBlock: true, preReason: "no proxy"}
	token := srv.RegisterRun(RunConfig{
		Tools: []actool.CoreTool{makeEchoTool(nil)},
		Hooks: hooks,
	})

	// 1. RunHookClient with env vars (base URL + token + event + cli).
	t.Setenv("XETRA_BRIDGE_URL", srv.BaseURL())
	t.Setenv("XETRA_BRIDGE_TOKEN", token)
	t.Setenv("XETRA_HOOK_EVENT", "PreToolUse")
	t.Setenv("XETRA_HOOK_CLI", "agy")

	var hookOut bytes.Buffer
	err = RunHookClient(context.Background(), nil, strings.NewReader(`{"toolCall":{"id":"1","name":"run_command","args":{"CommandLine":"ls"}}}`), &hookOut)
	if err != nil {
		t.Fatalf("RunHookClient: %v", err)
	}
	if strings.TrimSpace(hookOut.String()) != `{"decision":"deny","reason":"no proxy"}` {
		t.Fatalf("RunHookClient output = %q", hookOut.String())
	}

	// 2. RunStdioBridge with --url / --token flags (allow hooks now).
	hooks.mu.Lock()
	hooks.preBlock = false
	hooks.preReason = ""
	hooks.mu.Unlock()

	stdioIn := strings.Join([]string{
		`{"jsonrpc":"2.0","id":1,"method":"server/discover"}`,
		`{"jsonrpc":"2.0","id":2,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}`,
		`{"jsonrpc":"2.0","method":"notifications/initialized"}`,
		`{"jsonrpc":"2.0","id":3,"method":"tools/list"}`,
		`{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"echo_tool","arguments":{"msg":"stdio-ok"}}}`,
		"",
	}, "\n")
	var stdioOut bytes.Buffer
	if err := RunStdioBridge(context.Background(), []string{"--url", srv.BaseURL(), "--token", token}, strings.NewReader(stdioIn), &stdioOut); err != nil {
		t.Fatalf("RunStdioBridge: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(stdioOut.String()), "\n")
	if len(lines) != 4 {
		t.Fatalf("got %d stdio response lines, want 4:\n%s", len(lines), stdioOut.String())
	}
	joined := stdioOut.String()
	if !strings.Contains(joined, `"code":-32601`) || !strings.Contains(joined, `"stdio-ok|`) {
		t.Fatalf("unexpected stdio output:\n%s", joined)
	}

	if proxyHits.Load() != 0 {
		t.Fatalf("trap proxy was hit %d times, want 0", proxyHits.Load())
	}
}
