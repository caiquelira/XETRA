package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/Autumn-27/artex/mcpbridge"
)

func main() {
	if len(os.Args) > 1 && os.Args[1] == "hook" {
		sigCtx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
		defer stop()
		if err := mcpbridge.RunHookClient(sigCtx, os.Args[2:], os.Stdin, os.Stdout); err != nil {
			fmt.Fprintf(os.Stderr, "fakecli hook: %v\n", err)
			os.Exit(1)
		}
		return
	}

	if len(os.Args) > 1 && os.Args[1] == "__grandchild__" {
		runGrandchild(os.Args[2:])
		return
	}

	// Verify environment sanitization invariants on every CLI invocation.
	if val, ok := os.LookupEnv("CLAUDECODE"); ok && val != "" {
		fmt.Fprintf(os.Stderr, "CLAUDECODE leaked into subprocess env: %q\n", val)
		os.Exit(99)
	}
	if val, ok := os.LookupEnv("CLAUDE_CODE_ENTRYPOINT"); ok && val != "" {
		fmt.Fprintf(os.Stderr, "CLAUDE_CODE_ENTRYPOINT leaked into subprocess env: %q\n", val)
		os.Exit(99)
	}
	noProxy := strings.ToLower(os.Getenv("NO_PROXY") + "," + os.Getenv("no_proxy"))
	for _, host := range []string{"127.0.0.1", "localhost", "::1"} {
		if !strings.Contains(noProxy, host) {
			fmt.Fprintf(os.Stderr, "NO_PROXY/no_proxy missing %s: NO_PROXY=%q no_proxy=%q\n", host, os.Getenv("NO_PROXY"), os.Getenv("no_proxy"))
			os.Exit(98)
		}
	}

	if err := runFakeCLI(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func runGrandchild(args []string) {
	if len(args) < 1 {
		fmt.Fprintln(os.Stderr, "missing pid file")
		os.Exit(2)
	}
	pidFile := args[0]
	tmp := pidFile + ".tmp"
	if err := os.WriteFile(tmp, []byte(strconv.Itoa(os.Getpid())), 0o600); err != nil {
		fmt.Fprintf(os.Stderr, "write tmp pid: %v\n", err)
		os.Exit(2)
	}
	if err := os.Rename(tmp, pidFile); err != nil {
		fmt.Fprintf(os.Stderr, "rename pid: %v\n", err)
		os.Exit(2)
	}
	// Emit a harmless line so the parent knows the grandchild has attached to stdout.
	fmt.Fprintln(os.Stdout, `{"type":"system","subtype":"grandchild_ready"}`)
	time.Sleep(60 * time.Second)
}

func runFakeCLI() error {
	isAgy := slices.Contains(os.Args[1:], "--dangerously-skip-permissions")
	model := flagVal("--model")
	effort := flagVal("--effort")
	sessionID := flagVal("--session-id")
	resumeID := flagVal("--resume")
	convID := flagVal("--conversation")
	mcpConfigPath := flagVal("--mcp-config")
	settingsPath := flagVal("--settings")
	sysPrompt := flagVal("--system-prompt")

	var prompt string
	if isAgy {
		for _, a := range os.Args[1:] {
			if strings.HasPrefix(a, "-p=") {
				prompt = strings.TrimPrefix(a, "-p=")
				break
			}
		}
		if data, err := os.ReadFile(".xetra-prompt.md"); err == nil {
			prompt = string(data)
		}
		if data, err := os.ReadFile(filepath.Join(".agents", "plugins", "xetra", "rules", "AGENTS.md")); err == nil && sysPrompt == "" {
			sysPrompt = string(data)
		}
	} else {
		stdinBytes, err := io.ReadAll(os.Stdin)
		if err != nil {
			return fmt.Errorf("read stdin: %w", err)
		}
		prompt = string(stdinBytes)
	}

	// Optional invocation trace file for assertions in tests.
	if traceFile := os.Getenv("ARTEX_FAKE_TRACE_FILE"); traceFile != "" {
		rec := map[string]any{
			"is_agy":      isAgy,
			"args":        os.Args[1:],
			"model":       model,
			"effort":      effort,
			"session_id":  sessionID,
			"resume_id":   resumeID,
			"conv_id":     convID,
			"prompt":      prompt,
			"system":      sysPrompt,
			"http_proxy":  os.Getenv("HTTP_PROXY"),
			"https_proxy": os.Getenv("HTTPS_PROXY"),
			"no_proxy":    os.Getenv("NO_PROXY"),
		}
		if line, err := json.Marshal(rec); err == nil {
			f, err := os.OpenFile(traceFile, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
			if err == nil {
				_, _ = f.Write(append(line, '\n'))
				_ = f.Close()
			}
		}
	}

	switch model {
	case "model-empty-text":
		emitResult(isAgy, "sess-empty", "", false, "", 10, 0, 0, 0)
		return nil

	case "model-echo-prompt":
		reply := fmt.Sprintf("SYSTEM=%s|PROMPT=%s|EFFORT=%s", sysPrompt, prompt, effort)
		emitInit(isAgy, "sess-echo")
		emitText(isAgy, reply)
		emitResult(isAgy, "sess-echo", reply, false, "", 50, 15, 5, 2)
		return nil

	case "model-stream-chunks":
		emitInit(isAgy, "sess-chunks")
		emitThinking(isAgy, "think-1 ")
		emitThinking(isAgy, "think-2")
		emitText(isAgy, "part-1 ")
		emitText(isAgy, "part-2")
		emitResult(isAgy, "sess-chunks", "part-1 part-2", false, "", 120, 45, 20, 10)
		return nil

	case "model-quota-result":
		emitInit(isAgy, "sess-quota")
		emitResult(isAgy, "sess-quota", "", true, "You have hit your usage limit", 5, 0, 0, 0)
		os.Exit(1)

	case "model-quota-stderr":
		fmt.Fprintln(os.Stderr, "Error: insufficient_quota - Credit balance is too low")
		os.Exit(1)

	case "model-exit-error-no-result":
		fmt.Fprintln(os.Stderr, "fatal stderr failure line")
		os.Exit(2)

	case "model-usage-only":
		emitInit(isAgy, "sess-usage")
		emitUsage(isAgy, 300, 75, 40)
		emitText(isAgy, "budget ok")
		emitResult(isAgy, "sess-usage", "budget ok", false, "", 500, 120, 60, 10)
		return nil

	case "model-hang-grandchild":
		pidFile := strings.TrimSpace(prompt)
		if envPid := strings.TrimSpace(os.Getenv("ARTEX_FAKE_PID_FILE")); envPid != "" {
			pidFile = envPid
		}
		grand := exec.Command(os.Args[0], "__grandchild__", pidFile)
		grand.Env = os.Environ()
		grand.Stdout = os.Stdout
		grand.Stderr = os.Stderr
		if err := grand.Start(); err != nil {
			return fmt.Errorf("start grandchild: %w", err)
		}
		emitInit(isAgy, "sess-hang")
		emitToolUse(isAgy, 1, "tu_hang", "Bash", json.RawMessage(`{"command":"sleep 60"}`))
		time.Sleep(60 * time.Second)
		_ = grand.Wait()
		return nil

	case "model-session-resume":
		return handleSessionResume(isAgy, sessionID, resumeID, convID)

	case "model-mcp-and-hooks":
		return handleMCPAndHooks(isAgy, mcpConfigPath, settingsPath)

	default:
		// Default completion (e.g. "sonnet", "gemini-4-hydrogen-high") for TestConnection and general callers.
		sid := sessionID
		if sid == "" {
			sid = "sess-default-1"
		}
		emitInit(isAgy, sid)
		emitThinking(isAgy, "checking prompt")
		emitText(isAgy, "OK")
		emitResult(isAgy, sid, "OK", false, "", 40, 8, 12, 4)
		return nil
	}
	return nil
}

type fakeSessionState struct {
	Valid   []string `json:"valid"`
	Counter int      `json:"counter"`
}

func handleSessionResume(isAgy bool, sessionID, resumeID, convID string) error {
	statePath := ".fake-session-state.json"
	var st fakeSessionState
	if data, err := os.ReadFile(statePath); err == nil {
		_ = json.Unmarshal(data, &st)
	}

	targetResume := resumeID
	if isAgy {
		targetResume = convID
	}

	if targetResume != "" {
		if targetResume == "stale-stderr-id" {
			fmt.Fprintln(os.Stderr, "session not found: "+targetResume)
			os.Exit(1)
		}
		if !slices.Contains(st.Valid, targetResume) {
			msg := "No conversation found with session ID: " + targetResume
			if isAgy {
				msg = "conversation not found: " + targetResume
			}
			emitResult(isAgy, targetResume, "", true, msg, 0, 0, 0, 0)
			os.Exit(1)
		}
		emitInit(isAgy, targetResume)
		reply := "resumed:" + targetResume
		emitText(isAgy, reply)
		emitResult(isAgy, targetResume, reply, false, "", 30, 10, 5, 0)
		return nil
	}

	st.Counter++
	newID := sessionID
	if newID == "" {
		newID = fmt.Sprintf("agy-conv-%d", st.Counter)
	}
	st.Valid = append(st.Valid, newID)
	if data, err := json.Marshal(st); err == nil {
		_ = os.WriteFile(statePath, data, 0o600)
	}

	emitInit(isAgy, newID)
	reply := "fresh:" + newID
	emitText(isAgy, reply)
	emitResult(isAgy, newID, reply, false, "", 25, 10, 0, 0)
	return nil
}

func handleMCPAndHooks(isAgy bool, mcpConfigPath, settingsPath string) error {
	sid := "sess-mcp-1"
	emitInit(isAgy, sid)
	emitThinking(isAgy, "testing hooks and mcp")

	preCmd, postCmd, err := loadHookCommands(isAgy, settingsPath)
	if err != nil {
		return err
	}

	// 1. PreToolUse blocked command check.
	blockedIn := json.RawMessage(`{"command":"rm -rf /blocked"}`)
	if isAgy {
		blockedIn = json.RawMessage(`{"CommandLine":"rm -rf /blocked"}`)
	}
	emitToolUse(isAgy, 1, "tu_block_hook", bashToolName(isAgy), blockedIn)
	decision, reason, err := invokeHook(preCmd, buildHookPayload(isAgy, bashToolName(isAgy), blockedIn, "", false))
	if err != nil {
		return fmt.Errorf("pre-hook blocked call failed: %w", err)
	}
	if decision != "deny" {
		return fmt.Errorf("expected PreToolUse deny decision, got %q (%s)", decision, reason)
	}
	emitToolResult(isAgy, 1, "tu_block_hook", bashToolName(isAgy), "blocked:"+reason, true)

	// 2. PreToolUse allowed + PostToolUse.
	allowIn := json.RawMessage(`{"command":"echo allowed"}`)
	if isAgy {
		allowIn = json.RawMessage(`{"CommandLine":"echo allowed"}`)
	}
	emitToolUse(isAgy, 2, "tu_allow_hook", bashToolName(isAgy), allowIn)
	decision, reason, err = invokeHook(preCmd, buildHookPayload(isAgy, bashToolName(isAgy), allowIn, "", false))
	if err != nil {
		return fmt.Errorf("pre-hook allow call failed: %w", err)
	}
	if decision == "deny" {
		return fmt.Errorf("unexpected deny on allowed Bash call: %s", reason)
	}
	if postCmd != "" {
		if _, _, err := invokeHook(postCmd, buildHookPayload(isAgy, bashToolName(isAgy), allowIn, "allowed output", false)); err != nil {
			return fmt.Errorf("post-hook call failed: %w", err)
		}
	}
	emitToolResult(isAgy, 2, "tu_allow_hook", bashToolName(isAgy), "allowed output", false)

	// 3. MCP server tools/list and tools/call (allowed + blocked by hook inside mcpbridge).
	mcpURL, err := loadMCPURL(isAgy, mcpConfigPath)
	if err != nil {
		return err
	}
	client := &http.Client{Transport: &http.Transport{Proxy: nil}, Timeout: 10 * time.Second}
	mcpSessID, err := mcpInitialize(client, mcpURL)
	if err != nil {
		return err
	}
	toolNames, err := mcpListTools(client, mcpURL, mcpSessID)
	if err != nil {
		return err
	}
	if !slices.Contains(toolNames, "echo_tool") {
		return fmt.Errorf("tools/list missing echo_tool: %v", toolNames)
	}

	echoArgs := json.RawMessage(`{"msg":"hello-mcp"}`)
	emitMCPToolUse(isAgy, 3, "tu_mcp_ok", "echo_tool", echoArgs)
	outText, isErr, err := mcpCallTool(client, mcpURL, mcpSessID, "echo_tool", echoArgs)
	if err != nil {
		return fmt.Errorf("mcp tools/call echo_tool: %w", err)
	}
	emitToolResult(isAgy, 3, "tu_mcp_ok", "echo_tool", outText, isErr)

	denyArgs := json.RawMessage(`{"msg":"deny-me"}`)
	emitMCPToolUse(isAgy, 4, "tu_mcp_deny", "blocked_tool", denyArgs)
	denyText, denyErr, err := mcpCallTool(client, mcpURL, mcpSessID, "blocked_tool", denyArgs)
	if err != nil {
		return fmt.Errorf("mcp tools/call blocked_tool: %w", err)
	}
	emitToolResult(isAgy, 4, "tu_mcp_deny", "blocked_tool", denyText, denyErr)

	emitUsage(isAgy, 180, 50, 20)
	finalMsg := "MCP and hook run complete: " + outText
	emitText(isAgy, finalMsg)
	emitResult(isAgy, sid, finalMsg, false, "", 220, 65, 30, 5)
	return nil
}

func bashToolName(isAgy bool) string {
	if isAgy {
		return "run_command"
	}
	return "Bash"
}

func loadHookCommands(isAgy bool, settingsPath string) (string, string, error) {
	path := settingsPath
	if isAgy {
		path = filepath.Join(".agents", "plugins", "xetra", "hooks.json")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return "", "", fmt.Errorf("read hooks file %q: %w", path, err)
	}
	var cfg struct {
		Hooks map[string][]struct {
			Hooks []struct {
				Command string `json:"command"`
			} `json:"hooks"`
		} `json:"hooks"`
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return "", "", fmt.Errorf("parse hooks file %q: %w", path, err)
	}
	var pre, post string
	if len(cfg.Hooks["PreToolUse"]) > 0 && len(cfg.Hooks["PreToolUse"][0].Hooks) > 0 {
		pre = cfg.Hooks["PreToolUse"][0].Hooks[0].Command
	}
	if len(cfg.Hooks["PostToolUse"]) > 0 && len(cfg.Hooks["PostToolUse"][0].Hooks) > 0 {
		post = cfg.Hooks["PostToolUse"][0].Hooks[0].Command
	}
	if pre == "" {
		return "", "", fmt.Errorf("no PreToolUse command in %q", path)
	}
	return pre, post, nil
}

func invokeHook(cmdStr string, payload []byte) (decision string, reason string, err error) {
	parts, err := splitQuotedCommand(cmdStr)
	if err != nil || len(parts) == 0 {
		return "", "", fmt.Errorf("parse hook command %q: %v", cmdStr, err)
	}
	cmd := exec.Command(parts[0], parts[1:]...)
	cmd.Stdin = bytes.NewReader(payload)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return "", "", fmt.Errorf("run hook %v: %w (%s)", parts, err, strings.TrimSpace(stderr.String()))
	}
	raw := bytes.TrimSpace(stdout.Bytes())
	if len(raw) == 0 {
		return "allow", "", nil
	}
	var agyResp struct {
		Decision string `json:"decision"`
		Reason   string `json:"reason"`
	}
	if json.Unmarshal(raw, &agyResp) == nil && agyResp.Decision != "" {
		return agyResp.Decision, agyResp.Reason, nil
	}
	var claudeResp struct {
		HookSpecificOutput struct {
			PermissionDecision       string `json:"permissionDecision"`
			PermissionDecisionReason string `json:"permissionDecisionReason"`
		} `json:"hookSpecificOutput"`
	}
	if json.Unmarshal(raw, &claudeResp) == nil && claudeResp.HookSpecificOutput.PermissionDecision != "" {
		return claudeResp.HookSpecificOutput.PermissionDecision, claudeResp.HookSpecificOutput.PermissionDecisionReason, nil
	}
	return "allow", "", nil
}

func splitQuotedCommand(s string) ([]string, error) {
	var args []string
	var cur strings.Builder
	inQuotes := false
	hasToken := false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c == '\\' && i+1 < len(s) && s[i+1] == '"' {
			cur.WriteByte('"')
			i++
			hasToken = true
			continue
		}
		if c == '"' {
			inQuotes = !inQuotes
			hasToken = true
			continue
		}
		if !inQuotes && (c == ' ' || c == '\t') {
			if hasToken {
				args = append(args, cur.String())
				cur.Reset()
				hasToken = false
			}
			continue
		}
		cur.WriteByte(c)
		hasToken = true
	}
	if inQuotes {
		return nil, fmt.Errorf("unterminated quote")
	}
	if hasToken {
		args = append(args, cur.String())
	}
	return args, nil
}

func buildHookPayload(isAgy bool, toolName string, input json.RawMessage, output string, isErr bool) []byte {
	if isAgy {
		p := map[string]any{
			"toolCall": map[string]any{
				"name": toolName,
				"args": input,
			},
			"is_error": isErr,
		}
		if output != "" {
			p["output"] = output
		}
		b, _ := json.Marshal(p)
		return b
	}
	p := map[string]any{
		"tool_name":  toolName,
		"tool_input": input,
		"is_error":   isErr,
	}
	if output != "" {
		p["tool_response"] = output
	}
	b, _ := json.Marshal(p)
	return b
}

func loadMCPURL(isAgy bool, mcpConfigPath string) (string, error) {
	path := mcpConfigPath
	serverKey := "xetra"
	if isAgy {
		path = filepath.Join(".agents", "plugins", "xetra", "mcp_config.json")
		serverKey = "bridge"
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read mcp config %q: %w", path, err)
	}
	var cfg struct {
		MCPServers map[string]struct {
			URL string `json:"url"`
		} `json:"mcpServers"`
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return "", fmt.Errorf("parse mcp config %q: %w", path, err)
	}
	u := cfg.MCPServers[serverKey].URL
	if u == "" {
		return "", fmt.Errorf("missing server %q url in %q", serverKey, path)
	}
	return u, nil
}

func mcpPost(client *http.Client, url, sessionID string, payload any) (string, []byte, error) {
	body, _ := json.Marshal(payload)
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return "", nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	if sessionID != "" {
		req.Header.Set("Mcp-Session-Id", sessionID)
	}
	resp, err := client.Do(req)
	if err != nil {
		return "", nil, err
	}
	defer resp.Body.Close()
	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(respBytes)))
	}
	newSID := resp.Header.Get("Mcp-Session-Id")
	if newSID == "" {
		newSID = sessionID
	}
	return newSID, respBytes, nil
}

func mcpInitialize(client *http.Client, mcpURL string) (string, error) {
	sid, _, err := mcpPost(client, mcpURL, "", map[string]any{
		"jsonrpc": "2.0",
		"id":      1,
		"method":  "initialize",
		"params": map[string]any{
			"protocolVersion": "2025-06-18",
			"capabilities":    map[string]any{},
			"clientInfo":      map[string]any{"name": "fakecli", "version": "1.0"},
		},
	})
	if err != nil {
		return "", err
	}
	_, _, err = mcpPost(client, mcpURL, sid, map[string]any{
		"jsonrpc": "2.0",
		"method":  "notifications/initialized",
	})
	return sid, err
}

func mcpListTools(client *http.Client, mcpURL, sid string) ([]string, error) {
	_, body, err := mcpPost(client, mcpURL, sid, map[string]any{
		"jsonrpc": "2.0",
		"id":      2,
		"method":  "tools/list",
	})
	if err != nil {
		return nil, err
	}
	var resp struct {
		Result struct {
			Tools []struct {
				Name string `json:"name"`
			} `json:"tools"`
		} `json:"result"`
	}
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, err
	}
	var names []string
	for _, t := range resp.Result.Tools {
		names = append(names, t.Name)
	}
	return names, nil
}

func mcpCallTool(client *http.Client, mcpURL, sid, name string, args json.RawMessage) (string, bool, error) {
	_, body, err := mcpPost(client, mcpURL, sid, map[string]any{
		"jsonrpc": "2.0",
		"id":      3,
		"method":  "tools/call",
		"params": map[string]any{
			"name":      name,
			"arguments": args,
		},
	})
	if err != nil {
		return "", false, err
	}
	var resp struct {
		Result struct {
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
			IsError bool `json:"isError"`
		} `json:"result"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &resp); err != nil {
		return "", false, err
	}
	if resp.Error != nil {
		return resp.Error.Message, true, nil
	}
	var texts []string
	for _, c := range resp.Result.Content {
		texts = append(texts, c.Text)
	}
	return strings.Join(texts, "\n"), resp.Result.IsError, nil
}

func flagVal(name string) string {
	for i := 1; i < len(os.Args)-1; i++ {
		if os.Args[i] == name {
			return os.Args[i+1]
		}
	}
	return ""
}

func emitJSON(v any) {
	b, _ := json.Marshal(v)
	fmt.Println(string(b))
}

func emitInit(isAgy bool, sid string) {
	if isAgy {
		emitJSON(map[string]any{
			"event":           "init",
			"conversation_id": sid,
			"init":            map[string]any{},
		})
		return
	}
	emitJSON(map[string]any{
		"type":       "system",
		"subtype":    "init",
		"session_id": sid,
	})
}

func emitThinking(isAgy bool, text string) {
	if isAgy {
		emitJSON(map[string]any{
			"event": "step_update",
			"step_update": map[string]any{
				"step_index": 0,
				"state":      "ACTIVE",
				"step_type":  "thought",
				"text_delta": text,
			},
		})
		return
	}
	emitJSON(map[string]any{
		"type": "assistant",
		"message": map[string]any{
			"content": []map[string]any{{"type": "thinking", "thinking": text}},
		},
	})
}

func emitText(isAgy bool, text string) {
	if isAgy {
		emitJSON(map[string]any{
			"event": "step_update",
			"step_update": map[string]any{
				"step_index": 10,
				"state":      "ACTIVE",
				"step_type":  "agent_response",
				"text_delta": text,
			},
		})
		return
	}
	emitJSON(map[string]any{
		"type": "assistant",
		"message": map[string]any{
			"content": []map[string]any{{"type": "text", "text": text}},
		},
	})
}

func emitToolUse(isAgy bool, step int, id, name string, input json.RawMessage) {
	if isAgy {
		emitJSON(map[string]any{
			"event": "step_update",
			"step_update": map[string]any{
				"step_index": step,
				"state":      "ACTIVE",
				"step_type":  "tool",
				"tool_name":  name,
				"tool_info": map[string]any{
					"id":         id,
					"name":       name,
					"parameters": input,
				},
			},
		})
		return
	}
	emitJSON(map[string]any{
		"type": "assistant",
		"message": map[string]any{
			"content": []map[string]any{{
				"type":  "tool_use",
				"id":    id,
				"name":  name,
				"input": input,
			}},
		},
	})
}

func emitMCPToolUse(isAgy bool, step int, id, name string, args json.RawMessage) {
	if isAgy {
		wrapped, _ := json.Marshal(map[string]any{
			"ServerName": "xetra_bridge",
			"ToolName":   name,
			"Arguments":  args,
		})
		emitToolUse(true, step, id, "call_mcp_tool", wrapped)
		return
	}
	emitToolUse(false, step, id, name, args)
}

func emitToolResult(isAgy bool, step int, id, name, text string, isErr bool) {
	if isAgy {
		info := map[string]any{
			"id":   id,
			"name": name,
		}
		if isErr {
			info["error"] = text
		} else {
			info["output"] = text
		}
		emitJSON(map[string]any{
			"event": "step_update",
			"step_update": map[string]any{
				"step_index": step,
				"state":      "DONE",
				"step_type":  "tool",
				"tool_name":  name,
				"tool_info":  info,
			},
		})
		return
	}
	emitJSON(map[string]any{
		"type": "user",
		"message": map[string]any{
			"content": []map[string]any{{
				"type":        "tool_result",
				"tool_use_id": id,
				"content":     text,
				"is_error":    isErr,
			}},
		},
	})
}

func emitUsage(isAgy bool, in, out, cacheRead int) {
	if !isAgy {
		return
	}
	emitJSON(map[string]any{
		"event": "step_update",
		"step_update": map[string]any{
			"step_index": 99,
			"state":      "DONE",
			"step_type":  "usage_step",
			"usage": map[string]any{
				"input_tokens":      in,
				"output_tokens":     out,
				"cache_read_tokens": cacheRead,
			},
		},
	})
}

func emitResult(isAgy bool, sid, text string, isErr bool, errText string, in, out, cacheRead, cacheWrite int) {
	if isAgy {
		status := "SUCCESS"
		if isErr {
			status = "ERROR"
		}
		emitJSON(map[string]any{
			"event": "result",
			"result": map[string]any{
				"conversation_id": sid,
				"status":          status,
				"response":        text,
				"error":           errText,
				"num_turns":       1,
				"finished_reason": "STOP",
				"usage": map[string]any{
					"input_tokens":      in,
					"output_tokens":     out,
					"cache_read_tokens": cacheRead,
				},
			},
		})
		return
	}
	subtype := "success"
	var errs []string
	if isErr {
		subtype = "error_during_execution"
		if errText != "" {
			errs = []string{errText}
		}
	}
	emitJSON(map[string]any{
		"type":       "result",
		"subtype":    subtype,
		"is_error":   isErr,
		"result":     text,
		"session_id": sid,
		"num_turns":  1,
		"errors":     errs,
		"usage": map[string]any{
			"input_tokens":                in,
			"output_tokens":               out,
			"cache_read_input_tokens":     cacheRead,
			"cache_creation_input_tokens": cacheWrite,
		},
	})
}
