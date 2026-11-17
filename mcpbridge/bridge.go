// Package mcpbridge exposes an ephemeral loopback HTTP server bridging CLI
// agents (Claude Code `claude` and Antigravity CLI `agy`) to ARTEX's
// in-process tool registry ([]tool.CoreTool) and safety hooks (harness.HookRunner).
package mcpbridge

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/Autumn-27/norma/harness"
	actool "github.com/Autumn-27/norma/tool"
)

const (
	defaultProtocolVersion = "2025-06-18"
	maxBodyBytes           = 16 << 20 // 16 MiB
)

// RunConfig configures one active CLI agent run exposed via the bridge.
type RunConfig struct {
	// Context is the base context for the run (typically carrying TargetScope,
	// ConvID, and run cancellation). Defaults to context.Background() if nil.
	Context context.Context
	// Tools are exposed over Streamable HTTP MCP at /mcp/{token}.
	Tools []actool.CoreTool
	// Hooks are invoked for MCP tools/call and CLI hook callbacks at /hook/{token}/{event}.
	Hooks harness.HookRunner
	// WorkingDir, when non-empty, is passed via ToolContext{WorkingDir} to CoreTool.Call.
	// When empty, nil is passed as ToolContext.
	WorkingDir string
}

type runEntry struct {
	ctx        context.Context
	cancel     context.CancelFunc
	sessionID  string
	tools      map[string]actool.CoreTool
	toolOrder  []actool.CoreTool
	hooks      harness.HookRunner
	workingDir string
}

// Server hosts the ephemeral loopback MCP and Hook HTTP endpoints for active runs.
type Server struct {
	ln      net.Listener
	httpSrv *http.Server

	mu   sync.RWMutex
	runs map[string]*runEntry
}

// NewServer starts an ephemeral loopback HTTP server on 127.0.0.1:0.
func NewServer() (*Server, error) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, fmt.Errorf("mcpbridge: listen: %w", err)
	}
	s := &Server{
		ln:   ln,
		runs: make(map[string]*runEntry),
	}
	s.httpSrv = &http.Server{
		Handler:           s.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
	}
	go func() {
		if err := s.httpSrv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
			// Listener closed or unexpected error; nothing else to do here.
		}
	}()
	return s, nil
}

// Addr returns the bound loopback host:port (e.g. "127.0.0.1:54321").
func (s *Server) Addr() string {
	return s.ln.Addr().String()
}

// BaseURL returns the HTTP base URL (e.g. "http://127.0.0.1:54321").
func (s *Server) BaseURL() string {
	return "http://" + s.Addr()
}

// MCPURL returns the Streamable HTTP MCP endpoint URL for the given run token.
func (s *Server) MCPURL(token string) string {
	return s.BaseURL() + "/mcp/" + token
}

// HookURL returns the HTTP hook endpoint URL for the given run token and event.
func (s *Server) HookURL(token, event string) string {
	return s.BaseURL() + "/hook/" + token + "/" + event
}

// Handler returns the HTTP handler serving /mcp/ and /hook/ routes.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/mcp/", s.handleMCP)
	mux.HandleFunc("/hook/", s.handleHook)
	return mux
}

// Close shuts down the HTTP server and cancels all registered runs.
func (s *Server) Close() error {
	s.mu.Lock()
	for token, entry := range s.runs {
		entry.cancel()
		delete(s.runs, token)
	}
	s.mu.Unlock()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	return s.httpSrv.Shutdown(ctx)
}

// RegisterRun registers a run's tools and hooks and returns a unique bearer token.
func (s *Server) RegisterRun(cfg RunConfig) string {
	baseCtx := cfg.Context
	if baseCtx == nil {
		baseCtx = context.Background()
	}
	ctx, cancel := context.WithCancel(baseCtx)

	toolMap := make(map[string]actool.CoreTool, len(cfg.Tools))
	order := make([]actool.CoreTool, 0, len(cfg.Tools))
	for _, t := range cfg.Tools {
		if t == nil {
			continue
		}
		name := t.Name()
		if _, exists := toolMap[name]; !exists {
			order = append(order, t)
		}
		toolMap[name] = t
	}

	token := randomHex(16)
	sessionID := randomHex(16)

	s.mu.Lock()
	s.runs[token] = &runEntry{
		ctx:        ctx,
		cancel:     cancel,
		sessionID:  sessionID,
		tools:      toolMap,
		toolOrder:  order,
		hooks:      cfg.Hooks,
		workingDir: cfg.WorkingDir,
	}
	s.mu.Unlock()
	return token
}

// UnregisterRun removes the run associated with token and cancels its context.
func (s *Server) UnregisterRun(token string) {
	s.mu.Lock()
	entry, ok := s.runs[token]
	if ok {
		delete(s.runs, token)
	}
	s.mu.Unlock()
	if ok && entry.cancel != nil {
		entry.cancel()
	}
}

func (s *Server) lookupRun(token string) (*runEntry, bool) {
	if token == "" {
		return nil, false
	}
	s.mu.RLock()
	entry, ok := s.runs[token]
	s.mu.RUnlock()
	return entry, ok
}

func randomHex(nBytes int) string {
	b := make([]byte, nBytes)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

func mergeContext(runCtx, reqCtx context.Context) (context.Context, context.CancelFunc) {
	if runCtx == nil {
		runCtx = context.Background()
	}
	ctx, cancel := context.WithCancel(runCtx)
	if reqCtx != nil {
		stop := context.AfterFunc(reqCtx, cancel)
		return ctx, func() {
			stop()
			cancel()
		}
	}
	return ctx, cancel
}

// --- Streamable HTTP MCP (/mcp/{token}) ---

type rpcRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type rpcResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id"`
	Result  any             `json:"result,omitempty"`
	Error   *rpcError       `json:"error,omitempty"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

func (s *Server) handleMCP(w http.ResponseWriter, r *http.Request) {
	token := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/mcp/"), "/")
	if token == "" || strings.Contains(token, "/") {
		if hdr := strings.TrimSpace(r.Header.Get("X-Xetra-Bridge-Token")); hdr != "" && token == "" {
			token = hdr
		}
	}
	entry, ok := s.lookupRun(token)
	if !ok {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return
	}

	sessionID := strings.TrimSpace(r.Header.Get("Mcp-Session-Id"))
	if sessionID == "" {
		sessionID = entry.sessionID
	}
	w.Header().Set("Mcp-Session-Id", sessionID)

	switch r.Method {
	case http.MethodGet:
		w.Header().Set("Allow", "POST, DELETE")
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	case http.MethodDelete:
		w.WriteHeader(http.StatusOK)
		return
	case http.MethodPost:
		// Handled below.
	default:
		w.Header().Set("Allow", "POST, DELETE")
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(io.LimitReader(r.Body, maxBodyBytes))
	if err != nil {
		writeRPCError(w, json.RawMessage("null"), -32700, "read request body: "+err.Error())
		return
	}

	var req rpcRequest
	if err := json.Unmarshal(body, &req); err != nil {
		writeRPCError(w, json.RawMessage("null"), -32700, "invalid JSON: "+err.Error())
		return
	}
	if req.Method == "" {
		writeRPCError(w, req.ID, -32600, "invalid request: missing method")
		return
	}

	// JSON-RPC notifications have no "id" (or "id": null) and return 202 Accepted with empty body.
	trimmedID := bytes.TrimSpace(req.ID)
	if len(trimmedID) == 0 || bytes.Equal(trimmedID, []byte("null")) {
		w.WriteHeader(http.StatusAccepted)
		return
	}

	switch req.Method {
	case "initialize":
		s.handleMCPInitialize(w, req)
	case "ping":
		writeRPCResult(w, req.ID, map[string]any{})
	case "tools/list":
		s.handleMCPToolsList(w, req, entry)
	case "tools/call":
		ctx, cancel := mergeContext(entry.ctx, r.Context())
		defer cancel()
		s.handleMCPToolsCall(ctx, w, req, entry)
	default:
		// Includes agy's initial "server/discover" probe: returning -32601 with
		// HTTP 200 application/json + Mcp-Session-Id causes agy to proceed to initialize.
		writeRPCError(w, req.ID, -32601, "Method not found")
	}
}

func (s *Server) handleMCPInitialize(w http.ResponseWriter, req rpcRequest) {
	var params struct {
		ProtocolVersion string `json:"protocolVersion"`
	}
	if len(req.Params) > 0 {
		_ = json.Unmarshal(req.Params, &params)
	}
	proto := defaultProtocolVersion
	switch params.ProtocolVersion {
	case "2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25":
		proto = params.ProtocolVersion
	}
	writeRPCResult(w, req.ID, map[string]any{
		"protocolVersion": proto,
		"capabilities": map[string]any{
			"tools": map[string]any{},
		},
		"serverInfo": map[string]any{
			"name":    "xetra-mcp-bridge",
			"version": "1.0.0",
		},
	})
}

func (s *Server) handleMCPToolsList(w http.ResponseWriter, req rpcRequest, entry *runEntry) {
	type mcpToolDef struct {
		Name        string         `json:"name"`
		Description string         `json:"description,omitempty"`
		InputSchema map[string]any `json:"inputSchema"`
	}
	tools := make([]mcpToolDef, 0, len(entry.toolOrder))
	for _, t := range entry.toolOrder {
		schema := normalizeInputSchema(t.InputSchema())
		tools = append(tools, mcpToolDef{
			Name:        t.Name(),
			Description: t.Description(),
			InputSchema: schema,
		})
	}
	writeRPCResult(w, req.ID, map[string]any{
		"tools": tools,
	})
}

func normalizeInputSchema(in map[string]any) map[string]any {
	if in == nil {
		return map[string]any{"type": "object"}
	}
	if _, hasType := in["type"]; hasType {
		return in
	}
	cp := maps.Clone(in)
	cp["type"] = "object"
	return cp
}

func (s *Server) handleMCPToolsCall(ctx context.Context, w http.ResponseWriter, req rpcRequest, entry *runEntry) {
	var params struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	if len(req.Params) == 0 || json.Unmarshal(req.Params, &params) != nil || params.Name == "" {
		writeRPCError(w, req.ID, -32602, "invalid params: tool name is required")
		return
	}

	coreTool, ok := entry.tools[params.Name]
	if !ok {
		writeRPCError(w, req.ID, -32602, fmt.Sprintf("unknown tool %q", params.Name))
		return
	}

	argsJSON := bytes.TrimSpace(params.Arguments)
	if len(argsJSON) == 0 || bytes.Equal(argsJSON, []byte("null")) {
		argsJSON = json.RawMessage("{}")
	}

	text, isErr := executeMCPTool(ctx, coreTool, entry.hooks, entry.workingDir, argsJSON)
	writeRPCResult(w, req.ID, map[string]any{
		"content": []map[string]any{
			{
				"type": "text",
				"text": text,
			},
		},
		"isError": isErr,
	})
}

func executeMCPTool(ctx context.Context, coreTool actool.CoreTool, hooks harness.HookRunner, workingDir string, argsJSON json.RawMessage) (resultText string, isErr bool) {
	defer func() {
		if r := recover(); r != nil {
			resultText = fmt.Sprintf("tool panic: %v", r)
			isErr = true
		}
	}()

	toolName := coreTool.Name()
	if hooks != nil {
		block, reason, updated := hooks.PreToolUse(ctx, toolName, argsJSON)
		if block {
			if reason == "" {
				reason = "blocked by hook"
			}
			return reason, true
		}
		if len(updated) > 0 {
			argsJSON = updated
		}
	}

	var tc *actool.ToolContext
	if workingDir != "" {
		tc = &actool.ToolContext{WorkingDir: workingDir}
	}
	res, err := coreTool.Call(ctx, argsJSON, tc)
	if err != nil {
		if res.Flatten() == "" {
			res = actool.Errorf("Error: " + err.Error())
		} else {
			res.IsError = true
		}
	}

	resultText = res.Flatten()
	isErr = res.IsError

	if hooks != nil {
		hooks.PostToolUse(ctx, toolName, argsJSON, []byte(resultText), isErr)
	}
	return resultText, isErr
}

func writeRPCResult(w http.ResponseWriter, id json.RawMessage, result any) {
	writeJSON(w, http.StatusOK, rpcResponse{
		JSONRPC: "2.0",
		ID:      id,
		Result:  result,
	})
}

func writeRPCError(w http.ResponseWriter, id json.RawMessage, code int, message string) {
	if len(bytes.TrimSpace(id)) == 0 {
		id = json.RawMessage("null")
	}
	writeJSON(w, http.StatusOK, rpcResponse{
		JSONRPC: "2.0",
		ID:      id,
		Error: &rpcError{
			Code:    code,
			Message: message,
		},
	})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// --- CLI Hook Bridge (/hook/{token}/{event}) ---

const (
	eventPreToolUse  = "pretooluse"
	eventPostToolUse = "posttooluse"
	eventStop        = "stop"
)

func normalizeEvent(raw string) string {
	s := strings.ToLower(strings.TrimSpace(raw))
	s = strings.NewReplacer("-", "", "_", "").Replace(s)
	switch s {
	case eventPreToolUse, eventPostToolUse, eventStop:
		return s
	default:
		return ""
	}
}

func parseHookRoute(r *http.Request) (token string, event string) {
	rest := strings.Trim(strings.TrimPrefix(r.URL.Path, "/hook/"), "/")
	hdrToken := strings.TrimSpace(r.Header.Get("X-Xetra-Bridge-Token"))
	hdrEvent := strings.TrimSpace(r.Header.Get("X-Xetra-Hook-Event"))
	if hdrEvent == "" {
		hdrEvent = strings.TrimSpace(r.URL.Query().Get("event"))
	}

	if rest == "" {
		return hdrToken, normalizeEvent(hdrEvent)
	}
	parts := strings.Split(rest, "/")
	switch len(parts) {
	case 1:
		// Either /hook/{event} with X-Xetra-Bridge-Token, or /hook/{token} with X-Xetra-Hook-Event.
		if ev := normalizeEvent(parts[0]); ev != "" && hdrToken != "" {
			return hdrToken, ev
		}
		return parts[0], normalizeEvent(hdrEvent)
	case 2:
		tok := parts[0]
		if hdrToken != "" && hdrToken != tok {
			return "", ""
		}
		return tok, normalizeEvent(parts[1])
	default:
		return "", ""
	}
}

func (s *Server) handleHook(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", "POST")
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	token, event := parseHookRoute(r)
	entry, ok := s.lookupRun(token)
	if !ok {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return
	}
	if event == "" {
		http.Error(w, "unknown hook event", http.StatusNotFound)
		return
	}

	body, err := io.ReadAll(io.LimitReader(r.Body, maxBodyBytes))
	if err != nil {
		http.Error(w, "read body: "+err.Error(), http.StatusBadRequest)
		return
	}
	if len(bytes.TrimSpace(body)) == 0 {
		body = []byte("{}")
	}

	cli := strings.ToLower(strings.TrimSpace(r.URL.Query().Get("cli")))
	if cli == "" {
		cli = strings.ToLower(strings.TrimSpace(r.Header.Get("X-Xetra-Hook-CLI")))
	}
	if cli == "" {
		cli = detectCLI(body)
	}

	ctx, cancel := mergeContext(entry.ctx, r.Context())
	defer cancel()

	switch event {
	case eventPreToolUse:
		s.handleHookPreToolUse(ctx, w, entry, cli, body)
	case eventPostToolUse:
		s.handleHookPostToolUse(ctx, w, entry, cli, body)
	case eventStop:
		s.handleHookStop(ctx, w, entry, cli)
	}
}

func detectCLI(body []byte) string {
	var probe struct {
		ToolCall json.RawMessage `json:"toolCall"`
	}
	if json.Unmarshal(body, &probe) == nil && len(bytes.TrimSpace(probe.ToolCall)) > 0 && !bytes.Equal(bytes.TrimSpace(probe.ToolCall), []byte("null")) {
		return "agy"
	}
	return "claude"
}

func parseHookPayload(cli string, body []byte) (toolName string, rawInput []byte, resultBytes []byte, isErr bool, err error) {
	if cli == "agy" {
		var p struct {
			ToolCall struct {
				ID   string          `json:"id"`
				Name string          `json:"name"`
				Args json.RawMessage `json:"args"`
			} `json:"toolCall"`
			Output  json.RawMessage `json:"output"`
			Result  json.RawMessage `json:"result"`
			Error   string          `json:"error"`
			IsError bool            `json:"is_error"`
		}
		if err := json.Unmarshal(body, &p); err != nil {
			return "", nil, nil, false, err
		}
		rawInput = ensureJSONObject(p.ToolCall.Args)
		switch {
		case len(bytes.TrimSpace(p.Output)) > 0 && !bytes.Equal(bytes.TrimSpace(p.Output), []byte("null")):
			resultBytes = p.Output
		case len(bytes.TrimSpace(p.Result)) > 0 && !bytes.Equal(bytes.TrimSpace(p.Result), []byte("null")):
			resultBytes = p.Result
		case p.Error != "":
			resultBytes = []byte(p.Error)
		}
		isErr = p.IsError || p.Error != ""
		return p.ToolCall.Name, rawInput, resultBytes, isErr, nil
	}

	// Default: claude
	var p struct {
		ToolName     string          `json:"tool_name"`
		ToolInput    json.RawMessage `json:"tool_input"`
		ToolResponse json.RawMessage `json:"tool_response"`
		IsError      bool            `json:"is_error"`
	}
	if err := json.Unmarshal(body, &p); err != nil {
		return "", nil, nil, false, err
	}
	rawInput = ensureJSONObject(p.ToolInput)
	if len(bytes.TrimSpace(p.ToolResponse)) > 0 && !bytes.Equal(bytes.TrimSpace(p.ToolResponse), []byte("null")) {
		resultBytes = p.ToolResponse
	}
	isErr = p.IsError || responseHasError(p.ToolResponse)
	return p.ToolName, rawInput, resultBytes, isErr, nil
}

func ensureJSONObject(raw json.RawMessage) []byte {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return []byte("{}")
	}
	return trimmed
}

func responseHasError(raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || trimmed[0] != '{' {
		return false
	}
	var obj struct {
		IsError bool `json:"is_error"`
	}
	return json.Unmarshal(trimmed, &obj) == nil && obj.IsError
}

func (s *Server) handleHookPreToolUse(ctx context.Context, w http.ResponseWriter, entry *runEntry, cli string, body []byte) {
	toolName, rawInput, _, _, err := parseHookPayload(cli, body)
	if err != nil {
		http.Error(w, "invalid hook JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	normName, normInput := NormalizeHookTool(toolName, rawInput)
	var block bool
	var reason string
	if entry.hooks != nil {
		block, reason, _ = entry.hooks.PreToolUse(ctx, normName, normInput)
	}

	if cli == "agy" {
		if block {
			if reason == "" {
				reason = "blocked by hook"
			}
			writeJSON(w, http.StatusOK, map[string]any{
				"decision": "deny",
				"reason":   reason,
			})
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"decision": "allow",
		})
		return
	}

	// claude
	if block {
		if reason == "" {
			reason = "blocked by hook"
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"hookSpecificOutput": map[string]any{
				"hookEventName":            "PreToolUse",
				"permissionDecision":       "deny",
				"permissionDecisionReason": reason,
			},
		})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{})
}

func (s *Server) handleHookPostToolUse(ctx context.Context, w http.ResponseWriter, entry *runEntry, cli string, body []byte) {
	toolName, rawInput, resultBytes, isErr, err := parseHookPayload(cli, body)
	if err != nil {
		http.Error(w, "invalid hook JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	normName, normInput := NormalizeHookTool(toolName, rawInput)
	if entry.hooks != nil {
		entry.hooks.PostToolUse(ctx, normName, normInput, resultBytes, isErr)
	}
	writeJSON(w, http.StatusOK, map[string]any{})
}

func (s *Server) handleHookStop(ctx context.Context, w http.ResponseWriter, entry *runEntry, cli string) {
	if entry.hooks == nil {
		writeJSON(w, http.StatusOK, map[string]any{})
		return
	}
	prevent, blockingErrors, message := entry.hooks.Stop(ctx, nil)
	if !prevent && len(blockingErrors) == 0 {
		writeJSON(w, http.StatusOK, map[string]any{})
		return
	}
	reason := message
	if len(blockingErrors) > 0 {
		joined := strings.Join(blockingErrors, "; ")
		if reason == "" {
			reason = joined
		} else {
			reason = reason + ": " + joined
		}
	}
	if reason == "" {
		reason = "stopped by hook"
	}
	if cli == "agy" {
		writeJSON(w, http.StatusOK, map[string]any{
			"decision": "deny",
			"reason":   reason,
		})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"decision": "block",
		"reason":   reason,
	})
}

// NormalizeHookTool maps CLI-specific tool names and argument keys (such as
// Antigravity CLI's run_command, read_url_content, write_to_file,
// replace_file_content, and multi_replace_file_content) into ARTEX/norma
// canonical tool names and inputs before invoking HookRunner.
func NormalizeHookTool(toolName string, rawInput []byte) (string, []byte) {
	rawInput = ensureJSONObject(rawInput)
	switch toolName {
	case "run_command":
		return "Bash", remapJSONKey(rawInput, "CommandLine", "command")
	case "read_url_content":
		return "WebFetch", remapJSONKey(rawInput, "Url", "url")
	case "write_to_file":
		return "Write", remapJSONKey(rawInput, "TargetFile", "file_path")
	case "replace_file_content", "multi_replace_file_content":
		return "Edit", remapJSONKey(rawInput, "TargetFile", "file_path")
	default:
		return toolName, rawInput
	}
}

func remapJSONKey(raw []byte, fromKey, toKey string) []byte {
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil || m == nil {
		return raw
	}
	val, hasFrom := m[fromKey]
	if !hasFrom {
		return raw
	}
	if _, hasTo := m[toKey]; !hasTo {
		m[toKey] = val
	}
	delete(m, fromKey)
	out, err := json.Marshal(m)
	if err != nil {
		return raw
	}
	return out
}
