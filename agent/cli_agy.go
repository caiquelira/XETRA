package agent

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"unicode/utf8"

	"github.com/Autumn-27/norma/harness"
	"github.com/Autumn-27/norma/llm"
)

const (
	agyDefaultModel       = "gemini-4-hydrogen-high"
	agyInlinePromptLimit  = 4000
	agyRuleChunkLimit     = 20000
	agyPromptOverflowFile = ".xetra-prompt.md"
	agySystemOverflowFile = ".xetra-system.md"
)

func init() {
	RegisterCLIBackend(agyCLIBackend{})
}

type agyCLIBackend struct{}

func (agyCLIBackend) Format() llm.Format    { return FormatAgyCLI }
func (agyCLIBackend) Name() string          { return "Antigravity CLI" }
func (agyCLIBackend) DefaultBinary() string { return "agy" }
func (agyCLIBackend) DefaultModel() string  { return agyDefaultModel }

func (b agyCLIBackend) CheckReady(cfg Config) error {
	bin := resolveCLIBinary(cfg, "ARTEX_AGY_BIN", b.DefaultBinary())
	return checkCLIBinaryReady(b.Name(), bin)
}

func (b agyCLIBackend) BuildInvocation(_ context.Context, params CLIInvocationParams) (*CLIInvocation, error) {
	bin := resolveCLIBinary(params.Config, "ARTEX_AGY_BIN", b.DefaultBinary())

	workDir := strings.TrimSpace(params.WorkDir)
	ephemeralWorkDir := false
	needsWorkDirFiles := len(params.Prompt) > agyInlinePromptLimit ||
		strings.ContainsAny(params.Prompt, "\r\n") ||
		params.MCPURL != "" ||
		(params.HookURL != "" && !params.DisableTools) ||
		params.SystemPrompt != ""

	if workDir == "" && needsWorkDirFiles {
		dir, err := os.MkdirTemp("", "xetra-agy-*")
		if err != nil {
			return nil, fmt.Errorf("agy create ephemeral workdir: %w", err)
		}
		workDir = dir
		ephemeralWorkDir = true
	} else if workDir != "" && needsWorkDirFiles {
		if err := os.MkdirAll(workDir, 0o755); err != nil {
			return nil, fmt.Errorf("agy create workdir %q: %w", workDir, err)
		}
	}

	cleanup := makeCleanupFunc(func() {
		if workDir == "" {
			return
		}
		_ = os.Remove(filepath.Join(workDir, agyPromptOverflowFile))
		_ = os.Remove(filepath.Join(workDir, agySystemOverflowFile))
		_ = os.RemoveAll(filepath.Join(workDir, ".agents", "plugins", "xetra"))
		// os.Remove only removes empty directories, preserving any user/repo files.
		_ = os.Remove(filepath.Join(workDir, ".agents", "plugins"))
		_ = os.Remove(filepath.Join(workDir, ".agents"))
		if ephemeralWorkDir {
			_ = os.RemoveAll(workDir)
		}
	})
	fail := func(err error) (*CLIInvocation, error) {
		cleanup()
		return nil, err
	}

	args := []string{
		"--output-format", "stream-json",
		"--dangerously-skip-permissions",
	}

	model := strings.TrimSpace(params.Config.Model)
	if model == "" {
		model = b.DefaultModel()
	}
	args = append(args, "--model", model)

	if effort := strings.TrimSpace(params.Config.ReasoningEffort); effort != "" {
		args = append(args, "--effort", effort)
	}
	for _, dir := range params.ExtraDirs {
		if d := strings.TrimSpace(dir); d != "" {
			args = append(args, "--add-dir", d)
		}
	}
	if params.IsResume && params.SessionID != "" {
		args = append(args, "--conversation", params.SessionID)
	}

	// agy requires `-p=<non-empty>` because bare `-p` consumes the next flag.
	prompt := params.Prompt
	switch {
	case prompt == "":
		args = append(args, "-p=Proceed.")
	case len(prompt) <= agyInlinePromptLimit && !strings.ContainsAny(prompt, "\r\n"):
		args = append(args, "-p="+prompt)
	default:
		promptPath := filepath.Join(workDir, agyPromptOverflowFile)
		if err := os.WriteFile(promptPath, []byte(prompt), 0o600); err != nil {
			return fail(fmt.Errorf("agy write prompt file: %w", err))
		}
		args = append(args, "-p=Read and execute the complete instructions in .xetra-prompt.md in your working directory.")
	}

	var hookURL string
	mcpURL := strings.TrimSpace(params.MCPURL)
	mcpActive := mcpURL != "" && !params.DisableTools
	if !params.DisableTools && strings.TrimSpace(params.HookURL) != "" {
		hookURL = strings.TrimSpace(params.HookURL)
	}

	hasPlugin := mcpActive || hookURL != "" || params.SystemPrompt != ""
	if hasPlugin {
		pluginDir := filepath.Join(workDir, ".agents", "plugins", "xetra")
		if err := os.MkdirAll(pluginDir, 0o755); err != nil {
			return fail(fmt.Errorf("agy create plugin dir: %w", err))
		}
		if err := writeJSONFile(filepath.Join(pluginDir, "plugin.json"), map[string]any{
			"name": "xetra",
		}); err != nil {
			return fail(fmt.Errorf("agy write plugin.json: %w", err))
		}

		if mcpActive {
			mcpPayload := map[string]any{
				"mcpServers": map[string]any{
					"bridge": map[string]any{
						"url": mcpURL,
					},
				},
			}
			if err := writeJSONFile(filepath.Join(pluginDir, "mcp_config.json"), mcpPayload); err != nil {
				return fail(fmt.Errorf("agy write mcp_config.json: %w", err))
			}
		}

		if hookURL != "" {
			hookBin := resolveHookBin(params.HookBin)
			hooksPayload := map[string]any{
				"hooks": map[string]any{
					"PreToolUse": []map[string]any{
						{
							"matcher": "*",
							"hooks": []map[string]any{
								{
									"type":    "command",
									"command": buildHookCommand(hookBin, "agy", "pre-tool-use"),
								},
							},
						},
					},
					"PostToolUse": []map[string]any{
						{
							"matcher": "*",
							"hooks": []map[string]any{
								{
									"type":    "command",
									"command": buildHookCommand(hookBin, "agy", "post-tool-use"),
								},
							},
						},
					},
				},
			}
			if err := writeJSONFile(filepath.Join(pluginDir, "hooks.json"), hooksPayload); err != nil {
				return fail(fmt.Errorf("agy write hooks.json: %w", err))
			}
		}

		if err := writeAgyRules(pluginDir, params.SystemPrompt, mcpURL != "" && !params.DisableTools); err != nil {
			return fail(fmt.Errorf("agy write rules: %w", err))
		}
	}

	return &CLIInvocation{
		Binary:  bin,
		Args:    args,
		Dir:     workDir,
		Env:     hookEnvList(hookURL, params.HookToken, "agy", params.Env),
		Cleanup: cleanup,
	}, nil
}

func writeAgyRules(pluginDir, systemPrompt string, mcpActive bool) error {
	sysText := systemPrompt
	mcpNote := "XETRA domain tools (record_fact, add_intent, report_finding, etc.) are exposed via `call_mcp_tool` under server `xetra_bridge`."
	if mcpActive {
		if strings.TrimSpace(sysText) == "" {
			sysText = mcpNote
		} else {
			sysText = strings.TrimRight(sysText, "\r\n") + "\n\n## XETRA MCP Tools\n" + mcpNote + "\n"
		}
	}
	if strings.TrimSpace(sysText) == "" {
		return nil
	}

	rulesDir := filepath.Join(pluginDir, "rules")
	if err := os.MkdirAll(rulesDir, 0o755); err != nil {
		return err
	}

	if len(sysText) <= agyRuleChunkLimit {
		return os.WriteFile(filepath.Join(rulesDir, "AGENTS.md"), []byte(sysText), 0o600)
	}

	chunks := chunkMarkdown(sysText, agyRuleChunkLimit)
	var index strings.Builder
	index.WriteString("# System Instructions\n\nRead and strictly follow all instructions across the numbered rule files below:\n")
	for i, chunk := range chunks {
		name := fmt.Sprintf("%02d.md", i+1)
		if err := os.WriteFile(filepath.Join(rulesDir, name), []byte(chunk), 0o600); err != nil {
			return err
		}
		fmt.Fprintf(&index, "- `rules/%s`\n", name)
	}
	if mcpActive {
		index.WriteString("\n## XETRA MCP Tools\n")
		index.WriteString(mcpNote + "\n")
	}
	return os.WriteFile(filepath.Join(rulesDir, "AGENTS.md"), []byte(index.String()), 0o600)
}

func chunkMarkdown(text string, maxBytes int) []string {
	if len(text) <= maxBytes {
		return []string{text}
	}
	var chunks []string
	for len(text) > 0 {
		if len(text) <= maxBytes {
			chunks = append(chunks, text)
			break
		}
		cut := maxBytes
		// Prefer splitting at a newline boundary in the upper half of the window.
		if nl := strings.LastIndexByte(text[:cut], '\n'); nl > maxBytes/2 {
			cut = nl + 1
		} else {
			// Avoid cutting inside a multi-byte UTF-8 rune.
			for cut > 0 && !utf8.RuneStart(text[cut]) {
				cut--
			}
			if cut == 0 {
				cut = maxBytes
			}
		}
		chunks = append(chunks, text[:cut])
		text = text[cut:]
	}
	return chunks
}

func writeJSONFile(path string, payload any) error {
	data, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o600)
}

func (agyCLIBackend) NewParser() CLIStreamParser {
	return &agyStreamParser{
		seenToolActive: make(map[int]bool),
		seenToolDone:   make(map[int]bool),
	}
}

func (b agyCLIBackend) ParseLine(line []byte) ([]CLIEvent, error) {
	return b.NewParser().ParseLine(line)
}

func (b agyCLIBackend) ParseFinal(stdout []byte) (*CLIResult, error) {
	parser := b.NewParser()
	scanner := bufio.NewScanner(bytes.NewReader(stdout))
	scanner.Buffer(make([]byte, 0, 64*1024), cliScannerMaxTokenSize)

	var (
		initConvID  string
		textParts   []string
		finalResult *CLIResult
	)
	for scanner.Scan() {
		raw := bytes.TrimSpace(scanner.Bytes())
		if len(raw) == 0 {
			continue
		}
		events, err := parser.ParseLine(raw)
		if err != nil {
			continue
		}
		for _, ev := range events {
			switch ev.Kind {
			case "init":
				if ev.SessionID != "" {
					initConvID = ev.SessionID
				}
			case "text":
				if ev.Text != "" {
					textParts = append(textParts, ev.Text)
				}
			case "result":
				if ev.Result != nil {
					cp := *ev.Result
					finalResult = &cp
				}
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan agy output: %w", err)
	}
	if finalResult == nil {
		return nil, fmt.Errorf("agy stream ended without a result event")
	}
	if finalResult.SessionID == "" {
		finalResult.SessionID = initConvID
	}
	if finalResult.Text == "" && !finalResult.IsError && len(textParts) > 0 {
		finalResult.Text = strings.Join(textParts, "")
	}
	return finalResult, nil
}

func (b agyCLIBackend) ListModels(ctx context.Context, cfg Config) ([]string, error) {
	bin := resolveCLIBinary(cfg, "ARTEX_AGY_BIN", b.DefaultBinary())
	cmd := exec.CommandContext(ctx, bin, "models")
	out, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok && len(exitErr.Stderr) > 0 {
			return nil, fmt.Errorf("agy models failed: %w: %s", err, strings.TrimSpace(string(exitErr.Stderr)))
		}
		return nil, fmt.Errorf("agy models failed: %w", err)
	}
	return parseAgyModelsOutput(out), nil
}

func parseAgyModelsOutput(out []byte) []string {
	seen := make(map[string]struct{})
	var models []string
	for _, rawLine := range strings.Split(string(out), "\n") {
		line := strings.TrimRight(rawLine, "\r")
		id, _, hasTab := strings.Cut(line, "\t")
		if !hasTab {
			continue
		}
		id = strings.TrimSpace(id)
		if id == "" {
			continue
		}
		if _, dup := seen[id]; dup {
			continue
		}
		seen[id] = struct{}{}
		models = append(models, id)
	}
	return models
}

type agyStreamParser struct {
	seenToolActive map[int]bool
	seenToolDone   map[int]bool
}

type agyRawEvent struct {
	Event          string         `json:"event"`
	ConversationID string         `json:"conversation_id"`
	Init           *agyInitInfo   `json:"init"`
	StepUpdate     *agyStepUpdate `json:"step_update"`
	Result         *agyRawResult  `json:"result"`
}

type agyInitInfo struct {
	ConversationID string `json:"conversation_id"`
}

type agyStepUpdate struct {
	ConversationID string       `json:"conversation_id"`
	StepIndex      int          `json:"step_index"`
	State          string       `json:"state"`
	StepType       string       `json:"step_type"`
	TextDelta      string       `json:"text_delta"`
	ToolName       string       `json:"tool_name"`
	ToolInfo       *agyToolInfo `json:"tool_info"`
	Error          string       `json:"error"`
	Usage          *agyRawUsage `json:"usage"`
}

type agyToolInfo struct {
	ID         string          `json:"id"`
	Name       string          `json:"name"`
	Parameters json.RawMessage `json:"parameters"`
	Output     json.RawMessage `json:"output"`
	Error      string          `json:"error"`
}

type agyRawUsage struct {
	InputTokens     int `json:"input_tokens"`
	OutputTokens    int `json:"output_tokens"`
	ThinkingTokens  int `json:"thinking_tokens"`
	CacheReadTokens int `json:"cache_read_tokens"`
	TotalTokens     int `json:"total_tokens"`
}

type agyRawResult struct {
	ConversationID string       `json:"conversation_id"`
	Status         string       `json:"status"`
	Response       string       `json:"response"`
	Error          string       `json:"error"`
	NumTurns       int          `json:"num_turns"`
	Usage          *agyRawUsage `json:"usage"`
	FinishedReason string       `json:"finished_reason"`
}

func (p *agyStreamParser) ParseLine(line []byte) ([]CLIEvent, error) {
	trimmed := bytes.TrimSpace(line)
	if len(trimmed) == 0 {
		return nil, nil
	}
	var raw agyRawEvent
	if err := json.Unmarshal(trimmed, &raw); err != nil {
		return nil, fmt.Errorf("parse agy stream-json line: %w", err)
	}

	switch raw.Event {
	case "init":
		convID := raw.ConversationID
		if convID == "" && raw.Init != nil {
			convID = raw.Init.ConversationID
		}
		return []CLIEvent{{
			Kind:      "init",
			SessionID: convID,
		}}, nil

	case "step_update":
		if raw.StepUpdate == nil {
			return nil, nil
		}
		return p.parseStepUpdate(raw.StepUpdate), nil

	case "result":
		if raw.Result == nil {
			return nil, nil
		}
		return p.parseResult(raw.Result, raw.ConversationID), nil

	default:
		return nil, nil
	}
}

func (p *agyStreamParser) parseStepUpdate(su *agyStepUpdate) []CLIEvent {
	var events []CLIEvent
	stepType := strings.ToLower(strings.TrimSpace(su.StepType))
	state := strings.ToUpper(strings.TrimSpace(su.State))

	switch stepType {
	case "thought", "thinking":
		if su.TextDelta != "" {
			events = append(events, CLIEvent{
				Kind: "thinking",
				Text: su.TextDelta,
			})
		}
	case "agent_response", "assistant", "text":
		if su.TextDelta != "" {
			events = append(events, CLIEvent{
				Kind: "text",
				Text: su.TextDelta,
			})
		}
	case "user_input":
		// Echo of user prompt; do not emit as assistant text.
	default:
		isTool := stepType == "tool" || su.ToolName != "" || (su.ToolInfo != nil && su.ToolInfo.Name != "")
		if isTool {
			toolID, toolName, toolInput := resolveAgyToolCall(su)
			emitUse := func() {
				if !p.seenToolActive[su.StepIndex] {
					p.seenToolActive[su.StepIndex] = true
					events = append(events, CLIEvent{
						Kind:      "tool_use",
						ToolID:    toolID,
						ToolName:  toolName,
						ToolInput: toolInput,
					})
				}
			}
			emitResult := func() {
				if p.seenToolDone[su.StepIndex] {
					return
				}
				p.seenToolDone[su.StepIndex] = true
				errStr := strings.TrimSpace(su.Error)
				var outStr string
				if su.ToolInfo != nil {
					if su.ToolInfo.Error != "" {
						errStr = su.ToolInfo.Error
					}
					outStr = extractAgyToolOutput(su.ToolInfo.Output)
				}
				isErr := errStr != ""
				text := outStr
				if isErr {
					text = errStr
				}
				events = append(events, CLIEvent{
					Kind:     "tool_result",
					ToolID:   toolID,
					ToolName: toolName,
					Text:     text,
					IsError:  isErr,
				})
			}

			switch state {
			case "ACTIVE":
				emitUse()
			case "DONE":
				emitUse()
				emitResult()
			}
		}
	}

	if su.Usage != nil && (state == "DONE" || state == "") {
		if u := agyUsageToLLM(su.Usage); u.InputTokens+u.OutputTokens+u.CacheReadTokens > 0 {
			events = append(events, CLIEvent{
				Kind:  "usage",
				Usage: &u,
			})
		}
	}

	return events
}

func (p *agyStreamParser) parseResult(r *agyRawResult, fallbackConvID string) []CLIEvent {
	convID := r.ConversationID
	if convID == "" {
		convID = fallbackConvID
	}
	usage := agyUsageToLLM(r.Usage)
	reason, isErr := mapAgyTerminalReason(r.Status, r.FinishedReason, r.Error)

	errText := strings.TrimSpace(r.Error)
	if isErr && errText == "" && strings.TrimSpace(r.Status) != "" {
		errText = fmt.Sprintf("agy finished with status %s", strings.TrimSpace(r.Status))
		if fin := strings.TrimSpace(r.FinishedReason); fin != "" {
			errText += " (" + fin + ")"
		}
	}

	var isQuota, isMissing bool
	if isErr {
		isQuota = isCLIQuotaError(errText)
		isMissing = isCLISessionMissingError(errText)
	}

	res := &CLIResult{
		SessionID:        convID,
		Text:             r.Response,
		Reason:           reason,
		ErrorText:        errText,
		NumTurns:         r.NumTurns,
		Usage:            usage,
		IsError:          isErr,
		IsQuotaError:     isQuota,
		IsSessionMissing: isMissing,
	}
	return []CLIEvent{{
		Kind:      "result",
		SessionID: convID,
		Text:      r.Response,
		Usage:     &usage,
		Result:    res,
	}}
}

func resolveAgyToolCall(su *agyStepUpdate) (toolID, toolName string, toolInput json.RawMessage) {
	toolName = su.ToolName
	if toolName == "" && su.ToolInfo != nil {
		toolName = su.ToolInfo.Name
	}
	if su.ToolInfo != nil && su.ToolInfo.ID != "" {
		toolID = su.ToolInfo.ID
	} else {
		toolID = fmt.Sprintf("step-%d", su.StepIndex)
	}
	if su.ToolInfo != nil && len(bytes.TrimSpace(su.ToolInfo.Parameters)) > 0 {
		toolInput = bytes.TrimSpace(su.ToolInfo.Parameters)
	}

	if toolName == "call_mcp_tool" && len(toolInput) > 0 {
		var params map[string]json.RawMessage
		if err := json.Unmarshal(toolInput, &params); err == nil {
			if unwrappedName := firstJSONString(params, "ToolName", "tool_name", "name"); unwrappedName != "" {
				toolName = unwrappedName
			}
			if rawArgs := firstJSONValue(params, "Arguments", "arguments"); len(rawArgs) > 0 {
				toolInput = unwrapJSONArgument(rawArgs)
			}
		}
	}
	return toolID, toolName, toolInput
}

func firstJSONString(m map[string]json.RawMessage, keys ...string) string {
	for _, k := range keys {
		raw, ok := m[k]
		if !ok {
			continue
		}
		var s string
		if err := json.Unmarshal(raw, &s); err == nil && strings.TrimSpace(s) != "" {
			return strings.TrimSpace(s)
		}
	}
	return ""
}

func firstJSONValue(m map[string]json.RawMessage, keys ...string) json.RawMessage {
	for _, k := range keys {
		if raw, ok := m[k]; ok {
			trimmed := bytes.TrimSpace(raw)
			if len(trimmed) > 0 && !bytes.Equal(trimmed, []byte("null")) {
				return trimmed
			}
		}
	}
	return nil
}

func unwrapJSONArgument(raw json.RawMessage) json.RawMessage {
	var asStr string
	if err := json.Unmarshal(raw, &asStr); err == nil {
		trimmed := strings.TrimSpace(asStr)
		if json.Valid([]byte(trimmed)) {
			return json.RawMessage(trimmed)
		}
	}
	return raw
}

func extractAgyToolOutput(raw json.RawMessage) string {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return ""
	}
	var s string
	if err := json.Unmarshal(trimmed, &s); err == nil {
		return s
	}
	return string(trimmed)
}

func agyUsageToLLM(u *agyRawUsage) llm.Usage {
	if u == nil {
		return llm.Usage{}
	}
	return llm.Usage{
		InputTokens:     u.InputTokens,
		OutputTokens:    u.OutputTokens + u.ThinkingTokens,
		CacheReadTokens: u.CacheReadTokens,
	}
}

func mapAgyTerminalReason(status, finishedReason, errText string) (harness.TerminalReason, bool) {
	st := strings.ToUpper(strings.TrimSpace(status))
	fin := strings.ToLower(strings.TrimSpace(finishedReason))
	errLower := strings.ToLower(strings.TrimSpace(errText))

	if st == "SUCCESS" && errLower == "" {
		return harness.ReasonCompleted, false
	}
	if st == "TIMEOUT" || strings.Contains(fin, "timeout") || strings.Contains(fin, "deadline") {
		return harness.ReasonTimeout, true
	}
	if st == "MAX_TURNS" || strings.Contains(fin, "max_turns") || strings.Contains(fin, "max_steps") {
		return harness.ReasonMaxTurns, true
	}
	if st == "" && errLower == "" {
		return harness.ReasonCompleted, false
	}
	return harness.ReasonModelError, true
}
