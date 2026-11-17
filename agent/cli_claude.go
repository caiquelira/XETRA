package agent

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/Autumn-27/norma/harness"
	"github.com/Autumn-27/norma/llm"
)

const (
	claudeDefaultModel        = "sonnet"
	claudeInlineSystemLimit   = 16000
	claudeSystemOverflowFile  = ".xetra-system.md"
	cliScannerMaxTokenSize    = 16 * 1024 * 1024
)

func init() {
	RegisterCLIBackend(claudeCLIBackend{})
}

type claudeCLIBackend struct{}

func (claudeCLIBackend) Format() llm.Format    { return FormatClaudeCLI }
func (claudeCLIBackend) Name() string          { return "Claude Code" }
func (claudeCLIBackend) DefaultBinary() string { return "claude" }
func (claudeCLIBackend) DefaultModel() string  { return claudeDefaultModel }

func (b claudeCLIBackend) CheckReady(cfg Config) error {
	bin := resolveCLIBinary(cfg, "ARTEX_CLAUDE_BIN", b.DefaultBinary())
	return checkCLIBinaryReady(b.Name(), bin)
}

func (b claudeCLIBackend) BuildInvocation(_ context.Context, params CLIInvocationParams) (*CLIInvocation, error) {
	bin := resolveCLIBinary(params.Config, "ARTEX_CLAUDE_BIN", b.DefaultBinary())
	var tempPaths []string
	cleanupAll := func() {
		for _, p := range tempPaths {
			if p != "" {
				_ = os.Remove(p)
			}
		}
	}
	fail := func(err error) (*CLIInvocation, error) {
		cleanupAll()
		return nil, err
	}

	args := []string{
		"-p",
		"--output-format", "stream-json",
		"--verbose",
		"--permission-mode", "bypassPermissions",
	}

	model := strings.TrimSpace(params.Config.Model)
	if model == "" {
		model = b.DefaultModel()
	}
	args = append(args, "--model", model)

	if effort := strings.TrimSpace(params.Config.ReasoningEffort); effort != "" {
		args = append(args, "--effort", effort)
	}
	if params.MaxTurns > 0 {
		args = append(args, "--max-turns", strconv.Itoa(params.MaxTurns))
	}

	if params.DisableTools {
		args = append(args, "--tools", "", "--no-session-persistence")
	} else {
		switch {
		case params.IsResume && params.SessionID != "":
			args = append(args, "--resume", params.SessionID)
		case params.SessionID != "":
			args = append(args, "--session-id", params.SessionID)
		default:
			args = append(args, "--no-session-persistence")
		}
	}

	for _, dir := range params.ExtraDirs {
		if d := strings.TrimSpace(dir); d != "" {
			args = append(args, "--add-dir", d)
		}
	}

	if mcpURL := strings.TrimSpace(params.MCPURL); mcpURL != "" {
		mcpPayload := map[string]any{
			"mcpServers": map[string]any{
				"xetra": map[string]any{
					"type": "http",
					"url":  mcpURL,
				},
			},
		}
		mcpFile, err := writeTempJSON("xetra-claude-mcp-*.json", mcpPayload)
		if err != nil {
			return fail(fmt.Errorf("claude mcp config: %w", err))
		}
		tempPaths = append(tempPaths, mcpFile)
		args = append(args, "--mcp-config", mcpFile, "--strict-mcp-config")
	}

	var hookURL string
	if rawHookURL := strings.TrimSpace(params.HookURL); rawHookURL != "" && !params.DisableTools {
		hookURL = rawHookURL
		hookBin := resolveHookBin(params.HookBin)
		settingsPayload := map[string]any{
			"hooks": map[string]any{
				"PreToolUse": []map[string]any{
					{
						"matcher": "*",
						"hooks": []map[string]any{
							{
								"type":    "command",
								"command": buildHookCommand(hookBin, "claude", "pre-tool-use"),
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
								"command": buildHookCommand(hookBin, "claude", "post-tool-use"),
							},
						},
					},
				},
			},
		}
		settingsFile, err := writeTempJSON("xetra-claude-settings-*.json", settingsPayload)
		if err != nil {
			return fail(fmt.Errorf("claude hooks settings: %w", err))
		}
		tempPaths = append(tempPaths, settingsFile)
		args = append(args, "--settings", settingsFile)
	}

	if params.SystemPrompt != "" {
		if len(params.SystemPrompt) <= claudeInlineSystemLimit {
			args = append(args, "--system-prompt", params.SystemPrompt)
		} else {
			sysPath, err := writeClaudeSystemFile(params.WorkDir, params.SystemPrompt)
			if err != nil {
				return fail(fmt.Errorf("claude system prompt file: %w", err))
			}
			tempPaths = append(tempPaths, sysPath)
			args = append(args, "--system-prompt",
				fmt.Sprintf("Read and strictly follow all system instructions in %s before acting.", sysPath))
		}
	}

	return &CLIInvocation{
		Binary:  bin,
		Args:    args,
		Dir:     params.WorkDir,
		Stdin:   params.Prompt,
		Env:     hookEnvList(hookURL, params.HookToken, "claude", params.Env),
		Cleanup: makeCleanupFunc(cleanupAll),
	}, nil
}

func (claudeCLIBackend) NewParser() CLIStreamParser {
	return &claudeStreamParser{}
}

func (claudeCLIBackend) ParseLine(line []byte) ([]CLIEvent, error) {
	return (&claudeStreamParser{}).ParseLine(line)
}

func (b claudeCLIBackend) ParseFinal(stdout []byte) (*CLIResult, error) {
	parser := b.NewParser()
	scanner := bufio.NewScanner(bytes.NewReader(stdout))
	scanner.Buffer(make([]byte, 0, 64*1024), cliScannerMaxTokenSize)

	var (
		initSessionID string
		textParts     []string
		finalResult   *CLIResult
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
					initSessionID = ev.SessionID
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
		return nil, fmt.Errorf("scan claude output: %w", err)
	}
	if finalResult == nil {
		return nil, fmt.Errorf("claude stream ended without a result event")
	}
	if finalResult.SessionID == "" {
		finalResult.SessionID = initSessionID
	}
	if finalResult.Text == "" && !finalResult.IsError && len(textParts) > 0 {
		finalResult.Text = strings.Join(textParts, "")
	}
	return finalResult, nil
}

func (claudeCLIBackend) ListModels(_ context.Context, _ Config) ([]string, error) {
	return []string{"sonnet", "opus", "haiku"}, nil
}

type claudeStreamParser struct{}

type claudeRawLine struct {
	Type         string          `json:"type"`
	Subtype      string          `json:"subtype"`
	SessionID    string          `json:"session_id"`
	IsError      bool            `json:"is_error"`
	Result       string          `json:"result"`
	TotalCostUSD float64         `json:"total_cost_usd"`
	NumTurns     int             `json:"num_turns"`
	Errors       []string        `json:"errors"`
	Message      *claudeMessage  `json:"message"`
	Usage        *claudeRawUsage `json:"usage"`
}

type claudeMessage struct {
	Content []claudeContentBlock `json:"content"`
}

type claudeContentBlock struct {
	Type      string          `json:"type"`
	Thinking  string          `json:"thinking"`
	Text      string          `json:"text"`
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Input     json.RawMessage `json:"input"`
	ToolUseID string          `json:"tool_use_id"`
	Content   json.RawMessage `json:"content"`
	IsError   bool            `json:"is_error"`
}

type claudeRawUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
}

func (p *claudeStreamParser) ParseLine(line []byte) ([]CLIEvent, error) {
	trimmed := bytes.TrimSpace(line)
	if len(trimmed) == 0 {
		return nil, nil
	}

	var raw claudeRawLine
	if err := json.Unmarshal(trimmed, &raw); err != nil {
		return nil, fmt.Errorf("parse claude stream-json line: %w", err)
	}

	switch raw.Type {
	case "system":
		if raw.Subtype == "init" {
			return []CLIEvent{{
				Kind:      "init",
				SessionID: raw.SessionID,
			}}, nil
		}
		return nil, nil

	case "assistant":
		if raw.Message == nil {
			return nil, nil
		}
		var events []CLIEvent
		for _, blk := range raw.Message.Content {
			switch blk.Type {
			case "thinking":
				text := blk.Thinking
				if text == "" {
					text = blk.Text
				}
				events = append(events, CLIEvent{
					Kind: "thinking",
					Text: text,
				})
			case "text":
				events = append(events, CLIEvent{
					Kind: "text",
					Text: blk.Text,
				})
			case "tool_use":
				events = append(events, CLIEvent{
					Kind:      "tool_use",
					ToolID:    blk.ID,
					ToolName:  blk.Name,
					ToolInput: blk.Input,
				})
			}
		}
		return events, nil

	case "user":
		if raw.Message == nil {
			return nil, nil
		}
		var events []CLIEvent
		for _, blk := range raw.Message.Content {
			if blk.Type != "tool_result" {
				continue
			}
			events = append(events, CLIEvent{
				Kind:    "tool_result",
				ToolID:  blk.ToolUseID,
				Text:    extractClaudeToolResultText(blk.Content),
				IsError: blk.IsError,
			})
		}
		return events, nil

	case "result":
		var usage llm.Usage
		if raw.Usage != nil {
			usage = llm.Usage{
				InputTokens:      raw.Usage.InputTokens,
				OutputTokens:     raw.Usage.OutputTokens,
				CacheReadTokens:  raw.Usage.CacheReadInputTokens,
				CacheWriteTokens: raw.Usage.CacheCreationInputTokens,
			}
		}

		reason := mapClaudeTerminalReason(raw.Subtype, raw.IsError)
		isErr := raw.IsError || reason == harness.ReasonModelError
		var errText string
		if isErr || len(raw.Errors) > 0 {
			errText = combineClaudeErrors(raw.Errors, raw.Result)
		}
		var isQuota, isMissing bool
		if isErr || len(raw.Errors) > 0 {
			combined := strings.TrimSpace(errText + " " + raw.Result)
			isQuota = isCLIQuotaError(combined)
			isMissing = isCLISessionMissingError(combined)
		}

		res := &CLIResult{
			SessionID:        raw.SessionID,
			Text:             raw.Result,
			Reason:           reason,
			ErrorText:        errText,
			CostUSD:          raw.TotalCostUSD,
			NumTurns:         raw.NumTurns,
			Usage:            usage,
			IsError:          isErr,
			IsQuotaError:     isQuota,
			IsSessionMissing: isMissing,
		}
		return []CLIEvent{{
			Kind:      "result",
			SessionID: raw.SessionID,
			Text:      raw.Result,
			Usage:     &usage,
			Result:    res,
		}}, nil

	default:
		return nil, nil
	}
}

func mapClaudeTerminalReason(subtype string, isError bool) harness.TerminalReason {
	switch strings.TrimSpace(subtype) {
	case "success", "":
		if isError {
			return harness.ReasonModelError
		}
		return harness.ReasonCompleted
	case "error_max_turns", "error_max_budget_usd":
		return harness.ReasonMaxTurns
	case "error_during_execution":
		return harness.ReasonModelError
	default:
		if isError || strings.HasPrefix(subtype, "error") {
			return harness.ReasonModelError
		}
		return harness.ReasonCompleted
	}
}

func combineClaudeErrors(errs []string, result string) string {
	var parts []string
	for _, e := range errs {
		if t := strings.TrimSpace(e); t != "" {
			parts = append(parts, t)
		}
	}
	if len(parts) > 0 {
		joined := strings.Join(parts, "; ")
		if r := strings.TrimSpace(result); r != "" && !strings.Contains(joined, r) {
			return joined + ": " + r
		}
		return joined
	}
	return strings.TrimSpace(result)
}

func extractClaudeToolResultText(raw json.RawMessage) string {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return ""
	}
	var s string
	if err := json.Unmarshal(trimmed, &s); err == nil {
		return s
	}
	var blocks []struct {
		Type    string `json:"type"`
		Text    string `json:"text"`
		Content string `json:"content"`
	}
	if err := json.Unmarshal(trimmed, &blocks); err == nil {
		var parts []string
		for _, blk := range blocks {
			if blk.Text != "" {
				parts = append(parts, blk.Text)
			} else if blk.Content != "" {
				parts = append(parts, blk.Content)
			}
		}
		return strings.Join(parts, "\n")
	}
	return string(trimmed)
}

func writeTempJSON(pattern string, payload any) (string, error) {
	data, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return "", err
	}
	data = append(data, '\n')
	f, err := os.CreateTemp("", pattern)
	if err != nil {
		return "", err
	}
	path := f.Name()
	if _, err := f.Write(data); err != nil {
		_ = f.Close()
		_ = os.Remove(path)
		return "", err
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(path)
		return "", err
	}
	return path, nil
}

func writeClaudeSystemFile(workDir, systemPrompt string) (string, error) {
	if dir := strings.TrimSpace(workDir); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return "", err
		}
		path := filepath.Join(dir, claudeSystemOverflowFile)
		if err := os.WriteFile(path, []byte(systemPrompt), 0o600); err != nil {
			return "", err
		}
		return path, nil
	}
	f, err := os.CreateTemp("", "xetra-claude-system-*.md")
	if err != nil {
		return "", err
	}
	path := f.Name()
	if _, err := f.WriteString(systemPrompt); err != nil {
		_ = f.Close()
		_ = os.Remove(path)
		return "", err
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(path)
		return "", err
	}
	return path, nil
}
