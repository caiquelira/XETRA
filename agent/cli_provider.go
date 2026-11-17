package agent

import (
	"context"
	"fmt"
	"io"
	"iter"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/Autumn-27/artex/llmrec"
	"github.com/Autumn-27/norma/harness"
	"github.com/Autumn-27/norma/llm"
)

// cliCompletionProvider adapts a CLIBackend (tools disabled, single-shot) to
// Norma's llm.Provider interface for stateless callers such as TestConnection,
// Compactor, intercept's fallback judge, and sidequestion.
type cliCompletionProvider struct {
	backend CLIBackend
	cfg     Config
}

func newCLICompletionProvider(backend CLIBackend, cfg Config) *cliCompletionProvider {
	return &cliCompletionProvider{backend: backend, cfg: cfg}
}

func (p *cliCompletionProvider) Stream(ctx context.Context, req llm.CompletionRequest) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		start := time.Now()
		var (
			totalUsage      llm.Usage
			status          = "error"
			consumerStopped bool
		)
		modelUsed := cliModelName(p.backend, p.cfg)
		defer func() {
			llmrec.RecordCLIUsage(ctx, nil, modelUsed, p.cfg.ProfileName, "", totalUsage, int(time.Since(start).Milliseconds()), status)
		}()

		sysPrompt, promptText := formatCLIPrompt(req)
		cfg := p.cfg
		if req.Thinking == "disabled" {
			cfg.ThinkingType = "disabled"
			cfg.ReasoningEffort = ""
		}

		inv, err := p.backend.BuildInvocation(ctx, CLIInvocationParams{
			Config:       cfg,
			Prompt:       promptText,
			SystemPrompt: sysPrompt,
			MaxTurns:     1,
			DisableTools: true,
		})
		if err != nil {
			yield(llm.StreamEvent{}, err)
			return
		}
		defer inv.Cleanup()

		var stdin io.Reader
		if inv.Stdin != "" {
			stdin = strings.NewReader(inv.Stdin)
		}
		h, err := StartProcessTree(ctx, ProcessSpec{
			Path:  inv.Binary,
			Args:  inv.Args,
			Dir:   inv.Dir,
			Env:   sanitizeCLIEnv(os.Environ(), inv.Env),
			Stdin: stdin,
		})
		if err != nil {
			yield(llm.StreamEvent{}, err)
			return
		}

		parser := p.backend.NewParser()
		var (
			finalResult *CLIResult
			sawText     bool
		)
		scanErr := scanProcessLines(h.Stdout(), func(line []byte) {
			events, parseErr := parser.ParseLine(line)
			if parseErr != nil {
				return
			}
			for _, ev := range events {
				switch ev.Kind {
				case "thinking":
					if ev.Text != "" && !consumerStopped {
						if !yield(llm.StreamEvent{Type: llm.SEThinkingDelta, Text: ev.Text}, nil) {
							consumerStopped = true
							_ = h.Kill()
						}
					}
				case "text":
					if ev.Text != "" && !consumerStopped {
						sawText = true
						if !yield(llm.StreamEvent{Type: llm.SETextDelta, Text: ev.Text}, nil) {
							consumerStopped = true
							_ = h.Kill()
						}
					}
				case "usage":
					if ev.Usage != nil && hasNonZeroUsage(*ev.Usage) {
						totalUsage = *ev.Usage
					}
				case "result":
					if ev.Result != nil {
						cp := *ev.Result
						finalResult = &cp
						if hasNonZeroUsage(cp.Usage) {
							totalUsage = cp.Usage
						}
					}
				}
			}
		})
		waitErr := h.Wait()

		if consumerStopped {
			return
		}
		if ctx.Err() != nil {
			yield(llm.StreamEvent{}, ctx.Err())
			return
		}
		if finalResult == nil {
			yield(llm.StreamEvent{}, cliResultError(p.backend, nil, h.Stderr(), firstNonNil(waitErr, scanErr)))
			return
		}
		if finalResult.IsError {
			yield(llm.StreamEvent{}, cliResultError(p.backend, finalResult, h.Stderr(), waitErr))
			return
		}
		if !sawText && finalResult.Text != "" {
			if !yield(llm.StreamEvent{Type: llm.SETextDelta, Text: finalResult.Text}, nil) {
				return
			}
		}

		stopReason := "end_turn"
		if finalResult.Reason == harness.ReasonMaxTurns {
			stopReason = "max_tokens"
		}
		if !yield(llm.StreamEvent{Type: llm.SEMessageDelta, StopReason: stopReason, Usage: totalUsage}, nil) {
			return
		}
		status = "ok"
		yield(llm.StreamEvent{Type: llm.SEMessageStop}, nil)
	}
}

func (p *cliCompletionProvider) Complete(ctx context.Context, req llm.CompletionRequest) (llm.Message, string, llm.Usage, error) {
	acc := llm.NewAccumulator()
	for ev, err := range p.Stream(ctx, req) {
		if err != nil {
			return llm.Message{}, "", acc.Usage, err
		}
		acc.Add(ev)
	}
	return acc.Message(), acc.StopReason, acc.Usage, nil
}

func formatCLIPrompt(req llm.CompletionRequest) (systemPrompt, promptText string) {
	var sysParts []string
	for _, seg := range req.System {
		if trimmed := strings.TrimSpace(seg); trimmed != "" {
			sysParts = append(sysParts, trimmed)
		}
	}
	systemPrompt = strings.Join(sysParts, "\n\n")

	if len(req.Messages) == 0 {
		return systemPrompt, ""
	}
	if len(req.Messages) == 1 && req.Messages[0].Role == llm.RoleUser && messageHasOnlyText(req.Messages[0]) {
		var parts []string
		for _, blk := range req.Messages[0].Content {
			if blk.Type == llm.BlockText && blk.Text != "" {
				parts = append(parts, blk.Text)
			}
		}
		return systemPrompt, strings.Join(parts, "\n")
	}

	var turns []string
	for _, msg := range req.Messages {
		body := formatCLIMessageBlocks(msg.Content)
		if body == "" {
			continue
		}
		label := formatCLIRoleLabel(msg.Role)
		turns = append(turns, label+": "+body)
	}
	return systemPrompt, strings.Join(turns, "\n\n")
}

func messageHasOnlyText(msg llm.Message) bool {
	for _, blk := range msg.Content {
		if blk.Type != llm.BlockText {
			return false
		}
	}
	return true
}

func formatCLIRoleLabel(role llm.Role) string {
	switch role {
	case llm.RoleUser:
		return "User"
	case llm.RoleAssistant:
		return "Assistant"
	default:
		r := strings.TrimSpace(string(role))
		if r == "" {
			return "User"
		}
		return strings.ToUpper(r[:1]) + r[1:]
	}
}

func formatCLIMessageBlocks(blocks []llm.ContentBlock) string {
	var parts []string
	for _, blk := range blocks {
		switch blk.Type {
		case llm.BlockText:
			if blk.Text != "" {
				parts = append(parts, blk.Text)
			}
		case llm.BlockToolUse:
			in := strings.TrimSpace(string(blk.Input))
			if in == "" {
				in = "{}"
			}
			parts = append(parts, fmt.Sprintf("[Tool call: %s(%s)]", blk.Name, in))
		case llm.BlockToolResult:
			out := blocksText(blk.Content)
			prefix := "[Tool result]: "
			if blk.IsError {
				prefix = "[Tool result (error)]: "
			}
			parts = append(parts, prefix+out)
		case llm.BlockThinking, llm.BlockRedactedThinking:
			// Omit internal reasoning from flattened single-shot replay prompts.
		}
	}
	return strings.Join(parts, "\n")
}

// sanitizeCLIEnv builds the child environment from base (or os.Environ() when
// nil) and extra override slices, stripping nested CLI session guards such as
// CLAUDECODE and ensuring NO_PROXY / no_proxy include loopback hosts so
// mcpbridge calls never traverse an HTTP proxy.
func sanitizeCLIEnv(base []string, extra ...[]string) []string {
	if base == nil {
		base = os.Environ()
	}

	caseInsensitive := runtime.GOOS == "windows"
	type envEntry struct {
		key string
		val string
	}
	entries := make([]envEntry, 0, len(base))
	indexByKey := make(map[string]int, len(base))
	var noProxyItems []string
	seenNoProxy := make(map[string]struct{})

	addNoProxy := func(raw string) {
		for _, item := range strings.Split(raw, ",") {
			trimmed := strings.TrimSpace(item)
			if trimmed == "" {
				continue
			}
			norm := strings.ToLower(trimmed)
			if _, ok := seenNoProxy[norm]; ok {
				continue
			}
			seenNoProxy[norm] = struct{}{}
			noProxyItems = append(noProxyItems, trimmed)
		}
	}

	applySlice := func(slice []string) {
		for _, kv := range slice {
			k, v, ok := strings.Cut(kv, "=")
			if !ok || k == "" {
				continue
			}
			upper := strings.ToUpper(k)
			if upper == "CLAUDECODE" || upper == "CLAUDE_CODE_ENTRYPOINT" {
				continue
			}
			if upper == "NO_PROXY" {
				addNoProxy(v)
				continue
			}
			lookup := k
			if caseInsensitive {
				lookup = upper
			}
			if idx, exists := indexByKey[lookup]; exists {
				entries[idx] = envEntry{key: k, val: v}
			} else {
				indexByKey[lookup] = len(entries)
				entries = append(entries, envEntry{key: k, val: v})
			}
		}
	}

	applySlice(base)
	for _, s := range extra {
		applySlice(s)
	}
	addNoProxy("127.0.0.1")
	addNoProxy("localhost")
	addNoProxy("::1")
	noProxyVal := strings.Join(noProxyItems, ",")

	out := make([]string, 0, len(entries)+2)
	for _, e := range entries {
		out = append(out, e.key+"="+e.val)
	}
	out = append(out, "NO_PROXY="+noProxyVal, "no_proxy="+noProxyVal)
	return out
}

func cliModelName(backend CLIBackend, cfg Config) string {
	if m := strings.TrimSpace(cfg.Model); m != "" {
		return m
	}
	if backend != nil {
		return backend.DefaultModel()
	}
	return ""
}

func hasNonZeroUsage(u llm.Usage) bool {
	return u.InputTokens != 0 || u.OutputTokens != 0 || u.CacheReadTokens != 0 || u.CacheWriteTokens != 0
}

func cliResultError(backend CLIBackend, res *CLIResult, stderr string, waitErr error) error {
	name := "CLI"
	if backend != nil && backend.Name() != "" {
		name = backend.Name()
	}
	stderr = strings.TrimSpace(stderr)
	if res != nil && res.IsError {
		msg := strings.TrimSpace(res.ErrorText)
		if msg == "" {
			msg = strings.TrimSpace(res.Text)
		}
		if msg == "" {
			msg = stderr
		}
		if msg == "" {
			msg = fmt.Sprintf("reason=%s", res.Reason)
		}
		if res.IsQuotaError || isCLIQuotaError(msg) {
			return fmt.Errorf("%s CLI quota exhausted: %s", name, msg)
		}
		return fmt.Errorf("%s CLI error: %s", name, msg)
	}
	if stderr != "" && isCLIQuotaError(stderr) {
		return fmt.Errorf("%s CLI quota exhausted: %s", name, stderr)
	}
	if waitErr != nil {
		if stderr != "" {
			return fmt.Errorf("%s CLI failed: %w: %s", name, waitErr, stderr)
		}
		return fmt.Errorf("%s CLI failed: %w", name, waitErr)
	}
	if stderr != "" {
		return fmt.Errorf("%s CLI stream ended without a result event: %s", name, stderr)
	}
	return fmt.Errorf("%s CLI stream ended without a result event", name)
}

func firstNonNil(errs ...error) error {
	for _, err := range errs {
		if err != nil {
			return err
		}
	}
	return nil
}
