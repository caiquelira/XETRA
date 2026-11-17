package agent

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/Autumn-27/artex/db"
	"github.com/Autumn-27/artex/intercept"
	"github.com/Autumn-27/artex/llmrec"
	"github.com/Autumn-27/artex/mcpbridge"
	"github.com/Autumn-27/norma/harness"
	"github.com/Autumn-27/norma/llm"
	actool "github.com/Autumn-27/norma/tool"
	"github.com/google/uuid"
)

// CLIRunOptions configures a full multi-turn headless CLI agent execution
// (Claude Code `claude` or Antigravity `agy`) wired to XETRA's MCP tools,
// safety hooks, session persistence, activity stream, and usage metering.
type CLIRunOptions struct {
	Config       Config
	SystemPrompt string
	Prompt       string
	WorkDir      string
	ExtraDirs    []string
	Tools        []actool.CoreTool
	Hooks        harness.HookRunner
	Bridge       *mcpbridge.Server
	HookBin      string
	Env          []string
	SessionKey   string
	SessionDir   string
	MaxTurns     int
	WorkerLabel  string
	UsagePG      *db.DB
	Emit         func(db.Activity)
}

// runCLI executes a multi-turn CLI run over StartProcessTree, bridging tools and
// hooks via mcpbridge, persisting native session IDs with stale-session retry,
// emitting coalesced db.Activity events, and recording usage via llmrec.
func runCLI(ctx context.Context, opts CLIRunOptions) (string, harness.TerminalReason, error) {
	if ctx == nil {
		ctx = context.Background()
	}

	backend, ok := LookupCLIBackend(opts.Config.Format)
	if !ok {
		return "", harness.ReasonModelError, fmt.Errorf("unknown CLI backend %q", opts.Config.Format)
	}
	if err := backend.CheckReady(opts.Config); err != nil {
		return "", harness.ReasonModelError, err
	}
	if err := ctx.Err(); err != nil {
		return "", "", err
	}

	workDir := strings.TrimSpace(opts.WorkDir)
	if workDir != "" {
		if err := os.MkdirAll(workDir, 0o755); err != nil {
			return "", harness.ReasonModelError, fmt.Errorf("create cli workdir %q: %w", workDir, err)
		}
	}

	ctx, auditTrace := intercept.WithTrace(ctx, opts.Prompt, nil)
	defer auditTrace.Finish()

	var (
		mcpURL    string
		hookURL   string
		hookToken string
	)
	if len(opts.Tools) > 0 || opts.Hooks != nil {
		bridge := opts.Bridge
		if bridge == nil {
			srv, err := mcpbridge.NewServer()
			if err != nil {
				return "", harness.ReasonModelError, fmt.Errorf("start mcp bridge: %w", err)
			}
			defer srv.Close()
			bridge = srv
		}
		token := bridge.RegisterRun(mcpbridge.RunConfig{
			Context:    ctx,
			Tools:      opts.Tools,
			Hooks:      opts.Hooks,
			WorkingDir: workDir,
		})
		defer bridge.UnregisterRun(token)
		if len(opts.Tools) > 0 {
			mcpURL = bridge.MCPURL(token)
		}
		if opts.Hooks != nil {
			hookURL = strings.TrimRight(bridge.HookURL(token, ""), "/")
			hookToken = token
		}
	}

	var (
		store      *CLISessionStore
		storedRec  CLISessionRecord
		hasStored  bool
		sessionKey = strings.TrimSpace(opts.SessionKey)
	)
	if sessionKey != "" {
		sessionDir := strings.TrimSpace(opts.SessionDir)
		if sessionDir == "" && workDir != "" {
			sessionDir = filepath.Join(workDir, "cli-sessions")
		}
		if sessionDir == "" {
			return "", harness.ReasonModelError, fmt.Errorf("cli session key %q requires SessionDir or WorkDir", sessionKey)
		}
		store = NewCLISessionStore(sessionDir)
		var err error
		storedRec, hasStored, err = store.Load(sessionKey)
		if err != nil {
			return "", "", err
		}
	}

	rec := func(r db.Activity) {
		if r.Kind == "text" || r.Kind == "tool_result" {
			auditTrace.Append(db.InterceptContextEntry{
				Kind:      r.Kind,
				Tool:      r.Tool,
				ToolUseID: r.ToolUseID,
				Text:      r.Detail,
				IsError:   r.IsError,
			})
		}
		if opts.Emit != nil {
			opts.Emit(r)
		}
	}

	modelUsed := cliModelName(backend, opts.Config)

	for attempt := 0; attempt < 2; attempt++ {
		isResume := attempt == 0 && hasStored
		sessionID := ""
		if isResume {
			sessionID = storedRec.SessionID
		} else if opts.Config.Format == FormatClaudeCLI && sessionKey != "" {
			sessionID = uuid.NewString()
		}

		res, retryFresh, err := runCLIAttempt(ctx, cliAttemptParams{
			opts:       opts,
			backend:    backend,
			workDir:    workDir,
			mcpURL:     mcpURL,
			hookURL:    hookURL,
			hookToken:  hookToken,
			store:      store,
			sessionKey: sessionKey,
			sessionID:  sessionID,
			isResume:   isResume,
			canRetry:   isResume && attempt == 0 && store != nil,
			modelUsed:  modelUsed,
			auditTrace: auditTrace,
			rec:        rec,
		})
		if retryFresh {
			hasStored = false
			continue
		}
		return res.text, res.reason, err
	}

	return "", harness.ReasonModelError, fmt.Errorf("%s CLI failed after session retry", backend.Name())
}

type cliAttemptParams struct {
	opts       CLIRunOptions
	backend    CLIBackend
	workDir    string
	mcpURL     string
	hookURL    string
	hookToken  string
	store      *CLISessionStore
	sessionKey string
	sessionID  string
	isResume   bool
	canRetry   bool
	modelUsed  string
	auditTrace *intercept.Trace
	rec        func(db.Activity)
}

type cliAttemptResult struct {
	text   string
	reason harness.TerminalReason
}

func runCLIAttempt(ctx context.Context, p cliAttemptParams) (cliAttemptResult, bool, error) {
	start := time.Now()
	inv, err := p.backend.BuildInvocation(ctx, CLIInvocationParams{
		Config:       p.opts.Config,
		Prompt:       p.opts.Prompt,
		SystemPrompt: p.opts.SystemPrompt,
		WorkDir:      p.workDir,
		ExtraDirs:    p.opts.ExtraDirs,
		MCPURL:       p.mcpURL,
		HookURL:      p.hookURL,
		HookBin:      p.opts.HookBin,
		HookToken:    p.hookToken,
		SessionKey:   p.sessionKey,
		SessionID:    p.sessionID,
		IsResume:     p.isResume,
		MaxTurns:     p.opts.MaxTurns,
	})
	if err != nil {
		return cliAttemptResult{reason: harness.ReasonModelError}, false, err
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
		Env:   sanitizeCLIEnv(os.Environ(), p.opts.Env, inv.Env),
		Stdin: stdin,
	})
	if err != nil {
		if ctx.Err() != nil {
			lastTool := &runTrace{startedAt: start}
			sum, detail := terminalText(ctx, &harness.Terminal{Err: ctx.Err()}, lastTool)
			p.rec(db.Activity{Kind: "result", Summary: firstLine(sum, 400), Detail: detail})
			return cliAttemptResult{}, false, ctx.Err()
		}
		return cliAttemptResult{reason: harness.ReasonModelError}, false, err
	}

	var (
		tbuf           strings.Builder
		tkind          string
		toolNames      = map[string]string{}
		lastTool       = &runTrace{startedAt: start}
		latestUsage    llm.Usage
		hasLatestUsage bool
		finalResult    *CLIResult
		savedSessionID string
	)

	flush := func() {
		if tbuf.Len() == 0 {
			return
		}
		s := strings.TrimSpace(tbuf.String())
		k := tkind
		tbuf.Reset()
		tkind = ""
		if s != "" {
			p.rec(db.Activity{Kind: k, Summary: firstLine(s, 200), Detail: s})
		}
	}
	addDelta := func(kind, text string) {
		if text == "" {
			return
		}
		if tkind != "" && tkind != kind {
			flush()
		}
		tkind = kind
		tbuf.WriteString(text)
	}
	saveSession := func(sid string) {
		if p.store == nil {
			return
		}
		sid = strings.TrimSpace(sid)
		if sid == "" || sid == savedSessionID {
			return
		}
		if err := p.store.Save(p.sessionKey, CLISessionRecord{
			Format:    string(p.opts.Config.Format),
			SessionID: sid,
		}); err == nil {
			savedSessionID = sid
		}
	}

	parser := p.backend.NewParser()
	scanErr := scanProcessLines(h.Stdout(), func(line []byte) {
		events, parseErr := parser.ParseLine(line)
		if parseErr != nil {
			return
		}
		for _, ev := range events {
			switch ev.Kind {
			case "init":
				if !p.isResume && ev.SessionID != "" {
					saveSession(ev.SessionID)
				}
			case "thinking":
				addDelta("thinking", ev.Text)
			case "text":
				addDelta("text", ev.Text)
			case "tool_use":
				flush()
				toolNames[ev.ToolID] = ev.ToolName
				in := string(ev.ToolInput)
				lastTool.start(ev.ToolID, ev.ToolName, in)
				p.auditTrace.Start(ev.ToolID, ev.ToolName, ev.ToolInput)
				p.rec(db.Activity{
					Kind:      "tool_use",
					Tool:      ev.ToolName,
					ToolUseID: ev.ToolID,
					Summary:   ev.ToolName + " " + firstLine(in, 200),
					Detail:    in,
				})
			case "tool_result":
				flush()
				tName := toolNames[ev.ToolID]
				if tName == "" {
					tName = ev.ToolName
				}
				lastTool.done(ev.ToolID)
				p.auditTrace.Complete(ev.ToolID, ev.Text, ev.IsError)
				p.rec(db.Activity{
					Kind:      "tool_result",
					Tool:      tName,
					ToolUseID: ev.ToolID,
					IsError:   ev.IsError,
					Summary:   firstLine(ev.Text, 200),
					Detail:    ev.Text,
				})
			case "usage":
				if ev.Usage != nil {
					u := *ev.Usage
					latestUsage = u
					hasLatestUsage = true
					p.rec(db.Activity{
						Kind:             "usage",
						InputTokens:      &u.InputTokens,
						OutputTokens:     &u.OutputTokens,
						CacheReadTokens:  &u.CacheReadTokens,
						CacheWriteTokens: &u.CacheWriteTokens,
					})
				}
			case "result":
				if ev.Result != nil {
					cp := *ev.Result
					finalResult = &cp
				}
			}
		}
	})
	waitErr := h.Wait()
	latencyMs := int(time.Since(start).Milliseconds())

	totalUsage := latestUsage
	if finalResult != nil && hasNonZeroUsage(finalResult.Usage) {
		totalUsage = finalResult.Usage
	}

	if ctx.Err() != nil {
		flush()
		var reason harness.TerminalReason
		var finalText string
		if finalResult != nil {
			reason = finalResult.Reason
			finalText = finalResult.Text
		}
		var usagePtr *llm.Usage
		if hasLatestUsage || hasNonZeroUsage(totalUsage) {
			u := totalUsage
			usagePtr = &u
		}
		sum, detail := terminalText(ctx, &harness.Terminal{
			Reason: reason,
			Err:    ctx.Err(),
			Usage:  totalUsage,
			Text:   finalText,
		}, lastTool)
		p.rec(activityWithUsage(db.Activity{
			Kind:    "result",
			Summary: firstLine(sum, 400),
			Detail:  detail,
		}, usagePtr))
		llmrec.RecordCLIUsage(ctx, p.opts.UsagePG, p.modelUsed, p.opts.Config.ProfileName, p.opts.WorkerLabel, totalUsage, latencyMs, "error")
		return cliAttemptResult{text: finalText, reason: reason}, false, ctx.Err()
	}

	stderr := strings.TrimSpace(h.Stderr())
	if p.canRetry {
		isMissing := (finalResult != nil && finalResult.IsSessionMissing) ||
			((finalResult == nil || finalResult.IsError) && isCLISessionMissingError(stderr))
		if isMissing {
			_ = p.store.Delete(p.sessionKey)
			if hasNonZeroUsage(totalUsage) {
				llmrec.RecordCLIUsage(ctx, p.opts.UsagePG, p.modelUsed, p.opts.Config.ProfileName, p.opts.WorkerLabel, totalUsage, latencyMs, "error")
			}
			return cliAttemptResult{}, true, nil
		}
	}

	if finalResult == nil {
		flush()
		runErr := cliResultError(p.backend, nil, stderr, firstNonNil(waitErr, scanErr))
		var usagePtr *llm.Usage
		if hasLatestUsage || hasNonZeroUsage(totalUsage) {
			u := totalUsage
			usagePtr = &u
		}
		sum, detail := terminalText(ctx, &harness.Terminal{
			Reason: harness.ReasonModelError,
			Err:    runErr,
			Usage:  totalUsage,
		}, lastTool)
		p.rec(activityWithUsage(db.Activity{
			Kind:    "result",
			IsError: true,
			Summary: firstLine(sum, 400),
			Detail:  detail,
		}, usagePtr))
		llmrec.RecordCLIUsage(ctx, p.opts.UsagePG, p.modelUsed, p.opts.Config.ProfileName, p.opts.WorkerLabel, totalUsage, latencyMs, "error")
		return cliAttemptResult{reason: harness.ReasonModelError}, false, runErr
	}

	if !finalResult.IsSessionMissing {
		sid := strings.TrimSpace(finalResult.SessionID)
		if sid == "" && p.opts.Config.Format == FormatClaudeCLI && !finalResult.IsError {
			sid = p.sessionID
		}
		if sid != "" && (!finalResult.IsError || !p.isResume) {
			saveSession(sid)
		}
	}

	finalText := finalResult.Text
	reason := finalResult.Reason
	if tkind == "text" && strings.TrimSpace(tbuf.String()) == strings.TrimSpace(finalText) {
		tbuf.Reset()
		tkind = ""
	}
	if finalText == "" && !finalResult.IsError && tkind == "text" && tbuf.Len() > 0 {
		finalText = strings.TrimSpace(tbuf.String())
		tbuf.Reset()
		tkind = ""
	}
	flush()

	var runErr error
	status := "ok"
	if finalResult.IsError {
		status = "error"
		runErr = cliResultError(p.backend, finalResult, stderr, waitErr)
	}

	sum, detail := finalText, finalText
	if sum == "" || finalResult.IsError || reason == harness.ReasonAbortedTools || reason == harness.ReasonAbortedStreaming {
		tSum, tDetail := terminalText(ctx, &harness.Terminal{
			Reason: reason,
			Err:    runErr,
			Usage:  totalUsage,
			Turns:  finalResult.NumTurns,
			Text:   finalText,
		}, lastTool)
		if sum == "" {
			sum, detail = tSum, tDetail
		} else if finalResult.IsError {
			detail = tDetail
		}
	}

	u := totalUsage
	p.rec(db.Activity{
		Kind:             "result",
		IsError:          runErr != nil,
		Summary:          firstLine(sum, 400),
		Detail:           detail,
		InputTokens:      &u.InputTokens,
		OutputTokens:     &u.OutputTokens,
		CacheReadTokens:  &u.CacheReadTokens,
		CacheWriteTokens: &u.CacheWriteTokens,
	})
	llmrec.RecordCLIUsage(ctx, p.opts.UsagePG, p.modelUsed, p.opts.Config.ProfileName, p.opts.WorkerLabel, totalUsage, latencyMs, status)
	return cliAttemptResult{text: finalText, reason: reason}, false, runErr
}
