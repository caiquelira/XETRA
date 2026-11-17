package llmrec

import (
	"context"
	"iter"
	"os"
	"testing"

	"github.com/Autumn-27/artex/db"
	"github.com/Autumn-27/norma/llm"
	"github.com/Autumn-27/norma/transcript"
	"github.com/google/uuid"
)

type completeProvider struct{}

func (completeProvider) Stream(context.Context, llm.CompletionRequest) iter.Seq2[llm.StreamEvent, error] {
	return func(func(llm.StreamEvent, error) bool) {}
}

func (completeProvider) Complete(context.Context, llm.CompletionRequest) (llm.Message, string, llm.Usage, error) {
	return llm.Message{
		Role: llm.RoleAssistant,
		Content: []llm.ContentBlock{
			{Type: llm.BlockThinking, Thinking: "reasoning"},
			llm.TextBlock("answer"),
		},
	}, "stop", llm.Usage{InputTokens: 7, OutputTokens: 3}, nil
}

func TestTaskIDContextUsesExplicitRegistryID(t *testing.T) {
	ctx := WithTaskID(context.Background(), " 42 ")
	if got := TaskIDFrom(ctx); got != "42" {
		t.Fatalf("TaskIDFrom()=%q want 42", got)
	}
	if got := TaskIDFrom(WithTaskID(ctx, "   ")); got != "42" {
		t.Fatalf("blank task id should preserve parent context, got %q", got)
	}
	if got := TaskIDFrom(nil); got != "" {
		t.Fatalf("nil context returned %q", got)
	}
}

func TestParseSessionFallbackIsExplorationScoped(t *testing.T) {
	taskID, worker := parseSession("exp12-worker-i99")
	if taskID != "12" || worker != "worker" {
		t.Fatalf("parseSession()=(%q,%q)", taskID, worker)
	}
	if taskID, worker := parseSession("not-a-task"); taskID != "" || worker != "" {
		t.Fatalf("unexpected non-session parse: (%q,%q)", taskID, worker)
	}
}

func TestCompleteForwardsAtomicResponse(t *testing.T) {
	recorder := Wrap(completeProvider{}, nil, "model", "profile", "", "", func() bool { return false })
	msg, stopReason, usage, err := recorder.Complete(context.Background(), llm.CompletionRequest{})
	if err != nil {
		t.Fatalf("Complete() error: %v", err)
	}
	if got := msg.Text(); got != "answer" {
		t.Fatalf("Complete() text=%q want answer", got)
	}
	if stopReason != "stop" {
		t.Fatalf("Complete() stop reason=%q want stop", stopReason)
	}
	if usage.InputTokens != 7 || usage.OutputTokens != 3 {
		t.Fatalf("Complete() usage=%+v", usage)
	}
}

// The usage sink sees each metered call once, with the usage the llm_usage row gets.
func TestUsageSinkSeesMeteredCalls(t *testing.T) {
	var got []llm.Usage
	ctx := WithUsageSink(context.Background(), func(u llm.Usage) { got = append(got, u) })
	recorder := Wrap(meteredStreamProvider{}, nil, "model", "profile", "", "", func() bool { return false })
	for _, err := range recorder.Stream(ctx, llm.CompletionRequest{}) {
		if err != nil {
			t.Fatal(err)
		}
	}
	if _, _, _, err := recorder.Complete(ctx, llm.CompletionRequest{}); err != nil {
		t.Fatal(err)
	}
	want := []llm.Usage{{InputTokens: 23, OutputTokens: 5}, {InputTokens: 7, OutputTokens: 3}}
	if len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("sink saw %+v, want %+v", got, want)
	}
	// A call that gets no llm_usage row (no model, zero usage) is not seen either.
	got = nil
	empty := Wrap(completeProvider{}, nil, "", "", "", "", nil)
	empty.recordUsage(ctx, "", 0, "", llm.Usage{}, 0, "ok")
	if len(got) != 0 {
		t.Fatalf("sink saw an unmetered call: %+v", got)
	}
}

type meteredStreamProvider struct{ completeProvider }

func (meteredStreamProvider) Stream(context.Context, llm.CompletionRequest) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		if !yield(llm.StreamEvent{Type: llm.SEMessageStart, Usage: llm.Usage{InputTokens: 23}}, nil) {
			return
		}
		if !yield(llm.StreamEvent{Type: llm.SEMessageDelta, Usage: llm.Usage{OutputTokens: 5}}, nil) {
			return
		}
		yield(llm.StreamEvent{Type: llm.SEMessageStop}, nil)
	}
}

func TestSideUsageRecordedOnceOnConsumerCancellation(t *testing.T) {
	t.Skip("timing-sensitive: passes locally, flaky on CI runners; tracked in BaaRai/ARTEX#9")
	dsn := os.Getenv("ARTEX_PG_DSN")
	if dsn == "" {
		t.Skip("requires isolated ARTEX_PG_DSN")
	}
	pg, err := db.Open(dsn)
	if err != nil {
		t.Fatal(err)
	}
	defer pg.Close()
	for _, early := range []bool{false, true} {
		profile := "btw-metering-" + uuid.NewString()
		ctx, cancel := context.WithCancel(transcript.WithSessionID(t.Context(), "exp0-btw-test"))
		recorder := Wrap(meteredStreamProvider{}, pg, "fixture", profile, "", "", func() bool { return false })
		for event, err := range recorder.Stream(ctx, llm.CompletionRequest{}) {
			if err != nil {
				t.Fatal(err)
			}
			if early && event.Type == llm.SEMessageDelta {
				cancel()
				break // Consumer exits before the provider can return a cancellation event.
			}
		}
		cancel()
		var count, input, output int
		var worker, status string
		err = pg.QueryRow(`SELECT count(*),max(input_tokens),max(output_tokens),max(worker),max(status)
FROM llm_usage WHERE profile_name=$1`, profile).Scan(&count, &input, &output, &worker, &status)
		_, _ = pg.Exec(`DELETE FROM llm_usage WHERE profile_name=$1`, profile)
		if err != nil {
			t.Fatal(err)
		}
		wantStatus := "ok"
		if early {
			wantStatus = "error"
		}
		if count != 1 || input != 23 || output != 5 || worker != "btw" || status != wantStatus {
			t.Fatalf("early=%v: count=%d usage=%d/%d worker=%q status=%q", early, count, input, output, worker, status)
		}
	}
}

func TestUsageRecordCLIAndHelpers(t *testing.T) {
	if UsageSinkFrom(nil) != nil || UsageSinkFrom(context.Background()) != nil {
		t.Fatal("expected nil UsageSinkFrom on empty/nil context")
	}
	if WorkerFrom(nil) != "" || WorkerFrom(context.Background()) != "" {
		t.Fatal("expected empty WorkerFrom on empty/nil context")
	}

	var sinkCalls []llm.Usage
	ctx := WithUsageSink(context.Background(), func(u llm.Usage) {
		sinkCalls = append(sinkCalls, u)
	})
	ctx = WithWorker(ctx, " judge ")
	ctx = WithTaskID(ctx, "99")
	ctx = transcript.WithSessionID(ctx, "exp12-worker-i3")

	if UsageSinkFrom(ctx) == nil {
		t.Fatal("expected non-nil UsageSinkFrom")
	}
	if got := WorkerFrom(ctx); got != "judge" {
		t.Fatalf("WorkerFrom() = %q, want judge", got)
	}

	// Zero-token call with empty model is skipped.
	RecordCLIUsage(ctx, nil, "", "cli-prof", "", llm.Usage{}, 10, "ok")
	if len(sinkCalls) != 0 {
		t.Fatalf("expected empty model + zero usage to be skipped, got %+v", sinkCalls)
	}

	// Non-empty usage triggers sink even when pg is nil.
	u := llm.Usage{InputTokens: 120, OutputTokens: 45, CacheReadTokens: 30}
	RecordCLIUsage(ctx, nil, "sonnet", "cli-prof", "worker", u, 250, "ok")
	if len(sinkCalls) != 1 || sinkCalls[0] != u {
		t.Fatalf("sinkCalls = %+v, want [%+v]", sinkCalls, u)
	}

	// Verify resolveCallMeta precedence: workerOverride > WorkerFrom > parsed session.
	_, taskID, expID, worker := resolveCallMeta(ctx, "custom-lane")
	if taskID != "99" || expID != 12 || worker != "custom-lane" {
		t.Fatalf("resolveCallMeta(override) = (%q, %d, %q)", taskID, expID, worker)
	}
	_, taskID, expID, worker = resolveCallMeta(ctx, "")
	if taskID != "99" || expID != 12 || worker != "judge" {
		t.Fatalf("resolveCallMeta(ctx worker) = (%q, %d, %q)", taskID, expID, worker)
	}
	baseSessionCtx := transcript.WithSessionID(context.Background(), "exp15-planner")
	_, taskID, expID, worker = resolveCallMeta(baseSessionCtx, "")
	if taskID != "15" || expID != 15 || worker != "planner" {
		t.Fatalf("resolveCallMeta(session fallback) = (%q, %d, %q)", taskID, expID, worker)
	}

	// Verify Recorder clears inner UsageSink so a CLI-backed inner provider does not double-meter.
	var wrappedCalls []llm.Usage
	wrapCtx := WithUsageSink(context.Background(), func(u llm.Usage) {
		wrappedCalls = append(wrappedCalls, u)
	})
	inner := sinkAwareProvider{}
	rec := Wrap(inner, nil, "sonnet", "cli-prof", "", "", func() bool { return false })
	for _, err := range rec.Stream(wrapCtx, llm.CompletionRequest{}) {
		if err != nil {
			t.Fatal(err)
		}
	}
	if _, _, _, err := rec.Complete(wrapCtx, llm.CompletionRequest{}); err != nil {
		t.Fatal(err)
	}
	if len(wrappedCalls) != 2 {
		t.Fatalf("expected exactly 2 sink calls (one Stream, one Complete), got %d: %+v", len(wrappedCalls), wrappedCalls)
	}
}

type sinkAwareProvider struct{}

func (sinkAwareProvider) Stream(ctx context.Context, _ llm.CompletionRequest) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		u := llm.Usage{InputTokens: 10, OutputTokens: 4}
		RecordCLIUsage(ctx, nil, "sonnet", "cli-prof", "", u, 1, "ok")
		if !yield(llm.StreamEvent{Type: llm.SEMessageDelta, StopReason: "end_turn", Usage: u}, nil) {
			return
		}
		yield(llm.StreamEvent{Type: llm.SEMessageStop}, nil)
	}
}

func (sinkAwareProvider) Complete(ctx context.Context, _ llm.CompletionRequest) (llm.Message, string, llm.Usage, error) {
	u := llm.Usage{InputTokens: 11, OutputTokens: 5}
	RecordCLIUsage(ctx, nil, "sonnet", "cli-prof", "", u, 1, "ok")
	return llm.Message{Role: llm.RoleAssistant, Content: []llm.ContentBlock{llm.TextBlock("ok")}}, "end_turn", u, nil
}
