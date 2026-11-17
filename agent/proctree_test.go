package agent

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestProcTreeHelperProcess(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "1" {
		return
	}

	args := helperArgsAfterDoubleDash(os.Args)
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "missing helper mode")
		os.Exit(2)
	}

	switch args[0] {
	case "exit":
		code := 0
		if len(args) > 1 {
			var err error
			code, err = strconv.Atoi(args[1])
			if err != nil {
				fmt.Fprintf(os.Stderr, "bad exit code %q: %v\n", args[1], err)
				os.Exit(2)
			}
		}
		if len(args) > 2 && args[2] != "" {
			fmt.Fprint(os.Stdout, args[2])
		}
		if len(args) > 3 && args[3] != "" {
			fmt.Fprint(os.Stderr, args[3])
		}
		os.Exit(code)

	case "lines":
		fmt.Fprint(os.Stdout, "line-1\nline-2\r\n")
		fmt.Fprintf(os.Stdout, "long-%s\n", strings.Repeat("x", 100*1024))
		fmt.Fprint(os.Stdout, "tail-no-newline")
		fmt.Fprint(os.Stderr, "warn-stderr\n")
		os.Exit(0)

	case "bigstderr":
		payload := bytes.Repeat([]byte("E"), 128*1024)
		_, _ = os.Stderr.Write(payload)
		fmt.Fprint(os.Stdout, "done\n")
		os.Exit(0)

	case "inspect":
		wd, err := os.Getwd()
		if err != nil {
			fmt.Fprintf(os.Stderr, "getwd: %v\n", err)
			os.Exit(2)
		}
		stdinBytes, err := io.ReadAll(os.Stdin)
		if err != nil {
			fmt.Fprintf(os.Stderr, "read stdin: %v\n", err)
			os.Exit(2)
		}
		fmt.Fprintf(os.Stdout, "dir=%s|env=%s|stdin=%s", wd, os.Getenv("PROCTREE_TEST_VAR"), string(stdinBytes))
		os.Exit(0)

	case "child":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "missing pidFile")
			os.Exit(2)
		}
		pidFile := args[1]
		grand := exec.Command(os.Args[0], "-test.run=^TestProcTreeHelperProcess$", "--", "grandchild", pidFile)
		grand.Env = os.Environ()
		// Keep stdout and stderr pipe FDs inherited by the grandchild so tests
		// prove cancellation closes pipes across the entire process tree.
		grand.Stdout = os.Stdout
		grand.Stderr = os.Stderr
		if err := grand.Start(); err != nil {
			fmt.Fprintf(os.Stderr, "start grandchild: %v\n", err)
			os.Exit(2)
		}
		fmt.Fprintf(os.Stdout, "child_pid=%d\ngrandchild_pid=%d\n", os.Getpid(), grand.Process.Pid)
		time.Sleep(60 * time.Second)
		_ = grand.Wait()
		os.Exit(0)

	case "grandchild":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "missing pidFile")
			os.Exit(2)
		}
		pidFile := args[1]
		tmpFile := pidFile + ".tmp"
		if err := os.WriteFile(tmpFile, []byte(strconv.Itoa(os.Getpid())), 0o600); err != nil {
			fmt.Fprintf(os.Stderr, "write tmp pid: %v\n", err)
			os.Exit(2)
		}
		if err := os.Rename(tmpFile, pidFile); err != nil {
			fmt.Fprintf(os.Stderr, "rename pid file: %v\n", err)
			os.Exit(2)
		}
		fmt.Fprint(os.Stdout, "grandchild_ready\n")
		time.Sleep(60 * time.Second)
		os.Exit(0)

	default:
		fmt.Fprintf(os.Stderr, "unknown helper mode %q\n", args[0])
		os.Exit(2)
	}
}

func helperArgsAfterDoubleDash(osArgs []string) []string {
	for i, arg := range osArgs {
		if arg == "--" {
			return osArgs[i+1:]
		}
	}
	return nil
}

func helperSpec(helperArgs ...string) ProcessSpec {
	args := append([]string{"-test.run=^TestProcTreeHelperProcess$", "--"}, helperArgs...)
	return ProcessSpec{
		Path: os.Args[0],
		Args: args,
		Env:  append(os.Environ(), "GO_WANT_HELPER_PROCESS=1"),
	}
}

func waitForPIDFile(t *testing.T, pidFile string, timeout time.Duration) int {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		data, err := os.ReadFile(pidFile)
		if err == nil {
			pid, convErr := strconv.Atoi(strings.TrimSpace(string(data)))
			if convErr == nil && pid > 0 {
				return pid
			}
		}
		time.Sleep(15 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for pid file %s", pidFile)
	return 0
}

func waitUntilDead(t *testing.T, label string, pid int, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if !processAlive(pid) {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("%s (pid=%d) remained alive after %s", label, pid, timeout)
}

func cleanupPID(pid int) {
	if pid <= 0 || !processAlive(pid) {
		return
	}
	if p, err := os.FindProcess(pid); err == nil {
		_ = p.Kill()
	}
}

func TestProcTreeNormalExitAndExitCode(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	res, err := RunProcessTree(ctx, helperSpec("exit", "0", "hello stdout\n", "hello stderr\n"))
	if err != nil {
		t.Fatalf("expected nil error on exit 0, got %v", err)
	}
	if res.ExitCode != 0 {
		t.Fatalf("expected ExitCode 0, got %d", res.ExitCode)
	}
	if res.Stdout != "hello stdout\n" {
		t.Fatalf("unexpected stdout: %q", res.Stdout)
	}
	if res.Stderr != "hello stderr\n" {
		t.Fatalf("unexpected stderr: %q", res.Stderr)
	}

	resErr, err := RunProcessTree(ctx, helperSpec("exit", "42", "partial out", "fail err"))
	if err == nil {
		t.Fatalf("expected error on exit 42, got nil")
	}
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) {
		t.Fatalf("expected *exec.ExitError, got %T: %v", err, err)
	}
	if resErr.ExitCode != 42 {
		t.Fatalf("expected ExitCode 42, got %d", resErr.ExitCode)
	}
	if resErr.Stdout != "partial out" {
		t.Fatalf("unexpected stdout on non-zero exit: %q", resErr.Stdout)
	}
	if resErr.Stderr != "fail err" {
		t.Fatalf("unexpected stderr on non-zero exit: %q", resErr.Stderr)
	}
}

func TestProcTreeStreamLines(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	var lines []string
	err := StreamProcessTree(ctx, helperSpec("lines"), func(line []byte) {
		lines = append(lines, string(line))
	})
	if err != nil {
		t.Fatalf("StreamProcessTree failed: %v", err)
	}
	if len(lines) != 4 {
		t.Fatalf("expected 4 lines, got %d: %v", len(lines), lines)
	}
	if lines[0] != "line-1" || lines[1] != "line-2" {
		t.Fatalf("unexpected first lines: %v", lines[:2])
	}
	expectedLong := "long-" + strings.Repeat("x", 100*1024)
	if lines[2] != expectedLong {
		t.Fatalf("long line mismatch: got len=%d want len=%d", len(lines[2]), len(expectedLong))
	}
	if lines[3] != "tail-no-newline" {
		t.Fatalf("unexpected trailing line: %q", lines[3])
	}
}

func TestProcTreeCancelKillsChildAndGrandchild(t *testing.T) {
	pidFile := filepath.Join(t.TempDir(), "grandchild.pid")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	h, err := StartProcessTree(ctx, helperSpec("child", pidFile))
	if err != nil {
		t.Fatalf("StartProcessTree failed: %v", err)
	}

	childPID := h.Pid()
	var grandchildPID int
	t.Cleanup(func() {
		cleanupPID(grandchildPID)
		cleanupPID(childPID)
	})

	// Stream stdout lines until the grandchild confirms it is running and holding
	// the stdout/stderr pipes open.
	readyCh := make(chan struct{})
	scanErrCh := make(chan error, 1)
	go func() {
		scanner := bufio.NewScanner(h.Stdout())
		readySent := false
		for scanner.Scan() {
			if scanner.Text() == "grandchild_ready" && !readySent {
				readySent = true
				close(readyCh)
			}
		}
		scanErrCh <- scanner.Err()
	}()

	grandchildPID = waitForPIDFile(t, pidFile, 5*time.Second)

	select {
	case <-readyCh:
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for grandchild_ready on stdout")
	}

	if !processAlive(childPID) {
		t.Fatalf("expected child pid %d to be alive before cancel", childPID)
	}
	if !processAlive(grandchildPID) {
		t.Fatalf("expected grandchild pid %d to be alive before cancel", grandchildPID)
	}

	cancelStart := time.Now()
	cancel()

	// Stdout pipe must reach EOF promptly once the tree is killed, even though
	// the grandchild inherited stdout/stderr FDs.
	select {
	case scanErr := <-scanErrCh:
		if scanErr != nil {
			t.Fatalf("stdout scanner error after cancel: %v", scanErr)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("stdout pipe did not close promptly after ctx cancellation")
	}

	waitErr := h.Wait()
	if waitErr == nil {
		t.Fatal("expected non-nil error from Wait after cancellation")
	}
	if !errors.Is(waitErr, context.Canceled) {
		t.Fatalf("expected context.Canceled error, got %v", waitErr)
	}
	if elapsed := time.Since(cancelStart); elapsed > 4*time.Second {
		t.Fatalf("cancellation + Wait took too long: %s", elapsed)
	}

	waitUntilDead(t, "child", childPID, 2*time.Second)
	waitUntilDead(t, "grandchild", grandchildPID, 2*time.Second)
}

func TestProcTreeTimeoutKillsChildAndGrandchild(t *testing.T) {
	pidFile := filepath.Join(t.TempDir(), "grandchild-timeout.pid")
	ctx, cancel := context.WithTimeout(context.Background(), 600*time.Millisecond)
	defer cancel()

	var stdoutBuf bytes.Buffer
	spec := helperSpec("child", pidFile)
	spec.Stdout = &stdoutBuf

	res, err := RunProcessTree(ctx, spec)
	if err == nil {
		t.Fatal("expected deadline exceeded error, got nil")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected context.DeadlineExceeded, got %v", err)
	}
	if res.ExitCode == 0 {
		t.Fatalf("expected non-zero exit code on timeout, got %d", res.ExitCode)
	}

	data, readErr := os.ReadFile(pidFile)
	if readErr != nil {
		t.Fatalf("expected grandchild pid file %s to exist before timeout: %v", pidFile, readErr)
	}
	gpid, convErr := strconv.Atoi(strings.TrimSpace(string(data)))
	if convErr != nil || gpid <= 0 {
		t.Fatalf("invalid grandchild pid %q: %v", string(data), convErr)
	}
	t.Cleanup(func() { cleanupPID(gpid) })
	waitUntilDead(t, "grandchild", gpid, 2*time.Second)
}

func TestProcTreeKillMethod(t *testing.T) {
	pidFile := filepath.Join(t.TempDir(), "grandchild-kill.pid")
	ctx := context.Background()

	h, err := StartProcessTree(ctx, helperSpec("child", pidFile))
	if err != nil {
		t.Fatalf("StartProcessTree failed: %v", err)
	}

	childPID := h.Pid()
	var grandchildPID int
	t.Cleanup(func() {
		cleanupPID(grandchildPID)
		cleanupPID(childPID)
	})

	grandchildPID = waitForPIDFile(t, pidFile, 5*time.Second)

	go func() {
		_, _ = io.Copy(io.Discard, h.Stdout())
	}()

	if err := h.Kill(); err != nil {
		t.Fatalf("Kill failed: %v", err)
	}
	// Idempotent second Kill call must also succeed cleanly.
	_ = h.Kill()

	if err := h.Wait(); err == nil {
		t.Fatal("expected Wait error after Kill, got nil")
	}

	waitUntilDead(t, "child", childPID, 2*time.Second)
	waitUntilDead(t, "grandchild", grandchildPID, 2*time.Second)
}

func TestProcTreeBoundedStderr(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	res, err := RunProcessTree(ctx, helperSpec("bigstderr"))
	if err != nil {
		t.Fatalf("RunProcessTree failed: %v", err)
	}
	if res.ExitCode != 0 {
		t.Fatalf("expected exit 0, got %d", res.ExitCode)
	}
	if res.Stdout != "done\n" {
		t.Fatalf("unexpected stdout: %q", res.Stdout)
	}
	if len(res.Stderr) != defaultStderrLimit {
		t.Fatalf("expected stderr capped at %d bytes, got %d", defaultStderrLimit, len(res.Stderr))
	}
}

func TestProcTreeAlreadyCancelledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := StartProcessTree(ctx, helperSpec("exit", "0")); !errors.Is(err, context.Canceled) {
		t.Fatalf("StartProcessTree expected context.Canceled, got %v", err)
	}
	if _, err := RunProcessTree(ctx, helperSpec("exit", "0")); !errors.Is(err, context.Canceled) {
		t.Fatalf("RunProcessTree expected context.Canceled, got %v", err)
	}
}

func TestProcTreeSpecFields(t *testing.T) {
	t.Setenv("GO_WANT_HELPER_PROCESS", "1")
	t.Setenv("PROCTREE_TEST_VAR", "inherited-ok")

	workDir := t.TempDir()
	if resolved, err := filepath.EvalSymlinks(workDir); err == nil {
		workDir = resolved
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	res, err := RunProcessTree(ctx, ProcessSpec{
		Path:  os.Args[0],
		Args:  []string{"-test.run=^TestProcTreeHelperProcess$", "--", "inspect"},
		Dir:   workDir,
		Env:   nil, // must inherit os.Environ()
		Stdin: strings.NewReader("stdin-payload"),
	})
	if err != nil {
		t.Fatalf("RunProcessTree failed: %v", err)
	}

	want := fmt.Sprintf("dir=%s|env=inherited-ok|stdin=stdin-payload", workDir)
	if res.Stdout != want {
		t.Fatalf("got %q, want %q", res.Stdout, want)
	}
}

type blockingWriter struct {
	started chan struct{}
	unblock chan struct{}
	once    bool
}

func (w *blockingWriter) Write(p []byte) (int, error) {
	if !w.once {
		w.once = true
		close(w.started)
	}
	<-w.unblock
	return len(p), nil
}

func TestProcTreePostKillDrainTimeout(t *testing.T) {
	origTimeout := postKillDrainTimeout
	postKillDrainTimeout = 120 * time.Millisecond
	defer func() { postKillDrainTimeout = origTimeout }()

	bw := &blockingWriter{
		started: make(chan struct{}),
		unblock: make(chan struct{}),
	}
	t.Cleanup(func() { close(bw.unblock) })

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	spec := helperSpec("exit", "0", "block-me")
	spec.Stdout = bw

	h, err := StartProcessTree(ctx, spec)
	if err != nil {
		t.Fatalf("StartProcessTree failed: %v", err)
	}

	<-bw.started
	waitDone := make(chan error, 1)
	go func() {
		waitDone <- h.Wait()
	}()

	cancelStart := time.Now()
	cancel()

	select {
	case waitErr := <-waitDone:
		if !errors.Is(waitErr, context.Canceled) {
			t.Fatalf("expected context.Canceled on drain timeout, got %v", waitErr)
		}
		if elapsed := time.Since(cancelStart); elapsed > 2*time.Second {
			t.Fatalf("Wait exceeded bounded drain timeout: %s", elapsed)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("Wait deadlocked instead of honoring postKillDrainTimeout")
	}
}
