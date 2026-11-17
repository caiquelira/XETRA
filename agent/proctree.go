package agent

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"sync"
	"sync/atomic"
	"time"
)

const (
	// defaultStderrLimit caps captured stderr when ProcessSpec.Stderr is nil.
	defaultStderrLimit = 64 * 1024
)

// postKillDrainTimeout bounds how long Wait waits for stdout/stderr pipe
// draining after the process tree is killed, preventing Wait from hanging if
// an escaped descendant holds a pipe descriptor open.
var postKillDrainTimeout = 3 * time.Second

// ProcessSpec configures a subprocess executed as a killable process tree.
type ProcessSpec struct {
	Path   string
	Args   []string
	Dir    string
	Env    []string  // if nil, inherits os.Environ()
	Stdin  io.Reader // optional
	Stdout io.Writer // optional; if nil, ProcessHandle.Stdout exposes a pipe
	Stderr io.Writer // optional; if nil, stderr is captured into a bounded 64KB buffer
}

// ProcessResult holds the output and exit status returned by RunProcessTree.
type ProcessResult struct {
	Stdout   string
	Stderr   string
	ExitCode int
}

// ProcessHandle manages a running subprocess and its entire descendant tree.
type ProcessHandle struct {
	ctx        context.Context
	cmd        *exec.Cmd
	pid        int
	stdoutPipe io.ReadCloser
	stderrBuf  *cappedBuffer
	plat       *platformProcTree

	done         chan struct{}
	doneOnce     sync.Once
	killOnce     sync.Once
	killDone     chan struct{}
	killErr      error
	waitOnce     sync.Once
	waitErr      error
	cleanupOnce  sync.Once
	killedByCtx  atomic.Bool
	killedByUser atomic.Bool
}

type cappedBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
	max int
}

func (b *cappedBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.max > 0 {
		rem := b.max - b.buf.Len()
		if rem > 0 {
			if len(p) > rem {
				b.buf.Write(p[:rem])
			} else {
				b.buf.Write(p)
			}
		}
	}
	return len(p), nil
}

func (b *cappedBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

// StartProcessTree starts the command described by spec inside a platform-native
// process tree isolation boundary (Windows Job Object / Unix process group).
// When ctx is cancelled or expires, the entire process tree is terminated.
// Callers must call Wait() on the returned handle to release resources.
func StartProcessTree(ctx context.Context, spec ProcessSpec) (*ProcessHandle, error) {
	if ctx == nil {
		return nil, errors.New("proctree: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if spec.Path == "" {
		return nil, errors.New("proctree: empty executable path")
	}

	cmd := exec.Command(spec.Path, spec.Args...)
	cmd.Dir = spec.Dir
	if spec.Env != nil {
		cmd.Env = spec.Env
	}
	cmd.Stdin = spec.Stdin

	var stdoutPipe io.ReadCloser
	var err error
	if spec.Stdout != nil {
		cmd.Stdout = spec.Stdout
	} else {
		stdoutPipe, err = cmd.StdoutPipe()
		if err != nil {
			return nil, fmt.Errorf("proctree: stdout pipe: %w", err)
		}
	}

	var stderrBuf *cappedBuffer
	if spec.Stderr != nil {
		cmd.Stderr = spec.Stderr
	} else {
		stderrBuf = &cappedBuffer{max: defaultStderrLimit}
		cmd.Stderr = stderrBuf
	}

	plat, err := preparePlatformProcTree(cmd)
	if err != nil {
		return nil, err
	}

	if err := cmd.Start(); err != nil {
		plat.cleanup()
		return nil, err
	}

	pid := 0
	if cmd.Process != nil {
		pid = cmd.Process.Pid
	}
	plat.attach(pid)

	h := &ProcessHandle{
		ctx:        ctx,
		cmd:        cmd,
		pid:        pid,
		stdoutPipe: stdoutPipe,
		stderrBuf:  stderrBuf,
		plat:       plat,
		done:       make(chan struct{}),
		killDone:   make(chan struct{}),
	}

	go h.watchContext()
	return h, nil
}

// RunProcessTree executes spec to completion, capturing stdout (when spec.Stdout
// is nil) and bounded stderr, and returning the exit code and any error.
func RunProcessTree(ctx context.Context, spec ProcessSpec) (ProcessResult, error) {
	var stdoutBuf bytes.Buffer
	captureStdout := spec.Stdout == nil
	if captureStdout {
		spec.Stdout = &stdoutBuf
	}

	h, err := StartProcessTree(ctx, spec)
	if err != nil {
		return ProcessResult{ExitCode: -1}, err
	}

	waitErr := h.Wait()
	res := ProcessResult{
		Stderr:   h.Stderr(),
		ExitCode: exitCodeFromError(waitErr),
	}
	if captureStdout {
		res.Stdout = stdoutBuf.String()
	}
	return res, waitErr
}

// StreamProcessTree executes spec and invokes onLine for each stdout line (with
// trailing CR/LF stripped) while killing the entire process tree on ctx
// cancellation. spec.Stdout must be nil.
func StreamProcessTree(ctx context.Context, spec ProcessSpec, onLine func([]byte)) error {
	if spec.Stdout != nil {
		return errors.New("proctree: StreamProcessTree requires spec.Stdout to be nil")
	}
	h, err := StartProcessTree(ctx, spec)
	if err != nil {
		return err
	}

	readErr := scanProcessLines(h.Stdout(), onLine)
	waitErr := h.Wait()
	if waitErr != nil {
		return waitErr
	}
	return readErr
}

// Pid returns the root child process ID.
func (h *ProcessHandle) Pid() int {
	return h.pid
}

// Stdout returns the stdout pipe when ProcessSpec.Stdout was nil, or nil otherwise.
// Callers should read Stdout until EOF before calling Wait.
func (h *ProcessHandle) Stdout() io.ReadCloser {
	return h.stdoutPipe
}

// Stderr returns the bounded stderr output captured when ProcessSpec.Stderr was nil.
func (h *ProcessHandle) Stderr() string {
	if h.stderrBuf == nil {
		return ""
	}
	return h.stderrBuf.String()
}

// Kill terminates the entire process tree idempotently.
func (h *ProcessHandle) Kill() error {
	if !h.killedByCtx.Load() {
		h.killedByUser.Store(true)
	}
	return h.killTree()
}

// Wait waits for the root process to exit and for pipes to drain, bounding the
// post-kill drain so escaped descendants cannot deadlock Wait.
func (h *ProcessHandle) Wait() error {
	h.waitOnce.Do(func() {
		defer h.markDone()

		waitCh := make(chan error, 1)
		go func() {
			waitCh <- h.cmd.Wait()
		}()

		var rawErr error
		var timedOut bool

		select {
		case rawErr = <-waitCh:
		case <-h.killDone:
			select {
			case rawErr = <-waitCh:
			default:
				timer := time.NewTimer(postKillDrainTimeout)
				defer timer.Stop()
				select {
				case rawErr = <-waitCh:
				case <-timer.C:
					timedOut = true
				}
			}
		}

		h.cleanup()

		if timedOut {
			if h.killedByCtx.Load() && h.ctx.Err() != nil {
				h.waitErr = fmt.Errorf("process tree wait timed out after cancellation: %w", h.ctx.Err())
				return
			}
			h.waitErr = errors.New("process tree wait timed out after kill")
			return
		}
		if h.killedByCtx.Load() && rawErr != nil && h.ctx.Err() != nil {
			h.waitErr = fmt.Errorf("process terminated on context cancellation: %w", h.ctx.Err())
			return
		}
		h.waitErr = rawErr
	})
	return h.waitErr
}

func (h *ProcessHandle) watchContext() {
	select {
	case <-h.ctx.Done():
		select {
		case <-h.done:
			return
		default:
		}
		h.killedByCtx.Store(true)
		_ = h.killTree()
	case <-h.done:
	}
}

func (h *ProcessHandle) killTree() error {
	h.killOnce.Do(func() {
		defer close(h.killDone)
		h.killErr = h.plat.kill(h.cmd, h.done)
	})
	return h.killErr
}

func (h *ProcessHandle) cleanup() {
	h.cleanupOnce.Do(func() {
		h.plat.cleanup()
	})
}

func (h *ProcessHandle) markDone() {
	h.doneOnce.Do(func() {
		close(h.done)
	})
}

func scanProcessLines(r io.Reader, onLine func([]byte)) error {
	if r == nil {
		return nil
	}
	reader := bufio.NewReaderSize(r, 64*1024)
	for {
		line, err := reader.ReadBytes('\n')
		if len(line) > 0 {
			trimmed := bytes.TrimRight(line, "\r\n")
			if onLine != nil {
				onLine(trimmed)
			}
		}
		if err != nil {
			if errors.Is(err, io.EOF) {
				return nil
			}
			return err
		}
	}
}

func exitCodeFromError(err error) int {
	if err == nil {
		return 0
	}
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		return exitErr.ExitCode()
	}
	return -1
}
